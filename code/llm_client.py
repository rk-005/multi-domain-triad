from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class LLMClientError(RuntimeError):
    pass


class AnthropicClient:
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        throttle_seconds: float = 0.0,
        log_path: str | Path = "triage_run.log",
    ) -> None:
        load_dotenv()
        self.model = model
        self.throttle_seconds = throttle_seconds
        self._last_call_at = 0.0
        self._logger = self._build_logger(Path(log_path))
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.enabled = bool(self.api_key)
        self._client = None

        if self.enabled:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise LLMClientError(
                    "The anthropic package is required to use the LLM client."
                ) from exc

            self._client = Anthropic(api_key=self.api_key)

    def _build_logger(self, log_path: Path) -> logging.Logger:
        logger = logging.getLogger("triage_run")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler(log_path, encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s | ticket=%(ticket_index)s | %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _log(self, ticket_index: Optional[int], message: str) -> None:
        extra = {"ticket_index": ticket_index if ticket_index is not None else "-"}
        self._logger.info(message, extra=extra)

    def _respect_throttle(self) -> None:
        if self.throttle_seconds <= 0:
            return

        elapsed = time.time() - self._last_call_at
        if elapsed < self.throttle_seconds:
            time.sleep(self.throttle_seconds - elapsed)

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code == 429:
            return True
        return "rate limit" in str(error).lower()

    @staticmethod
    def _extract_text(response: object) -> str:
        content = getattr(response, "content", [])
        text_parts: list[str] = []
        for block in content:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
        return "".join(text_parts).strip()

    def call(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        ticket_index: int | None = None,
    ) -> str:
        if not self.enabled or self._client is None:
            raise LLMClientError(
                "Anthropic client is not configured. Set ANTHROPIC_API_KEY in .env."
            )

        last_error: Exception | None = None
        for attempt in range(3):
            self._respect_throttle()
            self._log(
                ticket_index,
                (
                    f"request model={self.model} prompt_chars="
                    f"{len(system) + len(user)} max_tokens={max_tokens}"
                ),
            )
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                self._last_call_at = time.time()
                text = self._extract_text(response)
                self._log(ticket_index, f"response_chars={len(text)}")
                return text
            except Exception as exc:
                last_error = exc
                self._log(ticket_index, f"error={exc.__class__.__name__}: {exc}")
                if self._is_rate_limit_error(exc) and attempt < 2:
                    time.sleep(2**attempt)
                    continue
                break

        raise LLMClientError(f"Anthropic API call failed: {last_error}") from last_error


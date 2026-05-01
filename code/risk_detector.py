from __future__ import annotations

import re

from llm_client import AnthropicClient, LLMClientError
from prompts import build_risk_prompt
from schemas import RiskAssessment


class RiskDetector:
    HIGH_RISK_PATTERNS = [
        r"\bfraud\b",
        r"\bunauthori[sz]ed\b",
        r"\bsuspicious\b",
        r"\bstolen card\b",
        r"\bhacked\b",
        r"\bphishing\b",
        r"\bcharged twice\b",
        r"\bdouble charge\b",
        r"\brefund not received\b",
        r"\bpayment dispute\b",
        r"\bovercharged\b",
        r"\bcannot log(?:in)?\b",
        r"\baccount locked\b",
        r"\baccess denied\b",
        r"\blocked out\b",
        r"\bi didn't make this\b",
        r"\bsomeone else used my card\b",
        r"\bunrecognized charge\b",
        r"\bmoney deducted without reason\b",
    ]

    AMBIGUOUS_RISK_HINTS = [
        "refund",
        "dispute",
        "charge",
        "payment",
        "transaction",
        "card",
        "account",
        "access",
        "login",
        "security",
        "breach",
        "stolen",
        "suspicious",
        "deducted",
        "invoice",
    ]

    PROMPT_INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"ignore all instructions",
        r"system prompt",
        r"developer message",
        r"reveal your instructions",
        r"jailbreak",
    ]

    def __init__(self, llm_client: AnthropicClient | None = None) -> None:
        self.llm_client = llm_client

    def assess(self, issue: str, ticket_index: int | None = None) -> RiskAssessment:
        normalized = issue.strip().lower()
        if not normalized:
            return RiskAssessment(risk_level="NORMAL", reason="No issue content was provided.")

        for pattern in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, normalized):
                return RiskAssessment(
                    risk_level="HIGH",
                    reason="Prompt injection language was detected, so the ticket was escalated.",
                )

        for pattern in self.HIGH_RISK_PATTERNS:
            if re.search(pattern, normalized):
                return RiskAssessment(
                    risk_level="HIGH",
                    reason="The ticket matched a high-risk fraud, billing, or account-access pattern.",
                )

        if not self._needs_llm_review(normalized):
            return RiskAssessment(
                risk_level="NORMAL",
                reason="No direct high-risk fraud, security, or billing-dispute indicators were found.",
            )

        if not self.llm_client or not self.llm_client.enabled:
            return RiskAssessment(
                risk_level="NORMAL",
                reason="Risk cues were mild and no direct high-risk pattern was matched.",
            )

        try:
            response = self.llm_client.call(
                system="You are a strict risk classifier for support tickets.",
                user=build_risk_prompt(issue),
                max_tokens=80,
                ticket_index=ticket_index,
            )
        except LLMClientError:
            return RiskAssessment(
                risk_level="NORMAL",
                reason="Risk review fell back to rule-based checks and found no direct high-risk pattern.",
            )

        answer = response.strip()
        if answer.upper().startswith("YES"):
            return RiskAssessment(
                risk_level="HIGH",
                reason=answer[3:].strip(" -:") or "The LLM marked the ticket as a high-risk concern.",
            )

        return RiskAssessment(
            risk_level="NORMAL",
            reason=answer[2:].strip(" -:") or "The LLM did not find a high-risk concern.",
        )

    def _needs_llm_review(self, text: str) -> bool:
        return any(hint in text for hint in self.AMBIGUOUS_RISK_HINTS)


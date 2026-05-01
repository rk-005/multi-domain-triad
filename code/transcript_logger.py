from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

from schemas import TicketInput


@dataclass
class TranscriptLogger:
    path: Path

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "hackerrank_orchestrate" / "log.txt"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def start_run(self, input_path: Path, output_path: Path, corpus_summary: dict[str, int]) -> None:
        lines = [
            "Multi-Domain Support Triage Run",
            f"Input: {input_path}",
            f"Output: {output_path}",
            "Corpus coverage:",
        ]
        for domain, count in corpus_summary.items():
            lines.append(f"- {domain}: {count} articles")
        self._append(lines)

    def log_ticket(
        self,
        ticket_index: int,
        ticket: TicketInput,
        result: dict[str, str],
        trace: dict,
    ) -> None:
        lines = [
            "",
            f"Ticket #{ticket_index + 1}",
            f"Issue: {ticket.issue.strip()}",
            f"Subject: {ticket.subject.strip()}",
            f"Provided company: {ticket.company.strip() or 'None'}",
            f"Detected company: {trace.get('company', result.get('company', 'None'))}",
            f"Request type: {trace.get('request_type', result.get('request_type', ''))}",
            f"Product area: {trace.get('product_area', result.get('product_area', ''))}",
            f"Risk level: {trace.get('risk_level', 'UNKNOWN')}",
            f"Status: {result.get('status', '')}",
            f"Decision note: {trace.get('decision_note', '')}",
        ]
        sources = trace.get("retrieved_sources", [])
        if sources:
            lines.append("Retrieved sources:")
            for source in sources:
                lines.append(
                    f"- [{source.get('domain')}] {source.get('title')} | {source.get('url')} | score={source.get('score')}"
                )
        else:
            lines.append("Retrieved sources: none")

        lines.append(f"Response: {result.get('response', '')}")
        lines.append(f"Justification: {result.get('justification', '')}")
        self._append(lines)

    def finish(self, total: int, replied: int, escalated: int, skipped: int) -> None:
        self._append(
            [
                "",
                "Run summary",
                f"Total tickets processed: {total}",
                f"Replied: {replied}",
                f"Escalated: {escalated}",
                f"Skipped rows: {skipped}",
            ]
        )

    def _append(self, lines: Iterable[str]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.write("\n")

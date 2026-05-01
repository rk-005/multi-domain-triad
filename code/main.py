from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from llm_client import AnthropicClient, LLMClientError
from retriever import SupportRetriever
from risk_detector import RiskDetector
from schemas import TicketInput
from scraper import import_local_corpus
from transcript_logger import TranscriptLogger
from triage_engine import TriageEngine


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus"
DEFAULT_INPUT_PATH = REPO_ROOT / "support_tickets" / "support_tickets.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "support_tickets" / "output.csv"
DEFAULT_LOG_PATH = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "hackerrank_orchestrate" / "log.txt"
DEFAULT_INDEX_PATH = REPO_ROOT / "retriever.index"
DEFAULT_META_PATH = REPO_ROOT / "retriever_meta.pkl"
SUPPORTED_DOMAINS = ("hackerrank", "claude", "visa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-domain support triage agent")
    parser.add_argument(
        "--import-corpus",
        type=str,
        help="Import a provided local corpus directory, JSON file, or ZIP into corpus/",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Input CSV path. Defaults to support_tickets/support_tickets.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output CSV path. Defaults to support_tickets/output.csv",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(DEFAULT_LOG_PATH),
        help="Transcript log path. Defaults to %%USERPROFILE%%\\hackerrank_orchestrate\\log.txt",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-ticket details")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directories() -> None:
    for path in [
        DEFAULT_CORPUS_DIR,
        DEFAULT_DATA_DIR,
        REPO_ROOT / "support_tickets",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def run_import_corpus(corpus_path: str, console: Console) -> None:
    results = import_local_corpus(corpus_path, output_dir=DEFAULT_CORPUS_DIR)
    table = Table(title="Imported Corpus")
    table.add_column("Domain")
    table.add_column("Articles", justify="right")
    for domain, count in results.items():
        table.add_row(domain, str(count))
    console.print(table)


def load_llm_client(ticket_count: int, console: Console) -> AnthropicClient | None:
    try:
        client = AnthropicClient(throttle_seconds=0.5 if ticket_count > 20 else 0.0)
        if not client.enabled:
            console.print(
                "[yellow]LLM client warning:[/yellow] ANTHROPIC_API_KEY is not set, "
                "so the agent will use extractive corpus-grounded replies when possible."
            )
        return client
    except LLMClientError as exc:
        console.print(f"[yellow]LLM client warning:[/yellow] {exc}")
        return None


def build_ticket(row: pd.Series) -> TicketInput:
    payload = {
        "issue": row.get("issue", ""),
        "subject": row.get("subject", ""),
        "company": row.get("company", "None"),
    }
    return TicketInput.model_validate(payload)


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_columns = {column: str(column).strip().lower() for column in df.columns}
    return df.rename(columns=normalized_columns)


def truncate(text: str, limit: int) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def print_verbose_ticket(console: Console, index: int, result: dict[str, str], risk_level: str) -> None:
    status_color = "green" if result["status"] == "replied" else "red"
    body = (
        f"Issue: {truncate(result['issue'], 80)}\n"
        f"Company: {result['company']} | Request Type: {result['request_type']} | Risk: {risk_level}\n"
        f"Status: [{status_color}]{result['status']}[/{status_color}]\n"
        f"Response: {truncate(result['response'], 140)}"
    )
    console.print(Panel(body, title=f"Ticket #{index}", expand=False))


def infer_risk_level(result: dict[str, str]) -> str:
    justification = result["justification"].lower()
    high_risk_signals = [
        "high-risk",
        "fraud",
        "billing",
        "account-access",
        "account access",
        "prompt injection",
        "unauthorized",
        "security",
    ]
    return "HIGH" if any(signal in justification for signal in high_risk_signals) else "NORMAL"


def print_summary(console: Console, results_df: pd.DataFrame, skipped_rows: int) -> None:
    summary = Table(title="Run Summary")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Total tickets processed", str(len(results_df)))
    summary.add_row("Replied", str((results_df["status"] == "replied").sum()))
    summary.add_row("Escalated", str((results_df["status"] == "escalated").sum()))
    summary.add_row("Skipped rows", str(skipped_rows))
    console.print(summary)

    company_table = Table(title="Breakdown by Company")
    company_table.add_column("Company")
    company_table.add_column("Count", justify="right")
    for company, count in results_df["company"].value_counts(dropna=False).items():
        company_table.add_row(str(company), str(count))
    console.print(company_table)

    request_table = Table(title="Breakdown by Request Type")
    request_table.add_column("Request Type")
    request_table.add_column("Count", justify="right")
    for request_type, count in results_df["request_type"].value_counts(dropna=False).items():
        request_table.add_row(str(request_type), str(count))
    console.print(request_table)


def load_corpus_article_counts(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> dict[str, int]:
    summary = {"HackerRank": 0, "Claude": 0, "Visa": 0}
    file_map = {
        "HackerRank": corpus_dir / "hackerrank.json",
        "Claude": corpus_dir / "claude.json",
        "Visa": corpus_dir / "visa.json",
    }
    for domain, path in file_map.items():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, list):
            summary[domain] = len(payload)
    return summary


def _latest_mtime_ns(root: Path) -> int:
    latest = 0
    if not root.exists():
        return latest

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            latest = max(latest, path.stat().st_mtime_ns)
        except OSError:
            continue
    return latest


def _looks_like_starter_data_dir(path: Path) -> bool:
    return path.exists() and all((path / domain).exists() for domain in SUPPORTED_DOMAINS)


def ensure_local_corpus_ready(console: Console, corpus_dir: Path = DEFAULT_CORPUS_DIR) -> dict[str, int]:
    starter_data_dir = DEFAULT_DATA_DIR
    legacy_data_dir = corpus_dir / "data"
    source_dir = None

    if _looks_like_starter_data_dir(starter_data_dir):
        source_dir = starter_data_dir
    elif _looks_like_starter_data_dir(legacy_data_dir):
        source_dir = legacy_data_dir

    current_counts = load_corpus_article_counts(corpus_dir)
    if source_dir is None:
        return current_counts

    expected_files = {
        "HackerRank": corpus_dir / "hackerrank.json",
        "Claude": corpus_dir / "claude.json",
        "Visa": corpus_dir / "visa.json",
    }
    raw_latest_mtime = _latest_mtime_ns(source_dir)
    imported_latest_mtime = max(
        (path.stat().st_mtime_ns for path in expected_files.values() if path.exists()),
        default=0,
    )

    needs_import = any(not path.exists() for path in expected_files.values()) or sum(current_counts.values()) == 0
    if raw_latest_mtime and raw_latest_mtime > imported_latest_mtime:
        needs_import = True

    if not needs_import:
        return current_counts

    console.print(f"[cyan]Importing local corpus from[/cyan] {source_dir}")
    imported_counts = import_local_corpus(source_dir, output_dir=corpus_dir)
    table = Table(title="Local Corpus Sync")
    table.add_column("Domain")
    table.add_column("Articles", justify="right")
    for domain, count in imported_counts.items():
        table.add_row(domain, str(count))
    console.print(table)
    return imported_counts


def process_tickets(
    input_path: Path,
    output_path: Path,
    verbose: bool,
    console: Console,
    log_path: Path,
) -> int:
    if not input_path.exists():
        console.print(f"[red]Input file not found:[/red] {input_path}")
        return 1

    df = normalize_input_columns(pd.read_csv(input_path))
    if "issue" not in df.columns:
        console.print("[red]Input CSV must include an 'issue' column.[/red]")
        return 1

    if "subject" not in df.columns:
        df["subject"] = ""
    if "company" not in df.columns:
        df["company"] = "None"

    corpus_counts = ensure_local_corpus_ready(console)
    llm_client = load_llm_client(len(df), console)
    retriever = SupportRetriever(
        corpus_dir=DEFAULT_CORPUS_DIR,
        index_path=DEFAULT_INDEX_PATH,
        meta_path=DEFAULT_META_PATH,
    )
    if not retriever.chunks:
        console.print("[yellow]Warning:[/yellow] No corpus chunks were loaded. Tickets may escalate as out of scope.")

    risk_detector = RiskDetector(llm_client=llm_client)
    engine = TriageEngine(retriever=retriever, llm_client=llm_client, risk_detector=risk_detector)
    transcript = TranscriptLogger(log_path)
    transcript.start_run(input_path, output_path, corpus_counts)

    results: list[dict[str, str]] = []
    skipped_rows = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task_id = progress.add_task("Processing tickets", total=len(df))
        for row_index, row in df.iterrows():
            try:
                ticket = build_ticket(row)
                result, trace = engine.triage(ticket, ticket_index=row_index)
                results.append(result)
                transcript.log_ticket(row_index, ticket, result, trace.to_dict())

                if verbose:
                    print_verbose_ticket(console, row_index + 1, result, infer_risk_level(result))
            except Exception as exc:
                skipped_rows += 1
                console.print(f"[yellow]Skipped row {row_index + 1}:[/yellow] {exc}")
            finally:
                progress.advance(task_id)

    if not results:
        console.print("[red]No ticket results were generated.[/red]")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    ordered_columns = [
        "issue",
        "subject",
        "company",
        "status",
        "product_area",
        "response",
        "justification",
        "request_type",
    ]
    results_df = results_df[ordered_columns]
    results_df.to_csv(output_path, index=False)

    transcript.finish(
        total=len(results_df),
        replied=int((results_df["status"] == "replied").sum()),
        escalated=int((results_df["status"] == "escalated").sum()),
        skipped=skipped_rows,
    )
    print_summary(console, results_df, skipped_rows)
    console.print(f"[green]Results written to[/green] {output_path}")
    console.print(f"[green]Transcript written to[/green] {log_path}")
    return 0


def main() -> int:
    configure_logging()
    ensure_directories()
    args = parse_args()
    console = Console()

    if args.import_corpus:
        run_import_corpus(args.import_corpus, console)

    input_path = Path(args.input)
    output_path = Path(args.output)
    log_path = Path(args.log_path)

    if args.input == str(DEFAULT_INPUT_PATH):
        console.print(f"[cyan]Using default input[/cyan] {input_path}")
    if args.output == str(DEFAULT_OUTPUT_PATH):
        console.print(f"[cyan]Using default output[/cyan] {output_path}")
    if args.log_path == str(DEFAULT_LOG_PATH):
        console.print(f"[cyan]Using default transcript path[/cyan] {log_path}")

    if args.import_corpus or input_path:
        return process_tickets(input_path, output_path, args.verbose, console, log_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

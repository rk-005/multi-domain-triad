# Support Triage Agent

This directory contains the HackerRank Orchestrate submission code.

## Requirements

- Python 3.10+
- Optional: `ANTHROPIC_API_KEY` in your environment or `.env`

## Install

```powershell
python -m pip install -r code\requirements.txt
```

## Run

From the repository root:

```powershell
python code\main.py
```

That command uses the starter-repo defaults:

- corpus source: `data/`
- input CSV: `support_tickets/support_tickets.csv`
- output CSV: `support_tickets/output.csv`
- transcript: `%USERPROFILE%\hackerrank_orchestrate\log.txt`

## Optional flags

```powershell
python code\main.py --verbose
python code\main.py --input "support_tickets\sample_support_tickets.csv" --output "support_tickets\output.csv"
python code\main.py --import-corpus "data"
python code\main.py --log-path "$env:USERPROFILE\hackerrank_orchestrate\log.txt"
```

## Packaging

The HackerRank submission should contain the `code/` directory only. From the repository root:

```powershell
Compress-Archive -Path code -DestinationPath code_submission.zip -Force
```

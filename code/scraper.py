from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TYPE_CHECKING
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser
from zipfile import ZipFile

import requests

if TYPE_CHECKING:
    from bs4 import BeautifulSoup


LOGGER = logging.getLogger(__name__)

SOURCE_CONFIG = [
    {
        "domain": "HackerRank",
        "home_url": "https://support.hackerrank.com/",
        "output_path": Path("corpus/hackerrank.json"),
    },
    {
        "domain": "Claude",
        "home_url": "https://support.claude.com/en/",
        "output_path": Path("corpus/claude.json"),
    },
    {
        "domain": "Visa",
        "home_url": "https://www.visa.co.in/support.html",
        "output_path": Path("corpus/visa.json"),
    },
]

SUPPORTED_DOMAINS = ("HackerRank", "Claude", "Visa")


@dataclass
class ScrapedArticle:
    domain: str
    title: str
    url: str
    content: str

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "title": self.title,
            "url": self.url,
            "content": self.content,
        }


class SupportScraper:
    def __init__(self, delay_seconds: float = 1.0, timeout: int = 20) -> None:
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; SupportTriageBot/1.0; "
                    "+https://example.local/support-triage)"
                )
            }
        )
        self._robots_cache: dict[str, RobotFileParser | None] = {}

    def _get_robot_parser(self, url: str) -> RobotFileParser | None:
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in self._robots_cache:
            parser = RobotFileParser()
            parser.set_url(urljoin(root, "/robots.txt"))
            try:
                parser.read()
                self._robots_cache[root] = parser
            except Exception:
                LOGGER.warning("Could not read robots.txt for %s; defaulting to allow.", root)
                self._robots_cache[root] = None
        return self._robots_cache[root]

    def _allowed_by_robots(self, url: str) -> bool:
        parser = self._get_robot_parser(url)
        if parser is None:
            return True
        try:
            return parser.can_fetch(self.session.headers["User-Agent"], url)
        except Exception:
            return True

    @staticmethod
    def _normalize_url(base_url: str, candidate: str) -> str | None:
        if not candidate:
            return None
        resolved = urljoin(base_url, candidate)
        cleaned, _ = urldefrag(resolved)
        parsed = urlparse(cleaned)
        if parsed.scheme not in {"http", "https"}:
            return None
        return cleaned

    @staticmethod
    def _is_internal_link(home_url: str, candidate_url: str) -> bool:
        home = urlparse(home_url)
        candidate = urlparse(candidate_url)
        if candidate.netloc != home.netloc:
            return False
        if candidate.path.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".svg", ".pdf")):
            return False
        return True

    @staticmethod
    def _extract_visible_text(soup: "BeautifulSoup") -> str:
        from bs4 import Comment

        for tag_name in [
            "script",
            "style",
            "noscript",
            "svg",
            "form",
            "header",
            "footer",
            "nav",
            "aside",
        ]:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        lines = [
            line.strip()
            for line in soup.get_text(separator="\n").splitlines()
            if line.strip()
        ]
        return "\n".join(lines)

    @staticmethod
    def _extract_title(soup: "BeautifulSoup") -> str:
        header = soup.find(["h1", "title"])
        if header and header.get_text(strip=True):
            return header.get_text(" ", strip=True)
        return "Untitled support page"

    def _fetch(self, url: str) -> str | None:
        if not self._allowed_by_robots(url):
            LOGGER.info("Skipping %s due to robots.txt rules.", url)
            return None

        time.sleep(self.delay_seconds)
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def crawl_domain(self, domain: str, home_url: str, max_depth: int = 2) -> list[dict]:
        queue: deque[tuple[str, int]] = deque([(home_url, 0)])
        visited: set[str] = set()
        articles: list[ScrapedArticle] = []

        while queue:
            current_url, depth = queue.popleft()
            if current_url in visited:
                continue
            visited.add(current_url)

            try:
                html = self._fetch(current_url)
            except requests.RequestException as exc:
                LOGGER.warning("Failed to fetch %s: %s", current_url, exc)
                continue

            if not html:
                continue

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            text = self._extract_visible_text(soup)
            if len(text) >= 120:
                articles.append(
                    ScrapedArticle(
                        domain=domain,
                        title=self._extract_title(soup),
                        url=current_url,
                        content=text,
                    )
                )

            if depth >= max_depth:
                continue

            for link in self._iter_links(soup, base_url=current_url):
                if link not in visited and self._is_internal_link(home_url, link):
                    queue.append((link, depth + 1))

        deduped = {article.url: article for article in articles}
        return [article.to_dict() for article in deduped.values()]

    def _iter_links(self, soup: "BeautifulSoup", base_url: str) -> Iterable[str]:
        for anchor in soup.find_all("a", href=True):
            normalized = self._normalize_url(base_url, anchor["href"])
            if normalized:
                yield normalized

    @staticmethod
    def save_corpus(articles: list[dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(articles, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def scrape_all(skip_existing: bool = True, max_depth: int = 2) -> dict[str, int]:
    scraper = SupportScraper()
    results: dict[str, int] = {}

    for source in SOURCE_CONFIG:
        output_path = source["output_path"]
        if skip_existing and _has_nonempty_corpus(output_path):
            LOGGER.info("Skipping %s because %s already exists.", source["domain"], output_path)
            results[source["domain"]] = -1
            continue

        articles = scraper.crawl_domain(
            domain=source["domain"],
            home_url=source["home_url"],
            max_depth=max_depth,
        )
        SupportScraper.save_corpus(articles, output_path)
        results[source["domain"]] = len(articles)

    return results


def import_local_corpus(source_path: str | Path, output_dir: str | Path = "corpus") -> dict[str, int]:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Corpus source does not exist: {source}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    records_by_domain: dict[str, list[dict]] = {domain: [] for domain in SUPPORTED_DOMAINS}

    if source.is_dir():
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            for record in _records_from_file(path):
                records_by_domain[record["domain"]].append(record)
    elif source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_path = Path(member.filename)
                if member_path.suffix.lower() not in {".json", ".txt", ".md", ".html", ".htm"}:
                    continue
                payload = archive.read(member).decode("utf-8", errors="ignore")
                for record in _records_from_payload(payload, member_path):
                    records_by_domain[record["domain"]].append(record)
    else:
        for record in _records_from_file(source):
            records_by_domain[record["domain"]].append(record)

    output_map = {
        "HackerRank": output_root / "hackerrank.json",
        "Claude": output_root / "claude.json",
        "Visa": output_root / "visa.json",
    }
    counts: dict[str, int] = {}
    for domain, output_path in output_map.items():
        deduped = _dedupe_records(records_by_domain[domain])
        output_path.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")
        counts[domain] = len(deduped)

    return counts


def _has_nonempty_corpus(output_path: Path) -> bool:
    if not output_path.exists():
        return False

    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return isinstance(data, list) and len(data) > 0


def _records_from_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return _records_from_json(payload, path)

    if suffix in {".txt", ".md", ".html", ".htm"}:
        payload = path.read_text(encoding="utf-8", errors="ignore")
        return _records_from_payload(payload, path)

    return []


def _records_from_payload(payload: str, path_hint: Path) -> list[dict]:
    metadata: dict[str, str] = {}
    text_payload = payload
    if path_hint.suffix.lower() in {".md", ".txt"}:
        metadata, text_payload = _parse_markdown_document(payload)

    domain = _infer_domain(path_hint.as_posix(), f"{metadata.get('title', '')}\n{text_payload}")
    if domain is None:
        return []

    if path_hint.suffix.lower() in {".html", ".htm"}:
        text = _html_to_text(payload)
    else:
        text = text_payload.strip()
    if len(text.strip()) < 80:
        return []

    return [
        {
            "domain": domain,
            "title": metadata.get("title") or _derive_title(path_hint, text),
            "url": metadata.get("source_url") or metadata.get("url") or _derive_url(path_hint),
            "content": text.strip(),
        }
    ]


def _records_from_json(payload: object, path_hint: Path) -> list[dict]:
    if isinstance(payload, list):
        records: list[dict] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_record(item, path_hint)
            if normalized:
                records.append(normalized)
        return records

    if isinstance(payload, dict):
        if {"domain", "title", "content"}.issubset(payload.keys()):
            normalized = _normalize_record(payload, path_hint)
            return [normalized] if normalized else []

        records: list[dict] = []
        for key, value in payload.items():
            if not isinstance(value, list):
                continue
            fallback_domain = _infer_domain(str(key), "")
            for item in value:
                if not isinstance(item, dict):
                    continue
                normalized = _normalize_record(item, path_hint, fallback_domain=fallback_domain)
                if normalized:
                    records.append(normalized)
        return records

    return []


def _normalize_record(item: dict, path_hint: Path, fallback_domain: str | None = None) -> dict | None:
    title = str(item.get("title") or item.get("name") or "").strip()
    content = str(item.get("content") or item.get("body") or item.get("text") or "").strip()
    url = str(item.get("url") or item.get("source_url") or item.get("link") or "").strip()
    raw_domain = str(item.get("domain") or item.get("company") or "").strip()
    domain = _canonical_domain(raw_domain) or fallback_domain or _infer_domain(path_hint.as_posix(), f"{title}\n{content}")

    if domain is None or len(content) < 80:
        return None

    return {
        "domain": domain,
        "title": title or _derive_title(path_hint, content),
        "url": url or _derive_url(path_hint),
        "content": content,
    }


def _dedupe_records(records: list[dict]) -> list[dict]:
    deduped: dict[tuple[str, str], dict] = {}
    for record in records:
        key = (record.get("url", "").strip().lower(), record.get("title", "").strip().lower())
        deduped[key] = record
    return list(deduped.values())


def _infer_domain(path_or_name: str, text: str) -> str | None:
    combined = f"{path_or_name}\n{text}".lower()
    if "hackerrank" in combined:
        return "HackerRank"
    if "claude" in combined or "anthropic" in combined:
        return "Claude"
    if "visa" in combined:
        return "Visa"
    return None


def _canonical_domain(value: str) -> str | None:
    mapping = {
        "hackerrank": "HackerRank",
        "claude": "Claude",
        "anthropic": "Claude",
        "visa": "Visa",
    }
    return mapping.get(value.strip().lower())


def _derive_title(path_hint: Path, text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first_line and len(first_line) <= 120:
        return first_line
    return path_hint.stem.replace("_", " ").replace("-", " ").strip() or "Imported support article"


def _derive_url(path_hint: Path) -> str:
    return f"local://{path_hint.as_posix()}"


def _html_to_text(payload: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(payload, "html.parser")
    for tag_name in ["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    lines = [
        line.strip()
        for line in soup.get_text(separator="\n").splitlines()
        if line.strip()
    ]
    return _repair_text("\n".join(lines))


def _parse_markdown_document(payload: str) -> tuple[dict[str, str], str]:
    metadata: dict[str, str] = {}
    body = payload

    if payload.startswith("---"):
        lines = payload.splitlines()
        closing_index = None
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                closing_index = index
                break
        if closing_index is not None:
            frontmatter_lines = lines[1:closing_index]
            body = "\n".join(lines[closing_index + 1 :])
            metadata = _parse_frontmatter(frontmatter_lines)

    return metadata, _markdown_to_text(body)


def _parse_frontmatter(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("- ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        cleaned_value = value.strip().strip('"').strip("'")
        if cleaned_value:
            metadata[key.strip()] = _repair_text(cleaned_value)
    return metadata


def _markdown_to_text(payload: str) -> str:
    text = payload
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _repair_text(text.strip())


def _repair_text(text: str) -> str:
    mojibake_markers = ("â€™", "â€œ", "â€", "â€“", "â€”", "â€‹", "ðŸ")
    if not any(marker in text for marker in mojibake_markers):
        return text

    try:
        repaired = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return text

    return repaired if repaired.strip() else text

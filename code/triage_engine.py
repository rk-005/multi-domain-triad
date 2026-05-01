from __future__ import annotations

from dataclasses import dataclass, field
import re
from collections import Counter

from llm_client import AnthropicClient, LLMClientError
from prompts import RESPONSE_SYSTEM_PROMPT, build_response_user_prompt
from retriever import RetrievedChunk, SupportRetriever
from risk_detector import RiskDetector
from schemas import TicketInput, TicketOutput


@dataclass
class TriageDecisionTrace:
    company: str = "None"
    request_type: str = ""
    product_area: str = ""
    risk_level: str = "NORMAL"
    decision_note: str = ""
    retrieved_sources: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "request_type": self.request_type,
            "product_area": self.product_area,
            "risk_level": self.risk_level,
            "decision_note": self.decision_note,
            "retrieved_sources": self.retrieved_sources,
        }


class TriageEngine:
    EXTRACTIVE_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "card",
        "company",
        "current",
        "currently",
        "different",
        "for",
        "from",
        "get",
        "has",
        "have",
        "help",
        "how",
        "hackerrank",
        "i",
        "issue",
        "if",
        "in",
        "is",
        "it",
        "claude",
        "me",
        "my",
        "need",
        "of",
        "on",
        "or",
        "our",
        "please",
        "process",
        "see",
        "start",
        "started",
        "support",
        "the",
        "this",
        "to",
        "us",
        "using",
        "visa",
        "was",
        "we",
        "working",
        "with",
        "you",
        "your",
        "able",
    }

    DOMAIN_KEYWORDS = {
        "HackerRank": [
            "assessment",
            "coding test",
            "submission",
            "leaderboard",
            "compile error",
            "interview",
            "candidate",
            "plagiarism",
            "test case",
            "hackerrank",
            "challenge",
        ],
        "Claude": [
            "claude",
            "anthropic",
            "api",
            "model",
            "tokens",
            "prompt",
            "context length",
            "api key",
            "sdk",
        ],
        "Visa": [
            "card",
            "payment",
            "transaction",
            "charged",
            "credit card",
            "debit card",
            "refund",
            "dispute",
            "fraud",
            "visa",
        ],
    }

    PRODUCT_AREA_KEYWORDS = {
        "screen": ["ui", "screen", "page", "display", "interface", "layout"],
        "community": ["forum", "discussion", "leaderboard", "community"],
        "privacy": ["privacy", "gdpr", "data", "personal info", "delete my data"],
        "billing": ["payment", "charge", "charged", "refund", "invoice", "billing", "dispute"],
        "account_access": ["login", "password", "locked", "2fa", "otp", "access", "sign in"],
        "assessment": ["test", "assessment", "coding challenge", "submission", "compile", "plagiarism"],
        "api": ["api", "sdk", "token", "tokens", "model", "rate limit", "api key", "prompt"],
        "travel_support": ["travel", "international", "foreign transaction", "abroad", "overseas"],
    }

    BUG_KEYWORDS = [
        "bug",
        "error",
        "crash",
        "broken",
        "not working",
        "fails",
        "failure",
        "exception",
        "compile error",
        "stuck",
    ]

    FEATURE_KEYWORDS = [
        "feature request",
        "can you add",
        "please add",
        "i wish there was",
        "it would be great if",
        "would love to see",
        "enhancement",
    ]

    INVALID_PATTERNS = [
        r"^\s*$",
        r"^(hi|hello|hey|thanks|thank you|ok|okay|cool|great)[!. ]*$",
    ]

    ENGLISH_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "can",
        "for",
        "from",
        "have",
        "help",
        "i",
        "in",
        "is",
        "it",
        "my",
        "not",
        "of",
        "on",
        "please",
        "the",
        "to",
        "with",
        "you",
    }

    FOREIGN_STOPWORDS = {
        "por",
        "para",
        "gracias",
        "hola",
        "bonjour",
        "merci",
        "und",
        "nicht",
        "こんにちは",
        "谢谢",
        "привет",
    }

    def __init__(
        self,
        retriever: SupportRetriever,
        llm_client: AnthropicClient | None = None,
        risk_detector: RiskDetector | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.risk_detector = risk_detector or RiskDetector(llm_client)

    def process_ticket(
        self,
        ticket: TicketInput,
        ticket_index: int | None = None,
    ) -> dict[str, str]:
        result, _ = self.triage(ticket, ticket_index=ticket_index)
        return result

    def triage(
        self,
        ticket: TicketInput,
        ticket_index: int | None = None,
    ) -> tuple[dict[str, str], TriageDecisionTrace]:
        issue = ticket.issue.strip()
        subject = ticket.subject.strip()
        combined_text = " ".join(part for part in [subject, issue] if part).strip()
        trace = TriageDecisionTrace()

        if not issue:
            trace.company = "None"
            trace.request_type = "invalid"
            trace.product_area = "general_support"
            trace.decision_note = "The ticket issue was empty."
            return self._build_invalid_output(
                ticket=ticket,
                company="None",
                product_area="general_support",
                justification="The ticket issue was empty, so it was treated as an invalid request.",
            ), trace

        if self._is_probably_non_english(combined_text):
            trace.company = self._canonical_company(ticket.company)
            trace.request_type = "product_issue"
            trace.product_area = "general_support"
            trace.decision_note = "The ticket appears to be non-English."
            return TicketOutput(
                issue=issue,
                subject=subject,
                company=self._canonical_company(ticket.company),
                status="escalated",
                product_area="general_support",
                response=(
                    "We were unable to process this request automatically because it appears "
                    "to be in a language outside the supported workflow. A support agent will review it."
                ),
                justification=(
                    "The ticket appears to be non-English based on simple language heuristics, "
                    "so it was escalated for human review."
                ),
                request_type="product_issue",
            ).model_dump(), trace

        request_type = self._classify_request_type(issue, subject)
        detected_domain, domain_note = self._detect_domain(ticket.company, combined_text)
        product_area = self._classify_product_area(combined_text)
        risk = self.risk_detector.assess(combined_text, ticket_index=ticket_index)

        trace.company = detected_domain
        trace.request_type = request_type
        trace.product_area = product_area
        trace.risk_level = risk.risk_level

        if request_type == "invalid":
            trace.decision_note = "The ticket content looked like a greeting or lacked a clear issue."
            return self._build_invalid_output(
                ticket=ticket,
                company=detected_domain,
                product_area=product_area,
                justification=(
                    "The ticket content looked like a greeting, appreciation, or otherwise lacked a clear issue."
                ),
            ), trace

        if risk.risk_level == "HIGH":
            trace.decision_note = risk.reason
            return TicketOutput(
                issue=issue,
                subject=subject,
                company=detected_domain,
                status="escalated",
                product_area=product_area,
                response=(
                    "This issue has been escalated to our support team. "
                    "A human agent will contact you shortly."
                ),
                justification=(f"{domain_note} {risk.reason}".strip()),
                request_type=request_type,
            ).model_dump(), trace

        if detected_domain != "None" and not self.retriever.has_domain_coverage(detected_domain):
            trace.decision_note = f"Support documentation for {detected_domain} was unavailable."
            return TicketOutput(
                issue=issue,
                subject=subject,
                company=detected_domain,
                status="escalated",
                product_area=product_area,
                response=(
                    "We were unable to identify a matching support article for this request. "
                    "A human agent will review your request."
                ),
                justification=(
                    f"{domain_note} Support documentation for {detected_domain} was unavailable or insufficient, so the ticket was escalated."
                ).strip(),
                request_type=request_type,
            ).model_dump(), trace

        retrieval_domain = None if detected_domain == "None" else detected_domain
        retrieved_chunks = self.retriever.retrieve(combined_text, domain=retrieval_domain, top_k=5)
        trace.retrieved_sources = self._trace_sources(retrieved_chunks)

        if detected_domain == "None":
            inferred_domain = self._infer_domain_from_retrieval(retrieved_chunks)
            if inferred_domain:
                detected_domain = inferred_domain
                trace.company = inferred_domain
                retrieval_domain = inferred_domain
                retrieved_chunks = self.retriever.retrieve(combined_text, domain=retrieval_domain, top_k=5)
                trace.retrieved_sources = self._trace_sources(retrieved_chunks)
                domain_note = (
                    f"The company field was unresolved, so the domain was inferred as {inferred_domain} from the support corpus."
                )

        if not self._has_clear_corpus_match(retrieved_chunks):
            trace.decision_note = "No strong support corpus match was found."
            return TicketOutput(
                issue=issue,
                subject=subject,
                company=detected_domain,
                status="escalated",
                product_area=product_area,
                response=(
                    "We were unable to identify a matching support article for this request. "
                    "A human agent will review your request."
                ),
                justification=(
                    f"{domain_note} No strong support corpus match was found, so the ticket was treated as out of scope."
                ).strip(),
                request_type=request_type,
            ).model_dump(), trace

        if not self.llm_client or not self.llm_client.enabled:
            extractive_response = self._generate_extractive_response(combined_text, retrieved_chunks)
            if extractive_response is None:
                trace.decision_note = "Relevant chunks were found but not enough for a clear grounded response."
                return TicketOutput(
                    issue=issue,
                    subject=subject,
                    company=detected_domain,
                    status="escalated",
                    product_area=product_area,
                    response=(
                        "I don't have enough information in our documentation to answer this. "
                        "A support agent will follow up."
                    ),
                    justification=(
                        f"{domain_note} Relevant corpus chunks were found, but they did not support a clear grounded response without additional documentation."
                    ).strip(),
                    request_type=request_type,
                ).model_dump(), trace

            trace.decision_note = "A grounded extractive response was generated from retrieved documentation."
            return TicketOutput(
                issue=issue,
                subject=subject,
                company=detected_domain,
                status="replied",
                product_area=product_area,
                response=extractive_response,
                justification=(
                    f"{domain_note} A grounded extractive response was generated directly from retrieved support documentation because the Anthropic client was unavailable."
                ).strip(),
                request_type=request_type,
            ).model_dump(), trace

        response = self._generate_grounded_response(
            combined_text,
            retrieved_chunks,
            ticket_index=ticket_index,
        )

        if response.startswith("I don't have enough information"):
            trace.decision_note = "Retrieved documentation did not fully cover the issue."
            return TicketOutput(
                issue=issue,
                subject=subject,
                company=detected_domain,
                status="escalated",
                product_area=product_area,
                response=(
                    "I don't have enough information in our documentation to answer this. "
                    "A support agent will follow up."
                ),
                justification=(
                    f"{domain_note} Retrieved documentation did not fully cover the issue, so the ticket was escalated."
                ).strip(),
                request_type=request_type,
            ).model_dump(), trace

        trace.decision_note = "A grounded LLM response was generated from retrieved documentation."
        return TicketOutput(
            issue=issue,
            subject=subject,
            company=detected_domain,
            status="replied",
            product_area=product_area,
            response=response,
            justification=(
                f"{domain_note} The ticket matched the {detected_domain} corpus and a grounded response was generated from retrieved support documentation."
            ).strip(),
            request_type=request_type,
        ).model_dump(), trace

    def _trace_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        return [
            {
                "domain": chunk.domain,
                "title": chunk.title,
                "url": chunk.url,
                "score": round(chunk.score, 4),
            }
            for chunk in chunks
        ]

    def _build_invalid_output(
        self,
        ticket: TicketInput,
        company: str,
        product_area: str,
        justification: str,
    ) -> dict[str, str]:
        return TicketOutput(
            issue=ticket.issue.strip(),
            subject=ticket.subject.strip(),
            company=company,
            status="replied",
            product_area=product_area,
            response="Happy to help! Could you provide more details about your issue?",
            justification=justification,
            request_type="invalid",
        ).model_dump()

    def _generate_grounded_response(
        self,
        issue: str,
        chunks: list[RetrievedChunk],
        ticket_index: int | None = None,
    ) -> str:
        prompt_chunks = [chunk.to_prompt_dict() for chunk in chunks]
        try:
            return self.llm_client.call(
                system=RESPONSE_SYSTEM_PROMPT,
                user=build_response_user_prompt(issue, prompt_chunks),
                max_tokens=220,
                ticket_index=ticket_index,
            )
        except LLMClientError:
            return (
                "I don't have enough information in our documentation to answer this. "
                "A support agent will follow up."
            )

    def _generate_extractive_response(
        self,
        issue: str,
        chunks: list[RetrievedChunk],
    ) -> str | None:
        query_terms = self._extract_query_terms(issue)
        if not query_terms:
            return None
        if not self._has_title_alignment(chunks, query_terms):
            return None

        candidate_sentences: list[tuple[float, str]] = []
        seen_sentences: set[str] = set()

        for chunk in chunks[:5]:
            for sentence in self._split_sentences(chunk.chunk_text):
                normalized = sentence.lower()
                if normalized in seen_sentences:
                    continue
                seen_sentences.add(normalized)

                if not self._is_viable_support_sentence(sentence):
                    continue

                score = self._score_sentence(sentence, query_terms)
                if score <= 0:
                    continue
                candidate_sentences.append((score + chunk.score, sentence.strip()))

        if not candidate_sentences:
            return None

        candidate_sentences.sort(key=lambda item: item[0], reverse=True)
        chosen_sentences: list[str] = []
        total_words = 0
        for _, sentence in candidate_sentences:
            sentence_words = len(sentence.split())
            if total_words + sentence_words > 70 and chosen_sentences:
                continue
            chosen_sentences.append(sentence)
            total_words += sentence_words
            if len(chosen_sentences) >= 2 or total_words >= 60:
                break

        if not chosen_sentences:
            return None

        if not any(self._is_actionable_support_sentence(sentence) for sentence in chosen_sentences):
            return None

        response = " ".join(chosen_sentences)
        response = re.sub(r"\s+", " ", response).strip()
        if len(response.split()) > 150:
            response = " ".join(response.split()[:150]).rstrip(".,;:") + "."
        if not self._response_covers_issue(response, query_terms):
            return None
        return response

    def _extract_query_terms(self, issue: str) -> set[str]:
        words = re.findall(r"[a-zA-Z0-9']+", issue.lower())
        return {
            word
            for word in words
            if len(word) > 2 and word not in self.EXTRACTIVE_STOPWORDS
        }

    def _split_sentences(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        if not normalized:
            return []
        parts = re.split(r"(?<=[.!?])\s+", normalized)
        return [part.strip(" -*") for part in parts if part.strip(" -*")]

    def _score_sentence(self, sentence: str, query_terms: set[str]) -> float:
        words = set(re.findall(r"[a-zA-Z0-9']+", sentence.lower()))
        overlap = len(words.intersection(query_terms))
        if overlap == 0:
            return 0.0

        score = float(overlap)
        if any(token in words for token in {"contact", "call", "issuer", "bank", "merchant", "account", "card"}):
            score += 1.0
        if any(
            token in words
            for token in {"check", "remove", "add", "manage", "report", "visit", "navigate", "invite", "login"}
        ):
            score += 0.75
        if any(
            phrase in sentence.lower()
            for phrase in {
                "please contact",
                "please visit",
                "what should i do",
                "you can find answers",
                "you can",
                "admins can",
                "navigate to",
                "check your",
                "contact your",
            }
        ):
            score += 0.75
        if sentence.endswith("?"):
            score -= 0.25
        return score

    def _is_viable_support_sentence(self, sentence: str) -> bool:
        lowered = sentence.lower().strip()
        if len(lowered.split()) < 6:
            return False
        if "|" in sentence:
            return False
        if lowered.startswith("last updated"):
            return False
        if lowered == "availability:":
            return False

        promotional_phrases = [
            "the magic of travel",
            "see the world",
            "technology",
            "innovation",
            "explore offers",
            "our promise",
            "leading by example",
            "trillion",
            "billion",
            "research",
            "card not present purchases are declined",
            "better authentication benefits everyone",
        ]
        if any(phrase in lowered for phrase in promotional_phrases):
            return False
        if "%" in sentence or "$" in sentence:
            return False

        return True

    def _is_actionable_support_sentence(self, sentence: str) -> bool:
        lowered = sentence.lower()
        support_markers = [
            "please contact",
            "please visit",
            "to dispute",
            "to review why",
            "to report",
            "visit our",
            "you can find answers",
            "your issuer or bank",
            "contact visa",
            "call visa",
            "freephone number",
            "you can",
            "admins can",
            "owners and primary owners",
            "navigate to",
            "check your",
            "contact your bank",
            "contact their it department",
            "sign up for",
            "remove a member",
            "add members",
            "report lost or stolen",
        ]
        return any(marker in lowered for marker in support_markers)

    def _response_covers_issue(self, response: str, query_terms: set[str]) -> bool:
        response_terms = set(re.findall(r"[a-zA-Z0-9']+", response.lower()))
        overlap = query_terms.intersection(response_terms)
        if len(overlap) >= 2:
            return True

        high_signal_terms = {
            "access",
            "admin",
            "api",
            "assessment",
            "atm",
            "billing",
            "certificate",
            "charge",
            "compile",
            "dispute",
            "fraud",
            "interview",
            "interviewer",
            "login",
            "merchant",
            "remove",
            "refund",
            "screen",
            "seat",
            "stolen",
            "submission",
            "token",
            "travel",
            "withdraw",
        }
        if len(overlap) >= 2 and overlap.intersection(high_signal_terms):
            return True
        return len(overlap) == 1 and bool(
            overlap.intersection(
                {
                    "access",
                    "admin",
                    "atm",
                    "charge",
                    "dispute",
                    "fraud",
                    "login",
                    "refund",
                    "seat",
                    "stolen",
                    "travel",
                    "withdraw",
                }
            )
        )

    def _has_title_alignment(self, chunks: list[RetrievedChunk], query_terms: set[str]) -> bool:
        title_signal_terms = {
            "access",
            "admin",
            "assessment",
            "billing",
            "charge",
            "dispute",
            "fraud",
            "interview",
            "interviewer",
            "login",
            "refund",
            "remove",
            "screen",
            "seat",
            "stolen",
            "submission",
            "team",
            "travel",
        }
        for chunk in chunks[:5]:
            title_terms = set(re.findall(r"[a-zA-Z0-9']+", chunk.title.lower()))
            overlap = query_terms.intersection(title_terms)
            if len(overlap) >= 2:
                return True
            if len(overlap) == 1 and overlap.intersection(title_signal_terms):
                return True
        return False

    def _detect_domain(self, company: str, text: str) -> tuple[str, str]:
        canonical_company = self._canonical_company(company)
        content_domain = self._score_domain_from_text(text)

        if canonical_company != "None" and content_domain != "None" and canonical_company != content_domain:
            return (
                content_domain,
                f"The provided company label conflicted with the ticket content, so {content_domain} was used.",
            )

        if content_domain != "None":
            if canonical_company == "None":
                return (
                    content_domain,
                    f"The domain was inferred from ticket content as {content_domain}.",
                )
            return canonical_company, f"The domain aligned with {canonical_company}."

        if canonical_company != "None":
            return canonical_company, f"The provided company field was used as {canonical_company}."

        return "None", "The ticket did not clearly match a supported domain."

    def _score_domain_from_text(self, text: str) -> str:
        lowered = text.lower()
        scores: Counter[str] = Counter()

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                scores[domain] += lowered.count(keyword.lower())

        if not scores:
            return "None"

        ranked = scores.most_common()
        best_domain, best_score = ranked[0]
        if best_score == 0:
            return "None"
        if len(ranked) > 1 and ranked[1][1] == best_score:
            return "None"
        return best_domain

    def _classify_request_type(self, issue: str, subject: str = "") -> str:
        lowered_issue = issue.strip().lower()
        if any(re.match(pattern, lowered_issue) for pattern in self.INVALID_PATTERNS):
            return "invalid"
        if self._is_generic_acknowledgement(lowered_issue):
            return "invalid"
        if self._looks_like_gibberish(lowered_issue):
            return "invalid"
        combined = " ".join(part for part in [subject.strip().lower(), lowered_issue] if part).strip()
        if any(keyword in combined for keyword in self.BUG_KEYWORDS):
            return "bug"
        if any(keyword in combined for keyword in self.FEATURE_KEYWORDS):
            return "feature_request"
        return "product_issue"

    def _classify_product_area(self, text: str) -> str:
        lowered = text.lower()
        scores: Counter[str] = Counter()
        for area, keywords in self.PRODUCT_AREA_KEYWORDS.items():
            for keyword in keywords:
                scores[area] += lowered.count(keyword.lower())

        if not scores:
            return "general_support"

        area, score = scores.most_common(1)[0]
        return area if score > 0 else "general_support"

    def _looks_like_gibberish(self, text: str) -> bool:
        if len(text) < 4:
            return True
        letters = [char for char in text if char.isalpha()]
        if not letters:
            return True
        vowel_ratio = sum(char in "aeiou" for char in letters) / len(letters)
        unique_ratio = len(set(text)) / max(len(text), 1)
        return vowel_ratio < 0.15 and unique_ratio < 0.6

    def _is_generic_acknowledgement(self, text: str) -> bool:
        normalized = re.sub(r"[^a-z ]+", " ", text)
        words = [word for word in normalized.split() if word]
        generic_words = {"hi", "hello", "hey", "thanks", "thank", "you", "ok", "okay", "cool", "great"}
        return bool(words) and len(words) <= 3 and all(word in generic_words for word in words)

    def _is_probably_non_english(self, text: str) -> bool:
        lowered = text.lower()
        word_tokens = set(re.findall(r"\w+", lowered, flags=re.UNICODE))
        if self.FOREIGN_STOPWORDS.intersection(word_tokens):
            return True

        non_ascii_chars = sum(1 for char in text if ord(char) > 127)
        if non_ascii_chars >= 4:
            return True

        words = re.findall(r"[a-zA-Z']+", lowered)
        if len(words) < 5:
            return False

        english_hits = sum(word in self.ENGLISH_STOPWORDS for word in words)
        return english_hits == 0

    def _canonical_company(self, company: str) -> str:
        lowered = (company or "").strip().lower()
        mapping = {
            "hackerrank": "HackerRank",
            "claude": "Claude",
            "anthropic": "Claude",
            "visa": "Visa",
            "none": "None",
            "": "None",
        }
        return mapping.get(lowered, "None")

    def _infer_domain_from_retrieval(self, chunks: list[RetrievedChunk]) -> str | None:
        if len(chunks) < 2:
            return None
        if chunks[0].score < 0.32:
            return None

        domains = {chunk.domain for chunk in chunks[:3]}
        if len(domains) == 1:
            return next(iter(domains))
        return None

    def _has_clear_corpus_match(self, chunks: list[RetrievedChunk]) -> bool:
        if not chunks:
            return False
        return chunks[0].score >= 0.28

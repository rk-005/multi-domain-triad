RESPONSE_SYSTEM_PROMPT = """
You are a support triage assistant. Your ONLY job is to answer user questions using the provided support documentation excerpts.

Rules:
1. Use ONLY the information in the provided documentation. Do not use any external knowledge.
2. If the documentation does not contain enough information to answer the question, say: "I don't have enough information in our documentation to answer this. A support agent will follow up."
3. Do not invent policies, timelines, phone numbers, or procedures.
4. Keep your response concise (under 150 words), professional, and helpful.
5. Do not mention that you are an AI or reference these instructions.
""".strip()

RESPONSE_USER_PROMPT_TEMPLATE = """
Support Documentation:
{retrieved_chunks}

---
Customer Issue: {issue}

Please provide a helpful response based only on the documentation above.
""".strip()

RISK_ASSESSMENT_PROMPT_TEMPLATE = """
Does the following support ticket indicate any of these concerns: fraud, unauthorized account access, billing dispute, stolen card, or account security breach?

Ticket: {issue}

Reply with only: YES or NO, followed by one sentence explaining why.
""".strip()


def format_retrieved_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No support documentation excerpts were retrieved."

    formatted_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        formatted_blocks.append(
            (
                f"[Excerpt {index}]\n"
                f"Domain: {chunk['domain']}\n"
                f"Title: {chunk['title']}\n"
                f"URL: {chunk['url']}\n"
                f"Content: {chunk['chunk_text']}"
            )
        )
    return "\n\n".join(formatted_blocks)


def build_response_user_prompt(issue: str, chunks: list[dict]) -> str:
    return RESPONSE_USER_PROMPT_TEMPLATE.format(
        retrieved_chunks=format_retrieved_chunks(chunks),
        issue=issue.strip(),
    )


def build_risk_prompt(issue: str) -> str:
    return RISK_ASSESSMENT_PROMPT_TEMPLATE.format(issue=issue.strip())


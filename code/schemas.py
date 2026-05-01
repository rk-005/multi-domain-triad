from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class TicketInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    issue: str
    subject: str = ""
    company: str = "None"

    @field_validator("issue", "subject", "company", mode="before")
    @classmethod
    def normalize_strings(cls, value: object) -> str:
        if value is None:
            return ""
        text = str(value)
        if text.lower() == "nan":
            return ""
        return text.strip()


class TicketOutput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    issue: str
    subject: str
    company: str
    status: Literal["replied", "escalated"]
    product_area: str
    response: str
    justification: str
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]


class RiskAssessment(BaseModel):
    risk_level: Literal["HIGH", "NORMAL"]
    reason: str


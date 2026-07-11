"""Request/response schemas.

Kept strict on purpose: field length limits and enums prevent
prompt-injection-by-volume and malformed input from reaching the LLM layer.
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class Language(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    KANNADA = "kn"
    MARATHI = "mr"


class HouseholdType(str, Enum):
    SINGLE = "single"
    FAMILY = "family"
    FAMILY_WITH_ELDERLY = "family_with_elderly"
    FAMILY_WITH_INFANTS = "family_with_infants"
    FAMILY_WITH_DISABILITY = "family_with_disability"


class DwellingType(str, Enum):
    APARTMENT = "apartment"
    INDEPENDENT_HOUSE = "independent_house"
    LOW_LYING_AREA = "low_lying_area"
    COASTAL = "coastal"
    HILLSIDE = "hillside"


class PreparednessRequest(BaseModel):
    location: str = Field(..., min_length=2, max_length=100)
    household_type: HouseholdType
    dwelling_type: DwellingType
    household_size: int = Field(..., ge=1, le=20)
    has_vehicle: bool = False
    language: Language = Language.ENGLISH
    special_needs: Optional[str] = Field(None, max_length=300)

    @field_validator("location", "special_needs")
    @classmethod
    def strip_and_check(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class ChecklistRequest(BaseModel):
    location: str = Field(..., min_length=2, max_length=100)
    dwelling_type: DwellingType
    household_size: int = Field(..., ge=1, le=20)
    language: Language = Language.ENGLISH


class TravelAdvisoryRequest(BaseModel):
    origin: str = Field(..., min_length=2, max_length=100)
    destination: str = Field(..., min_length=2, max_length=100)
    travel_date: str = Field(..., max_length=20)
    mode: str = Field(..., max_length=30)
    language: Language = Language.ENGLISH


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    target_language: Language


class AlertSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class Alert(BaseModel):
    id: str
    location: str
    severity: AlertSeverity
    headline: str
    message: str
    issued_at: str


class GeneratedPlan(BaseModel):
    summary: str
    action_items: List[str]
    checklist: List[str]
    emergency_contacts_hint: List[str]
    language: Language

"""Prompt construction for each feature.

Keeping prompts as pure functions (no I/O) makes them trivially unit-testable
and keeps main.py focused on HTTP concerns.
"""
from models import ChecklistRequest, PreparednessRequest, TravelAdvisoryRequest

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "mr": "Marathi",
}

PLAN_SYSTEM_INSTRUCTION = (
    "You are Monsoon Guard, a disaster-preparedness assistant for Indian monsoon "
    "conditions. You give practical, locally relevant, non-alarmist guidance. "
    "Always respond with ONLY valid JSON matching this schema: "
    '{"summary": str, "action_items": [str], "checklist": [str], '
    '"emergency_contacts_hint": [str]}. '
    "emergency_contacts_hint should be generic guidance like 'Save your local "
    "disaster helpline and nearest hospital number' rather than fabricated phone "
    "numbers, since you do not have verified live contact data."
)


def build_plan_prompt(req: PreparednessRequest) -> str:
    lang = LANGUAGE_NAMES[req.language.value]
    special = f"Special needs to account for: {req.special_needs}. " if req.special_needs else ""
    return (
        f"Create a personalized monsoon preparedness plan.\n"
        f"Location: {req.location}\n"
        f"Household type: {req.household_type.value}\n"
        f"Dwelling type: {req.dwelling_type.value}\n"
        f"Household size: {req.household_size}\n"
        f"Has vehicle: {req.has_vehicle}\n"
        f"{special}"
        f"Respond entirely in {lang}. "
        f"Give 5-8 concrete action_items and a 6-10 item checklist, tailored to the "
        f"dwelling type (e.g. low-lying areas need flood-specific advice, hillside "
        f"needs landslide advice)."
    )


CHECKLIST_SYSTEM_INSTRUCTION = (
    "You are Monsoon Guard, a disaster-preparedness assistant. Respond with ONLY "
    'valid JSON: {"checklist": [str]}. Each item should be a short, actionable '
    "checklist line (max ~12 words)."
)


def build_checklist_prompt(req: ChecklistRequest) -> str:
    lang = LANGUAGE_NAMES[req.language.value]
    return (
        f"Generate an emergency monsoon checklist for a household of "
        f"{req.household_size} in a {req.dwelling_type.value} in {req.location}. "
        f"Respond entirely in {lang}. Provide 8-12 items covering supplies, "
        f"documents, home safety, and communication."
    )


TRAVEL_SYSTEM_INSTRUCTION = (
    "You are Monsoon Guard, a travel-safety assistant for monsoon conditions in "
    'India. Respond with ONLY valid JSON: {"summary": str, "action_items": [str], '
    '"checklist": [str], "emergency_contacts_hint": [str]}.'
)


def build_travel_prompt(req: TravelAdvisoryRequest) -> str:
    lang = LANGUAGE_NAMES[req.language.value]
    return (
        f"Give a monsoon travel advisory for a trip from {req.origin} to "
        f"{req.destination} on {req.travel_date}, travelling by {req.mode}. "
        f"Respond entirely in {lang}. Note likely monsoon risks for this route "
        f"and mode (flooding, waterlogging, delays, visibility) and precautions."
    )


TRANSLATE_SYSTEM_INSTRUCTION = (
    "You translate short safety/preparedness messages accurately and simply, "
    'preserving urgency. Respond with ONLY valid JSON: {"translated_text": str}.'
)


def build_translate_prompt(text: str, target_language: str) -> str:
    lang = LANGUAGE_NAMES[target_language]
    return f"Translate the following text into {lang}:\n\n{text}"

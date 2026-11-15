"""Eligibility screening: hard structured checks, criterion-level evidence, optional LLM summary."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import requests

from .ctgov import Trial


@dataclass
class Check:
    name: str
    status: str  # "pass" | "fail" | "unknown"
    detail: str


def structured_checks(trial: Trial, age: float | None, sex: str | None) -> list[Check]:
    """Checks that can be decided from the registry's structured fields alone."""
    checks = []
    lo = f"{trial.min_age:g}" if trial.min_age is not None else "no minimum"
    hi = f"{trial.max_age:g}" if trial.max_age is not None else "no maximum"
    if age is None:
        checks.append(Check("Age", "unknown", f"Trial accepts {lo} to {hi} years"))
    else:
        ok = (trial.min_age is None or age >= trial.min_age) and \
             (trial.max_age is None or age <= trial.max_age)
        checks.append(Check("Age", "pass" if ok else "fail",
                            f"Patient {age:g}; trial accepts {lo} to {hi} years"))
    want = (trial.sex or "ALL").upper()
    if sex is None:
        checks.append(Check("Sex", "unknown", f"Trial accepts {want.lower()}"))
    else:
        ok = want == "ALL" or want == sex.upper()
        checks.append(Check("Sex", "pass" if ok else "fail",
                            f"Patient {sex.lower()}; trial accepts {want.lower()}"))
    return checks


_FEMALE = re.compile(r"\b(woman|female|girl|she|her|mother|pregnan\w*|G\dP\d)\b", re.I)
_MALE = re.compile(r"\b(man|male|boy|he|his|gentleman)\b", re.I)


def extract_demographics(note: str) -> tuple[float | None, str | None]:
    """Best-effort age (years) and sex from a free-text patient note."""
    age = None
    # Years first ("45-year-old", "32 yo"); infants only when explicitly "...-old",
    # so durations such as "3 days of fever" are not mistaken for an age.
    m = re.search(r"(\d{1,3})[- ]?(?:years?|yrs?|y/o|yo)\b", note, re.I)
    if m:
        age = float(m.group(1))
    else:
        m = re.search(r"(\d{1,3})[- ]?(month|week|day)s?[- ]old", note, re.I)
        if m:
            scale = {"month": 1 / 12, "week": 1 / 52.18, "day": 1 / 365.25}[m.group(2).lower()]
            age = float(m.group(1)) * scale
    f, mm = _FEMALE.search(note), _MALE.search(note)
    if f and (not mm or f.start() < mm.start()):
        sex = "FEMALE"
    elif mm:
        sex = "MALE"
    else:
        sex = None
    return age, sex


def is_excluded(checks: list[Check]) -> bool:
    return any(c.status == "fail" for c in checks)


def split_criteria(text: str) -> dict[str, list[str]]:
    """Split free-text criteria into inclusion and exclusion bullet lists."""
    parts = re.split(r"(?i)exclusion criteria\s*:?", text, maxsplit=1)
    inc = re.sub(r"(?i)inclusion criteria\s*:?", "", parts[0])
    exc = parts[1] if len(parts) > 1 else ""

    def bullets(block: str) -> list[str]:
        items = re.split(r"\n\s*(?:[-*•]|\d+[.)])\s+|\n{2,}", "\n" + block)
        return [re.sub(r"\s+", " ", i).strip() for i in items if len(i.strip()) > 8]

    return {"inclusion": bullets(inc), "exclusion": bullets(exc)}


def criterion_evidence(patient_text: str, criteria: dict[str, list[str]], embed_passages,
                       embed_query, top: int = 3) -> dict[str, list[tuple[str, float]]]:
    """Criteria most semantically related to the patient description, per section.
    These are pointers for a human reviewer, not eligibility decisions."""
    q = embed_query(patient_text)
    out = {}
    for section, items in criteria.items():
        if not items:
            out[section] = []
            continue
        sims = embed_passages(items) @ q
        idx = np.argsort(-sims)[:top]
        out[section] = [(items[i], float(sims[i])) for i in idx]
    return out


PROMPT = """You are assisting a clinical research coordinator. Using ONLY the information given,
explain in plain English whether this patient might qualify for the trial.

Patient: {patient}

Trial: {title}
Eligibility criteria:
{criteria}

Answer in at most 5 short bullet points: criteria the patient appears to meet, criteria that may
exclude them, and information that is missing. Do not invent patient facts. End with one line:
"Verdict: likely eligible / possibly eligible / likely ineligible"."""


def llm_explain(patient: str, trial: Trial) -> str | None:
    """Optional. Uses a local Ollama model if OLLAMA_MODEL is set, else Anthropic if
    ANTHROPIC_API_KEY is set, else returns None and the app shows the rule-based view only."""
    prompt = PROMPT.format(patient=patient, title=trial.title, criteria=trial.eligibility[:6000])
    try:
        if os.getenv("OLLAMA_MODEL"):
            r = requests.post(os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate",
                              json={"model": os.environ["OLLAMA_MODEL"], "prompt": prompt,
                                    "stream": False}, timeout=180)
            return r.json().get("response")
        if os.getenv("ANTHROPIC_API_KEY"):
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=120,
                              headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                       "anthropic-version": "2023-06-01"},
                              json={"model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
                                    "max_tokens": 500,
                                    "messages": [{"role": "user", "content": prompt}]})
            return r.json()["content"][0]["text"]
    except Exception:
        return None
    return None

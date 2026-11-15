"""Client for the public ClinicalTrials.gov API v2 (no key required)."""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

import requests

API = "https://clinicaltrials.gov/api/v2/studies"
_UNIT_YEARS = {"year": 1.0, "month": 1 / 12, "week": 1 / 52.18, "day": 1 / 365.25,
               "hour": 1 / 8766, "minute": 1 / 525960}


def parse_age(text: str | None) -> float | None:
    """'18 Years' -> 18.0, '6 Months' -> 0.5, None/'N/A' -> None."""
    if not text:
        return None
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)", text)
    if not m:
        return None
    unit = m.group(2).lower().rstrip("s")
    return float(m.group(1)) * _UNIT_YEARS[unit] if unit in _UNIT_YEARS else None


@dataclass
class Trial:
    nct_id: str
    title: str = ""
    summary: str = ""
    conditions: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    eligibility: str = ""
    min_age: float | None = None
    max_age: float | None = None
    sex: str = "ALL"
    healthy_volunteers: bool | None = None
    status: str = ""
    phase: str = ""
    countries: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"

    def text(self) -> str:
        """Single passage used for retrieval."""
        return (f"{self.title}. Conditions: {', '.join(self.conditions)}. "
                f"Interventions: {', '.join(self.interventions)}. {self.summary} "
                f"Eligibility: {self.eligibility}")

    def to_dict(self) -> dict:
        return asdict(self)


def parse_study(study: dict) -> Trial:
    p = study.get("protocolSection", {})
    ident, elig = p.get("identificationModule", {}), p.get("eligibilityModule", {})
    locs = p.get("contactsLocationsModule", {}).get("locations", [])
    return Trial(
        nct_id=ident.get("nctId", ""),
        title=ident.get("officialTitle") or ident.get("briefTitle", ""),
        summary=p.get("descriptionModule", {}).get("briefSummary", ""),
        conditions=p.get("conditionsModule", {}).get("conditions", []),
        interventions=[i.get("name", "") for i in
                       p.get("armsInterventionsModule", {}).get("interventions", [])],
        eligibility=elig.get("eligibilityCriteria", ""),
        min_age=parse_age(elig.get("minimumAge")),
        max_age=parse_age(elig.get("maximumAge")),
        sex=elig.get("sex", "ALL"),
        healthy_volunteers=elig.get("healthyVolunteers"),
        status=p.get("statusModule", {}).get("overallStatus", ""),
        phase=", ".join(p.get("designModule", {}).get("phases", [])),
        countries=sorted({l.get("country", "") for l in locs if l.get("country")}),
    )


def _get(params: dict, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(API, params=params, timeout=60,
                             headers={"User-Agent": "trialmatch-research/1.0"})
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ClinicalTrials.gov request failed: {params}")


def search(condition: str = "", term: str = "", recruiting_only: bool = True,
           country: str = "", max_results: int = 200) -> list[Trial]:
    """Live search. `condition` and `term` use the registry's own query syntax."""
    params = {"pageSize": min(max_results, 1000), "countTotal": "false"}
    if condition:
        params["query.cond"] = condition
    if term:
        params["query.term"] = term
    if country:
        params["query.locn"] = country
    if recruiting_only:
        params["filter.overallStatus"] = "RECRUITING|NOT_YET_RECRUITING"
    out: list[Trial] = []
    while len(out) < max_results:
        data = _get(params)
        out += [parse_study(s) for s in data.get("studies", [])]
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    return out[:max_results]


def fetch_by_ids(nct_ids: list[str], batch: int = 100, progress=None) -> dict[str, Trial]:
    """Fetch specific trials; IDs the registry no longer serves are simply absent."""
    found: dict[str, Trial] = {}
    for i in range(0, len(nct_ids), batch):
        chunk = nct_ids[i:i + batch]
        data = _get({"filter.ids": "|".join(chunk), "pageSize": batch})
        for s in data.get("studies", []):
            t = parse_study(s)
            found[t.nct_id] = t
        if progress:
            progress(min(i + batch, len(nct_ids)), len(nct_ids))
    return found

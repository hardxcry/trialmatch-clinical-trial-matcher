import numpy as np
import pytest

from trialmatch import ir_metrics as IR
from trialmatch.ctgov import Trial, parse_age, parse_study
from trialmatch.eligibility import (extract_demographics, is_excluded, split_criteria,
                                    structured_checks)
from trialmatch.retrieval import rrf, tokenize


@pytest.mark.parametrize("text,expected", [("18 Years", 18), ("6 Months", 0.5), (None, None),
                                           ("N/A", None), ("65 Years", 65)])
def test_parse_age(text, expected):
    got = parse_age(text)
    assert got is None if expected is None else got == pytest.approx(expected)


def test_parse_study_extracts_fields():
    study = {"protocolSection": {
        "identificationModule": {"nctId": "NCT00000001", "briefTitle": "A trial"},
        "eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria:\n- adults",
                              "minimumAge": "18 Years", "sex": "FEMALE"},
        "conditionsModule": {"conditions": ["Asthma"]},
        "statusModule": {"overallStatus": "RECRUITING"},
        "contactsLocationsModule": {"locations": [{"country": "India"}, {"country": "India"}]}}}
    t = parse_study(study)
    assert (t.nct_id, t.min_age, t.max_age, t.sex) == ("NCT00000001", 18, None, "FEMALE")
    assert t.countries == ["India"] and "Asthma" in t.text()


def test_structured_checks_age_and_sex():
    t = Trial("NCT1", min_age=18, max_age=65, sex="FEMALE")
    assert not is_excluded(structured_checks(t, 40, "FEMALE"))
    assert is_excluded(structured_checks(t, 70, "FEMALE"))
    assert is_excluded(structured_checks(t, 40, "MALE"))
    assert not is_excluded(structured_checks(t, None, None))  # unknown never excludes


@pytest.mark.parametrize("note,age,sex", [
    ("Patient is a 45-year-old man with a history of astrocytoma", 45, "MALE"),
    ("A 32 yo woman presents with headache", 32, "FEMALE"),
    ("An 8-month-old boy with fever", 8 / 12, "MALE"),
    ("Patient with diabetes", None, None)])
def test_extract_demographics(note, age, sex):
    a, s = extract_demographics(note)
    assert s == sex and (a is None if age is None else a == pytest.approx(age))


def test_split_criteria():
    c = split_criteria("Inclusion Criteria:\n- Age over 18 years\n- Confirmed diagnosis of asthma\n\n"
                       "Exclusion Criteria:\n- Pregnant or breastfeeding\n- Active smoker currently")
    assert len(c["inclusion"]) == 2 and len(c["exclusion"]) == 2
    assert "Pregnant" in c["exclusion"][0]


def test_rrf_prefers_consistently_high_item():
    a, b = np.array([3.0, 2.0, 1.0]), np.array([0.9, 0.1, 0.5])
    assert int(np.argmax(rrf([a, b]))) == 0


def test_ndcg_perfect_and_reversed():
    gains = np.array([2, 1, 0, 0])
    assert IR.ndcg_at_k(gains, gains, 10) == pytest.approx(1.0)
    assert IR.ndcg_at_k(gains[::-1], gains, 10) < 1.0
    assert IR.ndcg_at_k(np.zeros(4), np.zeros(4), 10) == 0.0


def test_precision_and_mrr():
    rel = np.array([0, 1, 0, 1])
    assert IR.precision_at_k(rel, 4) == 0.5
    assert IR.reciprocal_rank(rel) == 0.5
    assert IR.reciprocal_rank(np.zeros(3)) == 0.0


def test_randomization_test_detects_and_rejects():
    rng = np.random.default_rng(0)
    base = rng.random(60)
    assert IR.paired_randomization_p(base + 0.2, base) < 0.001
    assert IR.paired_randomization_p(base, base) == 1.0


def test_tokenize():
    assert tokenize("Non-small-cell LUNG cancer (NSCLC)") == ["non", "small", "cell", "lung", "cancer", "nsclc"]

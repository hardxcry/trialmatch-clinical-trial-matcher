# Dense retrieval with structured eligibility rules outperforms hybrid re-ranking for patient-to-trial matching

**Hariharan S**, Department of Biomedical Engineering

## Abstract

**Background.** Many clinical trials struggle to recruit, partly because finding trials a given
patient qualifies for means reading long free-text eligibility criteria. **Objective.** To
compare common retrieval strategies for ranking trials against a free-text patient description
and to test whether simple rules on structured registry fields add value. **Methods.** Six
configurations (BM25, a dense bi-encoder, reciprocal-rank fusion of both, cross-encoder
re-ranking, and age/sex rule-filtered variants) were evaluated on the TREC 2021 Clinical Trials
track: 75 patient topics with 35,810 graded expert judgments. **Results.** The dense bi-encoder
reached NDCG@10 0.511 (95% CI 0.463 to 0.558) against 0.202 for BM25. Fusing with BM25 (0.443)
and adding a cross-encoder (0.426) both reduced performance. Filtering on registry age and sex
limits raised the dense model to 0.539 (p = 0.004) and Precision@10 from 0.367 to 0.408.
**Conclusion.** For long clinical notes, a compact dense model plus two structured rules beat a
more elaborate hybrid pipeline. Component choices should be benchmarked, not assumed.

## 1. Introduction

Patient-to-trial matching is an information-retrieval problem with an unusual query: a
paragraph of clinical narrative rather than a few keywords. Current practice in retrieval
systems is to combine lexical and dense search and re-rank with a cross-encoder. This project
tests whether that recipe holds for clinical notes, and builds a working matcher on the best
configuration.

## 2. Methods

**Benchmark.** TREC 2021 Clinical Trials track (NIST). Each topic is a synthetic patient case
of 5 to 10 sentences. Assessors graded pooled trials as not relevant (0), excluded (1: relevant
condition but the patient fails the criteria) or eligible (2).

**Trial records.** All 26,162 judged trials were requested from the ClinicalTrials.gov API v2;
26,150 were available. Each trial is represented by its title, conditions, interventions,
brief summary and eligibility criteria.

**Protocol.** For each topic the systems re-rank that topic's judged pool (mean 477 trials).
This isolates ranking quality and needs no unjudged-document assumptions.

**Systems.**
1. BM25 (Okapi) over lower-cased alphanumeric tokens.
2. Dense: `BAAI/bge-small-en-v1.5` (33M parameters), cosine similarity, 512-token passages.
3. Hybrid: reciprocal-rank fusion of 1 and 2 (k = 60).
4. Hybrid + cross-encoder: top 50 from 3 re-scored by `ms-marco-MiniLM-L-6-v2`.
5. System 4 with trials demoted when the patient's age or sex, extracted from the note by
   regular expression, falls outside the registry's structured limits.
6. System 2 with the same filter.

**Metrics.** NDCG@10 with graded gains; Precision@10 and MRR counting only eligible trials.
Means over topics with bootstrap 95% CIs (5,000 resamples); paired randomisation tests
(20,000 permutations).

## 3. Results

| System | NDCG@10 | P@10 (eligible) | MRR (eligible) |
|---|---|---|---|
| BM25 | 0.202 | 0.124 | 0.262 |
| Dense | 0.511 | 0.367 | 0.642 |
| Hybrid (RRF) | 0.443 | 0.299 | 0.544 |
| Hybrid + cross-encoder | 0.426 | 0.295 | 0.486 |
| Hybrid + cross-encoder + filter | 0.469 | 0.353 | 0.539 |
| Dense + filter | 0.539 | 0.408 | 0.678 |

Confidence intervals and p-values: [RESULTS.md](RESULTS.md). Every neural system beat BM25
(p < 0.001). Dense beat hybrid + cross-encoder on NDCG@10 (p = 0.0005). The filter improved the
dense system on NDCG@10 (p = 0.004) and P@10 (p = 0.001); the MRR gain was borderline
(p = 0.048). The filter demoted 5,548 judged trials, 172 (3.1%) of which assessors had graded
eligible.

## 4. Discussion

**Why BM25 is weak here.** The query is a whole case history. Many of its terms (medications,
past procedures, vital signs) are incidental to the trial the patient needs, so term matching
rewards trials that mention the same incidental words.

**Why fusion and re-ranking hurt.** Rank fusion gives the weak lexical ranking equal weight, so
it pulls good dense results down. The cross-encoder was trained on short web search questions
and truncates input at 512 tokens, which is shorter than a patient note plus a trial record.
A domain-specific re-ranker or query summarisation might reverse this; that is untested here.

**Why two rules help.** Age and sex limits are stored as structured fields. A neural model must
infer them from text; a rule reads them exactly. The 3.1% wrongly demoted trials arise from
extraction errors on the note and from registry records edited after the 2021 assessments.

**Limitations.** (1) Judged-pool re-ranking is easier than full-corpus retrieval, so these
scores are not comparable with official TREC leaderboard numbers. (2) Trial records are current
versions, not the April 2021 snapshot the assessors saw. (3) Topics are synthetic English case
descriptions; real clinical notes are messier. (4) Only one dense model and one re-ranker were
tested. (5) The system ranks; it does not determine eligibility.

**Future work.** Biomedical encoders (MedCPT, PubMedBERT variants), LLM-based
criterion-by-criterion eligibility assessment, and evaluation on the TREC 2022 and 2023 topics.

## References

1. Roberts K, et al. Overview of the TREC 2021 Clinical Trials Track. NIST, 2021.
2. Robertson S, Zaragoza H. The probabilistic relevance framework: BM25 and beyond. 2009.
3. Xiao S, et al. C-Pack: packaged resources to advance general Chinese embedding (BGE). 2023.
4. Cormack GV, Clarke CLA, Buettcher S. Reciprocal rank fusion outperforms Condorcet and
   individual rank learning methods. SIGIR 2009.
5. Smucker MD, Allan J, Carterette B. A comparison of statistical significance tests for
   information retrieval evaluation. CIKM 2007.

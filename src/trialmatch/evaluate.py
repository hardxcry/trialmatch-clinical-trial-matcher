"""Benchmark the ranking strategies on the TREC 2021 Clinical Trials track.

Protocol: for each of the 75 patient topics, re-rank the pool of trials that NIST assessors
judged for that topic (0 = not relevant, 1 = excluded, 2 = eligible).

Run:  python -m trialmatch.evaluate
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

from . import ir_metrics as IR
from . import retrieval as R
from .ctgov import Trial, fetch_by_ids
from .eligibility import extract_demographics, is_excluded, structured_checks

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TOPICS_URL = "https://www.trec-cds.org/topics2021.xml"
QRELS_URL = "https://trec.nist.gov/data/trials/qrels2021.txt"
SYSTEMS = ["bm25", "dense", "hybrid_rrf", "hybrid_rerank", "hybrid_rerank_filter", "dense_filter"]
LABELS = {"bm25": "BM25", "dense": "Dense (bge-small)", "hybrid_rrf": "Hybrid (RRF)",
          "hybrid_rerank": "Hybrid + cross-encoder",
          "hybrid_rerank_filter": "Hybrid + cross-encoder + age/sex filter",
          "dense_filter": "Dense + age/sex filter"}


def _download(url: str, path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0 (trialmatch research)"})
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def load_benchmark() -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    xml = _download(TOPICS_URL, DATA / "topics2021.xml")
    topics = {n: re.sub(r"\s+", " ", t).strip()
              for n, t in re.findall(r'<topic number="(\d+)">(.*?)</topic>', xml, flags=re.S)}
    qrels: dict[str, dict[str, int]] = {}
    for line in _download(QRELS_URL, DATA / "qrels2021.txt").splitlines():
        t, _, doc, rel = line.split()
        qrels.setdefault(t, {})[doc] = int(rel)
    return topics, qrels


def load_trials(ids: list[str]) -> dict[str, Trial]:
    cache = DATA / "trials.jsonl"
    trials: dict[str, Trial] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            t = Trial(**json.loads(line))
            trials[t.nct_id] = t
    missing = [i for i in ids if i not in trials]
    tried = set((DATA / "unavailable.txt").read_text().split()) if (DATA / "unavailable.txt").exists() else set()
    missing = [i for i in missing if i not in tried]
    if missing:
        got = fetch_by_ids(missing, progress=lambda a, b: print(f"\rfetching trials {a}/{b}", end=""))
        print()
        trials.update(got)
        with cache.open("a", encoding="utf-8") as f:
            for t in got.values():
                f.write(json.dumps(t.to_dict()) + "\n")
        (DATA / "unavailable.txt").write_text("\n".join(sorted(tried | (set(missing) - set(got)))))
    return trials


def main() -> None:
    topics, qrels = load_benchmark()
    all_ids = sorted({d for q in qrels.values() for d in q})
    trials = load_trials(all_ids)
    ids = [i for i in all_ids if i in trials]
    pos = {d: i for i, d in enumerate(ids)}
    texts = [trials[i].text() for i in ids]
    print(f"topics {len(topics)} | judged trials {len(all_ids)} | retrieved {len(ids)}")

    emb_path = DATA / "embeddings.npy"
    if emb_path.exists() and np.load(emb_path, mmap_mode="r").shape[0] == len(ids):
        emb = np.load(emb_path)
    else:
        emb = R.embed_passages(texts, batch_size=128, show_progress=True)
        np.save(emb_path, emb)
    tokens = [R.tokenize(t) for t in texts]

    per_topic = {s: {"ndcg10": [], "p10": [], "mrr": []} for s in SYSTEMS}
    used_topics, n_filtered, n_filtered_eligible = [], 0, 0
    for tid, query in topics.items():
        pool = [d for d in qrels.get(tid, {}) if d in pos]
        gains = np.array([qrels[tid][d] for d in pool])
        if not len(pool) or gains.max() < 2:
            continue  # no eligible trial left in the pool; topic cannot be scored
        used_topics.append(tid)
        idx = [pos[d] for d in pool]
        ptexts = [texts[i] for i in idx]
        s_bm25 = R.bm25_scores(query, [tokens[i] for i in idx])
        s_dense = R.dense_scores(R.embed_query(query), emb[idx])
        s_rrf = R.rrf([s_bm25, s_dense])
        s_rer = R.rerank(query, ptexts, s_rrf, top_k=50)
        age, sex = extract_demographics(query)
        fail = np.array([is_excluded(structured_checks(trials[d], age, sex)) for d in pool])
        n_filtered += int(fail.sum())
        n_filtered_eligible += int((fail & (gains == 2)).sum())
        s_filt = np.where(fail, s_rer.min() - 1e6, s_rer)
        s_dfilt = np.where(fail, s_dense.min() - 1e6, s_dense)
        for name, s in zip(SYSTEMS, [s_bm25, s_dense, s_rrf, s_rer, s_filt, s_dfilt]):
            ranked = gains[np.argsort(-s, kind="stable")]
            per_topic[name]["ndcg10"].append(IR.ndcg_at_k(ranked, gains, 10))
            per_topic[name]["p10"].append(IR.precision_at_k(ranked == 2, 10))
            per_topic[name]["mrr"].append(IR.reciprocal_rank(ranked == 2))
        print(f"\rscored topic {len(used_topics)}", end="")
    print()

    judged = sum(len(q) for q in qrels.values())
    kept = sum(1 for t in used_topics for d in qrels[t] if d in pos)
    out = {
        "benchmark": {"topics_total": len(topics), "topics_scored": len(used_topics),
                      "judgments_total": judged, "unique_trials_judged": len(all_ids),
                      "unique_trials_retrieved": len(ids), "judgments_in_scored_pools": kept,
                      "mean_pool_size": kept / len(used_topics),
                      "trials_demoted_by_filter": n_filtered,
                      "eligible_trials_wrongly_demoted": n_filtered_eligible},
        "models": {"dense": R.DENSE_MODEL, "cross_encoder": R.RERANK_MODEL, "device": R.device()},
        "systems": {},
    }
    for s in SYSTEMS:
        out["systems"][s] = {"label": LABELS[s]}
        for m, vals in per_topic[s].items():
            lo, hi = IR.bootstrap_mean_ci(vals)
            out["systems"][s][m] = {"mean": float(np.mean(vals)), "ci95": [lo, hi],
                                    "p_vs_bm25": None if s == "bm25" else
                                    IR.paired_randomization_p(vals, per_topic["bm25"][m])}
    out["dense_filter_vs_dense_p"] = {m: IR.paired_randomization_p(
        per_topic["dense_filter"][m], per_topic["dense"][m]) for m in ["ndcg10", "p10", "mrr"]}
    out["dense_vs_hybrid_rerank_p"] = {m: IR.paired_randomization_p(
        per_topic["dense"][m], per_topic["hybrid_rerank"][m]) for m in ["ndcg10", "p10", "mrr"]}
    out["per_topic"] = {"topics": used_topics,
                        **{s: per_topic[s]["ndcg10"] for s in SYSTEMS}}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(out, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=140)
    colors = ["#94a3b8", "#60a5fa", "#2563eb", "#1e3a8a", "#0f766e", "#10b981"]
    for ax, (m, title) in zip(axes, [("ndcg10", "NDCG@10"), ("p10", "Precision@10 (eligible)"),
                                     ("mrr", "MRR (eligible)")]):
        means = [out["systems"][s][m]["mean"] for s in SYSTEMS]
        err = [[mu - out["systems"][s][m]["ci95"][0] for s, mu in zip(SYSTEMS, means)],
               [out["systems"][s][m]["ci95"][1] - mu for s, mu in zip(SYSTEMS, means)]]
        ax.bar(range(len(SYSTEMS)), means, yerr=err, color=colors, capsize=3)
        ax.set_xticks(range(len(SYSTEMS)), ["BM25", "Dense", "RRF", "RRF\n+CE", "RRF+CE\n+filter", "Dense\n+filter"], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (ROOT / "reports" / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "reports" / "figures" / "systems.png")

    L = ["# Results", "", "Generated by `python -m trialmatch.evaluate` from `reports/metrics.json`.", "",
         f"Benchmark: TREC 2021 Clinical Trials track. {len(used_topics)} of {len(topics)} topics scored; "
         f"{len(ids)} of {len(all_ids)} judged trials were still retrievable from ClinicalTrials.gov; "
         f"mean pool size {kept / len(used_topics):.0f} trials per topic.", "",
         "Mean over topics with bootstrap 95% CI. p: paired randomisation test against BM25.", "",
         "| System | NDCG@10 | p | P@10 (eligible) | p | MRR (eligible) | p |", "|---|---|---|---|---|---|---|"]
    for s in SYSTEMS:
        row = [LABELS[s]]
        for m in ["ndcg10", "p10", "mrr"]:
            v = out["systems"][s][m]
            row += [f"{v['mean']:.3f} ({v['ci95'][0]:.3f} to {v['ci95'][1]:.3f})",
                    "ref" if v["p_vs_bm25"] is None else
                    ("<0.0001" if v["p_vs_bm25"] < 1e-4 else f"{v['p_vs_bm25']:.4f}")]
        L.append("| " + " | ".join(row) + " |")
    L += ["", f"Age/sex filter: demoted {n_filtered} judged trials across all topics, of which "
              f"{n_filtered_eligible} had been judged eligible by assessors."]
    (ROOT / "reports" / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[6:]))


if __name__ == "__main__":
    main()

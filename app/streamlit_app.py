"""TrialMatch: find recruiting clinical trials for a patient described in plain language.

Run:  streamlit run app/streamlit_app.py
Research prototype. Trial eligibility must always be confirmed by the study team.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from trialmatch import retrieval as R
from trialmatch.ctgov import search
from trialmatch.eligibility import (criterion_evidence, extract_demographics, is_excluded,
                                    llm_explain, split_criteria, structured_checks)

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="TrialMatch", page_icon="🧬", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg,#0f172a 0%,#1e3a8a 55%,#2563eb 100%);
       padding: 1.6rem 2rem; border-radius: 18px; color:#fff; margin-bottom: 1.2rem;}
.hero h1 {margin:0; font-size:2rem; color:#fff;} .hero p {margin:.3rem 0 0; opacity:.9;}
.trial {border:1px solid rgba(127,127,127,.25); border-radius:14px; padding:1rem 1.2rem;
        margin-bottom:.9rem; background: rgba(127,127,127,.05);}
.trial h4 {margin:0 0 .35rem;} .trial a {text-decoration:none;}
.meta {font-size:.82rem; opacity:.75; margin-bottom:.5rem;}
.chip {display:inline-block; padding:.15rem .65rem; border-radius:999px; font-size:.78rem;
       font-weight:600; margin-right:.35rem;}
.pass {background:#dcfce7; color:#166534;} .fail {background:#fee2e2; color:#991b1b;}
.unknown {background:#e2e8f0; color:#334155;}
.bar {height:6px; border-radius:6px; background:rgba(127,127,127,.2); margin:.5rem 0;}
.bar > div {height:6px; border-radius:6px; background:linear-gradient(90deg,#60a5fa,#1e3a8a);}
.crit {font-size:.86rem; margin:.15rem 0; padding-left:.6rem; border-left:3px solid #94a3b8;}
.crit.ex {border-left-color:#f87171;} .crit.in {border-left-color:#4ade80;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the language model (first run downloads about 130 MB)...")
def warm_models():
    R.dense_model()
    return True


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(condition: str, country: str, recruiting: bool, n: int):
    return [t.to_dict() for t in search(condition=condition, country=country,
                                        recruiting_only=recruiting, max_results=n)]


EXAMPLE = ("58-year-old woman with type 2 diabetes for 12 years, HbA1c 8.9% on metformin and "
           "glimepiride, BMI 33, stage 2 chronic kidney disease, no history of heart failure. "
           "Not on insulin. Interested in new treatment options.")

st.markdown('<div class="hero"><h1>🧬 TrialMatch</h1><p>Describe a patient. Get recruiting '
            'clinical trials ranked by relevance, with eligibility evidence.</p></div>',
            unsafe_allow_html=True)
tab_match, tab_bench, tab_about = st.tabs(["Find trials", "Benchmark", "About"])

with tab_match:
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        st.subheader("Patient")
        condition = st.text_input("Main condition", "type 2 diabetes",
                                  help="Used to pull candidate trials from ClinicalTrials.gov")
        note = st.text_area("Clinical description", EXAMPLE, height=190)
        g_age, g_sex = extract_demographics(note)
        a, b = st.columns(2)
        age = a.number_input("Age (years)", 0.0, 120.0, float(g_age) if g_age is not None else 50.0)
        sex = b.selectbox("Sex", ["FEMALE", "MALE"], index=0 if g_sex != "MALE" else 1,
                          format_func=str.capitalize)
        country = st.text_input("Country (optional)", "India")
        recruiting = st.toggle("Recruiting trials only", True)
        n_cand = st.slider("Candidate trials to rank", 50, 300, 150, 50)
        go = st.button("Find matching trials", type="primary", use_container_width=True)
        st.caption("Do not enter names or other identifying details.")

    with right:
        if not go:
            st.info("Enter a patient description and press **Find matching trials**.")
        elif not condition.strip() or not note.strip():
            st.warning("Both a condition and a description are needed.")
        else:
            warm_models()
            from trialmatch.ctgov import Trial
            with st.spinner("Searching ClinicalTrials.gov..."):
                trials = [Trial(**d) for d in cached_search(condition, country, recruiting, n_cand)]
            if not trials:
                st.warning("No trials found. Try a broader condition or clear the country.")
            else:
                with st.spinner(f"Ranking {len(trials)} trials..."):
                    texts = [t.text() for t in trials]
                    # Dense retrieval + age/sex rules was the best configuration on the
                    # TREC 2021 benchmark (see reports/RESULTS.md), so the app uses it.
                    scores = R.dense_scores(R.embed_query(note), R.embed_passages(texts))
                    checks = [structured_checks(t, age, sex) for t in trials]
                order = [i for i in np.argsort(-scores) if not is_excluded(checks[i])]
                ruled_out = [i for i in np.argsort(-scores) if is_excluded(checks[i])]
                top = order[:10]
                st.subheader(f"Top {len(top)} of {len(trials)} trials")
                st.caption(f"{len(ruled_out)} trials ruled out by age or sex limits.")
                finite = scores[np.isfinite(scores)]
                lo, hi = float(finite.min()), float(finite.max())
                for rank, i in enumerate(top, 1):
                    t = trials[i]
                    pct = 100 * (scores[i] - lo) / (hi - lo + 1e-9)
                    chips = "".join(f'<span class="chip {c.status}">{c.name}: {html.escape(c.detail)}'
                                    f'</span>' for c in checks[i])
                    meta = " · ".join(x for x in [t.nct_id, t.phase or "Phase n/a",
                                                  t.status.replace("_", " ").title(),
                                                  ", ".join(t.countries[:4])] if x)
                    st.markdown(f'<div class="trial"><h4>{rank}. <a href="{t.url}" target="_blank">'
                                f'{html.escape(t.title)}</a></h4><div class="meta">{html.escape(meta)}'
                                f'</div><div class="bar"><div style="width:{pct:.0f}%"></div></div>'
                                f'{chips}</div>', unsafe_allow_html=True)
                    with st.expander("Eligibility evidence"):
                        ev = criterion_evidence(note, split_criteria(t.eligibility),
                                                R.embed_passages, R.embed_query)
                        c1, c2 = st.columns(2)
                        c1.markdown("**Most relevant inclusion criteria**")
                        for text, _ in ev["inclusion"]:
                            c1.markdown(f'<div class="crit in">{html.escape(text[:400])}</div>',
                                        unsafe_allow_html=True)
                        c2.markdown("**Most relevant exclusion criteria**")
                        for text, _ in ev["exclusion"]:
                            c2.markdown(f'<div class="crit ex">{html.escape(text[:400])}</div>',
                                        unsafe_allow_html=True)
                        if rank <= 3:
                            summary = llm_explain(note, t)
                            if summary:
                                st.markdown("**AI eligibility summary**")
                                st.write(summary)
                        st.caption("Pointers for a human reviewer, not an eligibility decision.")
                if ruled_out:
                    with st.expander(f"Ruled out ({len(ruled_out)})"):
                        st.dataframe(pd.DataFrame([{
                            "Trial": trials[i].nct_id, "Title": trials[i].title[:90],
                            "Reason": "; ".join(c.detail for c in checks[i] if c.status == "fail")}
                            for i in ruled_out[:50]]), use_container_width=True, hide_index=True)

with tab_bench:
    mpath = ROOT / "reports" / "metrics.json"
    if not mpath.exists():
        st.info("Run `python -m trialmatch.evaluate` to generate benchmark results.")
    else:
        m = json.loads(mpath.read_text())
        b = m["benchmark"]
        st.subheader("TREC 2021 Clinical Trials track")
        st.caption(f"{b['topics_scored']} patient topics, {b['judgments_in_scored_pools']:,} expert "
                   f"judgments, about {b['mean_pool_size']:.0f} candidate trials per patient.")
        best = max(m["systems"], key=lambda s: m["systems"][s]["ndcg10"]["mean"])
        cols = st.columns(3)
        for col, (k, label) in zip(cols, [("ndcg10", "NDCG@10"), ("p10", "Precision@10"), ("mrr", "MRR")]):
            v, base = m["systems"][best][k]["mean"], m["systems"]["bm25"][k]["mean"]
            col.metric(f"{label} (best system)", f"{v:.3f}", f"{v - base:+.3f} vs BM25")
        st.image(str(ROOT / "reports/figures/systems.png"))
        st.dataframe(pd.DataFrame({s["label"]: {"NDCG@10": s["ndcg10"]["mean"], "P@10": s["p10"]["mean"],
                                                "MRR": s["mrr"]["mean"]}
                                   for s in m["systems"].values()}).T.round(3), use_container_width=True)

with tab_about:
    st.markdown("""
**How it works.** Candidate trials come live from the ClinicalTrials.gov API. Each is scored
against the patient description with a neural sentence-embedding model, and age and sex limits
from the registry are applied as hard rules. This configuration beat keyword search, hybrid
fusion and cross-encoder re-ranking on the TREC 2021 benchmark (see the Benchmark tab).

**What it is not.** It does not decide eligibility. Free-text criteria need a clinician or
study coordinator. Do not enter identifiable patient information.

**Optional AI summary.** Set `OLLAMA_MODEL` (local) or `ANTHROPIC_API_KEY` before starting
the app to add a plain-English eligibility summary for the top three trials.
""")

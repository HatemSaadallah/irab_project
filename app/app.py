"""I'rāb-Guided Arabic Diacritization — Streamlit demo for HF Spaces.

Loads a trained model from HF Hub (or local path) and exposes the Predictor.
"""

import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch

# The package is pip-installed from pyproject.toml (see app/requirements.txt)
from irab_tashkeel.inference.predictor import Predictor


# -------- Page config --------
st.set_page_config(
    page_title="I'rāb-Guided Arabic Diacritization",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# RTL + Arabic font CSS
st.markdown("""
<style>
    .arabic {
        font-family: "Noto Naskh Arabic", "Amiri", "Cairo", serif;
        font-size: 26px;
        direction: rtl;
        text-align: right;
        line-height: 2;
    }
    .word-card {
        display: inline-block;
        margin: 6px;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)


# -------- Model loading --------
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "your-username/irab-tashkeel-model")
LOCAL_MODEL_CKPT = os.environ.get("MODEL_CKPT")  # e.g. runs/medium/best.pt


@st.cache_resource(show_spinner="Loading model …")
def load_predictor() -> Predictor:
    """Load Predictor from local path (MODEL_CKPT) or HF Hub (HF_MODEL_REPO)."""
    if LOCAL_MODEL_CKPT and Path(LOCAL_MODEL_CKPT).exists():
        return Predictor.from_checkpoint(LOCAL_MODEL_CKPT)

    # Fall back to HF Hub
    try:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo_id=HF_MODEL_REPO, filename="model.pt")
        return Predictor.from_checkpoint(ckpt)
    except Exception as e:
        st.error(
            f"Couldn't load model. Set $MODEL_CKPT to a local file, or "
            f"$HF_MODEL_REPO to your Hub repo. Error: {e}"
        )
        st.stop()


def role_color(role: str) -> str:
    palette = {
        "fiil":         "#FEEAE6",
        "N_marfu":      "#E1F5EE",
        "N_mansub":     "#FAEEDA",
        "ism_majrur":   "#EEEDFE",
        "mudaf_ilayh":  "#EEEDFE",
        "harf_jarr":    "#F1EFE8",
        "harf_atf":     "#F1EFE8",
        "harf_nafy":    "#F1EFE8",
        "mabni_noun":   "#FBEAF0",
        "other":        "#F7F7F9",
    }
    return palette.get(role, "#F7F7F9")


# -------- Sidebar --------
with st.sidebar:
    st.title("📖 I'rāb + Tashkīl")
    st.markdown("**Explainable Arabic diacritization**")
    st.markdown("---")
    st.markdown("""
A hybrid neural + rule-based system that:
- Adds diacritics (**tashkīl**)
- Labels each word's grammatical role (**i'rāb**)
- Detects orthographic and grammatical errors
- Explains each decision in Arabic & English
""")
    st.markdown("---")

    show_confidence = st.checkbox("Show confidence scores", value=True)
    show_en = st.checkbox("Show English labels", value=True)
    conf_threshold = st.slider("Low-confidence flag threshold", 0.0, 1.0, 0.6, 0.05)

    st.markdown("---")
    st.markdown("[GitHub](https://github.com/) · Built with character-level Transformer + QAC + Tashkeela + I3rab")


# -------- Main --------
st.title("I'rāb-Guided Arabic Diacritization")
st.caption("Tashkīl with a why — every diacritic comes with a grammatical justification")

predictor = load_predictor()
predictor.confidence_threshold = conf_threshold

examples = {
    "Simple verbal sentence": "ذهب الطالب إلى المدرسة",
    "Nominal sentence":       "العلم نور والجهل ظلام",
    "Iḍāfa construction":     "كتاب الطالب جديد",
    "Sentence with errors":   "ذهب الطالب الى المدرسه",
    "With adjective":         "قرأت كتابا مفيدا عن اللغة العربية",
    "kāna sentence (Tier 2)": "كان الطالب مجتهدا في دروسه",
    "Relative clause (Tier 3)": "قرأت الكتاب الذي أعجبني",
    "Custom":                 "",
}

col_ex, col_in = st.columns([1, 3])
with col_ex:
    example_choice = st.selectbox("Load example", list(examples.keys()))
with col_in:
    default = examples[example_choice]
    user_input = st.text_area(
        "Enter Arabic text (undiacritized or partially diacritized)",
        value=default, height=80,
    )

if st.button("Diacritize + analyze", type="primary") and user_input.strip():
    with st.spinner("Running inference…"):
        result = predictor.predict(user_input)

    # ---- Diacritized output ----
    st.subheader("Diacritized text")
    st.markdown(f"<div class='arabic'>{result.diacritized}</div>", unsafe_allow_html=True)

    # ---- Tier info ----
    if result.tier > 1:
        tier_messages = {
            2: "⚠️ Tier 2 detected (kāna/inna/jussive). Neural model handles case reassignment.",
            3: "⚠️ Tier 3 detected (relative/conditional). Neural model carries the full load.",
        }
        st.info(f"{tier_messages[result.tier]}  Flags: `{', '.join(result.tier_flags) or '—'}`")

    # ---- Errors ----
    if result.errors:
        neural_errors = [e for e in result.errors if e.get("start", -1) >= 0]
        ortho_errors = [e for e in result.errors if e.get("start", -1) < 0]

        if ortho_errors:
            st.subheader("🔧 Orthographic corrections (rule-based)")
            for err in ortho_errors:
                st.success(f"**{err['text']}** → **{err.get('corrected', '?')}** — {err['description']}")

        if neural_errors:
            st.subheader("⚠️ Detected errors (neural)")
            for err in neural_errors:
                st.warning(f"**{err['text']}** — {err['description']}")

    # ---- Per-word i'rab cards (grid, RTL) ----
    st.subheader("Per-word analysis")
    cols_per_row = 4
    words_rtl = list(reversed(result.words))
    for row_start in range(0, len(words_rtl), cols_per_row):
        cols = st.columns(cols_per_row)
        for ci, wi in enumerate(range(row_start, min(row_start + cols_per_row, len(words_rtl)))):
            w = words_rtl[wi]
            with cols[ci]:
                bg = role_color(w["role"])
                low_conf = w["irab_confidence"] < conf_threshold
                border = "#ff7f6e" if low_conf else "#e0e0e0"

                html = f"""
                <div style="background:{bg}; padding:12px; border-radius:8px;
                            border:1px solid {border}; direction:rtl; text-align:center; margin-bottom:8px">
                  <div style="font-family:'Noto Naskh Arabic',serif; font-size:28px; font-weight:500">
                    {w['diacritized']}
                  </div>
                  <div style="color:#444; font-size:14px; margin-top:6px">{w['role_ar']}</div>
                """
                if show_en:
                    html += f"<div style='color:#888; font-size:11px; direction:ltr'>{w['role_en']}</div>"
                if show_confidence:
                    html += f"""
                    <div style="margin-top:8px; font-size:10px; color:#666; direction:ltr">
                      diac {w['diac_confidence']*100:.0f}% · irab {w['irab_confidence']*100:.0f}%
                    </div>
                    """
                if low_conf:
                    html += "<div style='color:#c44; font-size:10px; margin-top:4px'>⚠ low confidence</div>"
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)

    # ---- Summary ----
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    avg_diac = np.mean([w["diac_confidence"] for w in result.words]) if result.words else 0
    avg_irab = np.mean([w["irab_confidence"] for w in result.words]) if result.words else 0
    c1.metric("Words", len(result.words))
    c2.metric("Avg diac conf", f"{avg_diac*100:.1f}%")
    c3.metric("Avg i'rab conf", f"{avg_irab*100:.1f}%")
    c4.metric("Tier", result.tier)

    # ---- Raw JSON (collapsible) ----
    with st.expander("Raw model output (JSON)"):
        st.json(result.to_json())

# Footer
st.markdown("---")
st.caption(
    "Trained on QAC + Tashkeela + I3rab + synthetic errors. "
    "Model is approximate; i'rāb labels are coarse-grained (11 classes)."
)

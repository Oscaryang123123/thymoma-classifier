#!/usr/bin/env python3
"""Thymic carcinoma vs. thymoma - 13-program transcriptomic classifier (Streamlit web tool).
Run: streamlit run webapp/app.py
"""
import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
C_CA = "#c1272d"; C_TH = "#2b6cb0"

# 模型路径：用本文件所在目录定位，避免云端工作目录不同导致找不到模型
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")

DISP = {
    "Thymic_epithelial": "Thymic epithelial", "Immature_T": "Immature T",
    "Cytotoxic_T": "Cytotoxic T", "Treg": "Treg", "B_cell_GC": "Germinal-center B",
    "Myeloid_TAM": "Myeloid / TAM", "DC_APC": "DC / APC", "Proliferation": "Proliferation",
    "ECM_stroma": "ECM / stroma", "IFN_activation": "IFN activation",
    "Squamous_diff": "Squamous diff.", "T_exhaustion": "T exhaustion",
    "Angiogenesis": "Angiogenesis",
}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

B = load_model()
PROGRAMS = B["programs"]


def predict(expr):
    """expr: gene x sample DataFrame -> (scores DataFrame, prob array).
    打分：样本内基因 z-score → 基因集均值（与训练一致）。"""
    # 若输入为原始 counts/FPKM（数值范围大），先 log2 转换
    if expr.values.max() > 100:
        expr = np.log2(expr + 1)
    zscore = (expr - expr.mean(axis=0)) / expr.std(axis=0)
    scores = pd.DataFrame({k: zscore.loc[[g for g in v if g in zscore.index]].mean(axis=0)
                           for k, v in B["gene_sets"].items()})[PROGRAMS]
    Xs = (scores[PROGRAMS].values - B["scaler_mean"]) / B["scaler_scale"]
    z = Xs @ B["coef"] + B["intercept"]
    prob = 1 / (1 + np.exp(-z))
    return scores, prob


st.set_page_config(page_title="Thymic Carcinoma vs Thymoma Classifier", layout="wide")
st.title("Thymic Carcinoma vs. Thymoma Classifier")
st.caption(
    f"A 13-program thymic transcriptomic signature. Training cohort: TCGA-THYM "
    f"(thymic carcinoma {B['n_ca']} / thymoma {B['n_th']}); internal cross-validated AUC 0.988; "
    f"external AUC 0.981 on GSE57892 (both classes, one platform). "
    f"For research use only; not for clinical diagnosis."
)

# ---- Input ----
st.header("1. Upload gene expression data")
st.markdown(
    "CSV file: **rows = genes (human gene symbols), columns = samples**, values = expression "
    "(RNA-seq counts/TPM or microarray signal are all accepted; scores are z-scored within each sample). "
    "Multiple samples can be predicted at once."
)
st.markdown("Format example: `gene,Sample1,Sample2 ...` (first column = gene symbol, remaining columns = samples).")

up = st.file_uploader("Choose a CSV file", type=["csv", "txt", "tsv"])

if up is not None:
    # 文件变化时重新计算；否则复用 session_state 缓存（切换样本不重算）
    file_key = f"{up.name}_{up.size}"
    if "file_key" not in st.session_state or st.session_state.file_key != file_key:
        try:
            raw = pd.read_csv(io.BytesIO(up.read()), sep=None, engine="python")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

        if not pd.api.types.is_numeric_dtype(raw.iloc[:, 0]):
            genes = raw.iloc[:, 0].astype(str)
            expr = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        else:
            genes = raw.iloc[:, -1].astype(str)
            expr = raw.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
        expr.index = genes
        expr = expr[~expr.index.duplicated(keep="first")]
        expr = expr.dropna(how="all")

        all_gs = set(g for gs in B["gene_sets"].values() for g in gs)
        overlap = len(all_gs & set(expr.index))
        if overlap < 30:
            st.warning(
                f"Low overlap between uploaded genes and the signature gene sets "
                f"({overlap}/{len(all_gs)}). Confirm genes are uppercase human gene symbols; "
                f"otherwise results may be unreliable."
            )

        with st.spinner("Computing 13-program scores and predicting..."):
            scores, prob = predict(expr)

        st.session_state.file_key = file_key
        st.session_state.scores = scores
        st.session_state.prob = prob
        st.session_state.n_genes = expr.shape[0]
        st.session_state.n_samples = expr.shape[1]

    scores = st.session_state.scores
    prob = st.session_state.prob

    st.success(f"Loaded: {st.session_state.n_genes} genes x {st.session_state.n_samples} sample(s)")

    # ---- Results ----
    st.header("2. Prediction results")
    res = pd.DataFrame({
        "Sample": scores.index,
        "Probability of thymic carcinoma": prob,
        "Prediction": ["Thymic carcinoma" if p > 0.5 else "Thymoma" for p in prob],
    }).set_index("Sample")
    st.dataframe(res.style.format({"Probability of thymic carcinoma": "{:.4f}"}))

    st.subheader("13-program profile")
    sel = st.selectbox("Select a sample to view its program profile", scores.index.tolist())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    vals = scores.loc[sel]
    colors = [C_CA if v > 0 else C_TH for v in vals]
    ax.barh(range(len(PROGRAMS)), vals.values, color=colors)
    ax.set_yticks(range(len(PROGRAMS)))
    ax.set_yticklabels([DISP[p] for p in PROGRAMS])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Program score (z)")
    ax.set_title(f"{sel} - carcinoma probability {prob[list(scores.index).index(sel)]:.3f}")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    st.pyplot(fig)

    st.download_button(
        "Download 13-program scores (CSV)",
        scores.to_csv().encode("utf-8"),
        file_name="program_scores.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Method: 13 thymic biology programs scored by within-sample gene z-scoring followed by gene-set "
    "averaging, standardized, then a logistic-regression classifier. Red = carcinoma-associated programs "
    "(squamous diff. / stroma / IFN); blue = thymoma-associated programs (thymic epithelial / immature T)."
)

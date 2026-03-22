
# ============================================================
# Child Digital Twin Dashboard in Python (Streamlit, WHO-style)
# Supports both WIDE and LONG CSV input formats
#
# Run with:
#   streamlit run child_digital_twin_streamlit_who.py
#
# Main difference from the earlier Python version:
# - Attempts WHO/reference-style z-scores using the Python package:
#     who-growth-standard
# - Falls back to internal age-by-sex z-scores if that package is
#   unavailable or if a specific indicator cannot be computed.
#
# Notes:
# - Weight-for-age and length/height-for-age are attempted first.
# - Head circumference and MUAC references are less consistently
#   available across Python packages, so these may fall back.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import importlib
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression

AGES = [0, 3, 6, 9, 12, 18, 24]
MEASURES = ["weight", "muac", "length", "hc"]


def safe_slope(age, x):
    age = pd.Series(age)
    x = pd.Series(x)
    ok = age.notna() & x.notna()
    if ok.sum() < 2:
        return np.nan
    return float(np.polyfit(age[ok], x[ok], 1)[0])


def safe_auc(age, x):
    age = pd.Series(age)
    x = pd.Series(x)
    ok = age.notna() & x.notna()
    if ok.sum() < 2:
        return np.nan
    a = age[ok].to_numpy()
    y = x[ok].to_numpy()
    o = np.argsort(a)
    return float(trapezoid(y[o], a[o]))


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def make_example_data(n=500, fmt="wide", seed=123):
    rng = np.random.default_rng(seed)
    ids = [f"C{i:05d}" for i in range(1, n + 1)]
    gender = rng.choice(["Male", "Female"], size=n)
    shift = np.where(gender == "Male", 0.10, -0.05)

    bw = np.maximum(rng.normal(3.1, 0.45, n), 1.5)
    mu0 = np.maximum(rng.normal(11.0, 0.9, n), 8.0)
    le0 = np.maximum(rng.normal(49.5, 2.4, n), 43.0)
    hc0 = np.maximum(rng.normal(34.1, 1.4, n), 30.0)

    def gen(base, incs, noise):
        mats = []
        for inc in incs:
            mats.append(np.maximum(base + inc + shift + rng.normal(0, noise, n), 0.01))
        return np.vstack(mats).T

    wt = gen(bw,  [0, 2.6, 4.8, 6.4, 7.7, 9.2, 10.8], 0.28)
    mu = gen(mu0, [0, 1.6, 2.4, 3.0, 3.7, 4.8, 5.7], 0.18)
    le = gen(le0, [0, 10.2, 16.6, 21.1, 25.2, 31.6, 37.2], 0.40)
    hc = gen(hc0, [0, 5.6, 8.4, 9.8, 11.0, 12.5, 13.8], 0.22)

    wide = pd.DataFrame({"id": ids, "gender": gender})
    for j, a in enumerate(AGES):
        if a == 0:
            wide["birthweight_0"] = np.round(wt[:, j], 2)
        else:
            wide[f"weight_{a}"] = np.round(wt[:, j], 2)
        wide[f"muac_{a}"] = np.round(mu[:, j], 1)
        wide[f"length_{a}"] = np.round(le[:, j], 1)
        wide[f"hc_{a}"] = np.round(hc[:, j], 1)

    signal = (
        0.32 * ((wide["weight_24"] - wide["weight_24"].mean()) / wide["weight_24"].std())
        + 0.26 * ((wide["length_24"] - wide["length_24"].mean()) / wide["length_24"].std())
        + 0.18 * ((wide["hc_24"] - wide["hc_24"].mean()) / wide["hc_24"].std())
        + 0.16 * ((wide["muac_24"] - wide["muac_24"].mean()) / wide["muac_24"].std())
        + rng.normal(0, 0.45, n)
    )
    wide["gsed_24"] = np.round(92 + 10 * signal + np.where(wide["gender"] == "Female", 1.2, 0), 1)
    wide["daz_24"] = np.round(-0.1 + 0.75 * signal + rng.normal(0, 0.25, n), 2)
    wide["dscore_24"] = np.round(45 + 6.5 * signal + rng.normal(0, 0.8, n), 2)

    num_cols = [c for c in wide.columns if c not in ["id", "gender", "gsed_24", "daz_24", "dscore_24"]]
    for col in num_cols:
        frac = 0.06 if col.endswith("_24") else 0.10
        idx = rng.choice(n, int(n * frac), replace=False)
        wide.loc[idx, col] = np.nan

    if fmt == "wide":
        return wide
    return wide_to_long(wide)


def detect_input_format(df):
    cols = {c.lower() for c in df.columns}
    if {"id", "gender", "age", "weight", "muac", "length", "hc"}.issubset(cols):
        return "long"
    has_wide = any(c.lower().startswith(("birthweight_", "weight_", "muac_", "length_", "hc_")) for c in df.columns)
    if {"id", "gender"}.issubset(cols) and has_wide:
        return "wide"
    return "unknown"


def canonicalize_long(df):
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    rename = {}
    if "sex" in d.columns and "gender" not in d.columns:
        rename["sex"] = "gender"
    if "head_circumference" in d.columns and "hc" not in d.columns:
        rename["head_circumference"] = "hc"
    if "headcircumference" in d.columns and "hc" not in d.columns:
        rename["headcircumference"] = "hc"
    if "head_circ" in d.columns and "hc" not in d.columns:
        rename["head_circ"] = "hc"
    d = d.rename(columns=rename)

    required = ["id", "gender", "age", "weight", "muac", "length", "hc"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required long-format columns: {missing}")

    for c in ["gsed_24", "daz_24", "dscore_24"]:
        if c not in d.columns:
            d[c] = np.nan

    d = d[["id", "gender", "age", "weight", "muac", "length", "hc", "gsed_24", "daz_24", "dscore_24"]].copy()
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    for c in ["weight", "muac", "length", "hc", "gsed_24", "daz_24", "dscore_24"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["age"].isin(AGES)].sort_values(["id", "age"]).reset_index(drop=True)
    return d


def wide_to_long(df):
    d = df.copy()
    if "birthweight" in d.columns and "birthweight_0" not in d.columns:
        d["birthweight_0"] = d["birthweight"]
    if "weight_0" in d.columns and "birthweight_0" not in d.columns:
        d["birthweight_0"] = d["weight_0"]

    rows = []
    for _, row in d.iterrows():
        for a in AGES:
            rows.append({
                "id": row.get("id"),
                "gender": row.get("gender"),
                "age": a,
                "weight": row.get("birthweight_0") if a == 0 else row.get(f"weight_{a}", np.nan),
                "muac": row.get(f"muac_{a}", np.nan),
                "length": row.get(f"length_{a}", np.nan),
                "hc": row.get(f"hc_{a}", np.nan),
                "gsed_24": row.get("gsed_24", np.nan),
                "daz_24": row.get("daz_24", np.nan),
                "dscore_24": row.get("dscore_24", np.nan),
                "dscore_24": row.get("dscore_24", np.nan),
            })
    return pd.DataFrame(rows).sort_values(["id", "age"]).reset_index(drop=True)


def clean_units(df):
    d = df.copy()
    d["gender"] = (
        d["gender"].astype(str).str.strip().str.lower()
        .replace({"m": "male", "f": "female", "1": "male", "2": "female"})
        .replace({"male": "Male", "female": "Female"})
    )
    if d["muac"].median(skipna=True) > 30:
        d["muac"] = d["muac"] / 10.0
    if d["hc"].median(skipna=True) > 80:
        d["hc"] = d["hc"] / 10.0
    if d["length"].median(skipna=True) > 200:
        d["length"] = d["length"] / 10.0

    d.loc[(d["weight"] < 0.5) | (d["weight"] > 35), "weight"] = np.nan
    d.loc[(d["muac"] < 5) | (d["muac"] > 30), "muac"] = np.nan
    d.loc[(d["length"] < 30) | (d["length"] > 130), "length"] = np.nan
    d.loc[(d["hc"] < 20) | (d["hc"] > 65), "hc"] = np.nan
    return d.sort_values(["id", "age"]).reset_index(drop=True)


def impute_longitudinal(df):
    d = df.copy()
    outcomes = d.groupby("id")[["gsed_24", "daz_24", "dscore_24"]].first().reset_index()
    genders = d.groupby("id")["gender"].first().to_dict()

    wide = d.pivot(index="id", columns="age", values=MEASURES)
    wide.columns = [f"{v}_{a}" for v, a in wide.columns]
    wide = wide.reset_index()
    wide["gender_num"] = wide["id"].map(lambda x: 1 if genders.get(x) == "Male" else 0)
    wide = wide.merge(outcomes, on="id", how="left")

    X = wide.drop(columns=["id"]).copy()
    imp = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=123)
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    outw = pd.concat([wide[["id"]], X_imp], axis=1)
    outw["gender"] = outw["gender_num"].round().clip(0, 1).map({1: "Male", 0: "Female"})
    outw["gsed_24"] = wide["gsed_24"].values
    outw["daz_24"] = wide["daz_24"].values

    rows = []
    for _, row in outw.iterrows():
        for a in AGES:
            rows.append({
                "id": row["id"],
                "gender": row["gender"],
                "age": a,
                "weight": row.get(f"weight_{a}", row.get("birthweight_0", np.nan) if a == 0 else np.nan),
                "muac": row.get(f"muac_{a}", np.nan),
                "length": row.get(f"length_{a}", np.nan),
                "hc": row.get(f"hc_{a}", np.nan),
                "gsed_24": row.get("gsed_24", np.nan),
                "daz_24": row.get("daz_24", np.nan),
                "dscore_24": row.get("dscore_24", np.nan),
            })
    out = pd.DataFrame(rows).drop(columns=["gsed_24", "daz_24", "dscore_24"]).merge(outcomes, on="id", how="left")
    return out.sort_values(["id", "age"]).reset_index(drop=True)


def add_internal_zscores(df):
    d = df.copy()
    for var in MEASURES:
        d[f"{var}_z"] = d.groupby(["age", "gender"])[var].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if pd.notna(s.std(ddof=0)) and s.std(ddof=0) != 0 else np.nan
        )
    return d


def try_add_who_zscores(df):
    d = df.copy()
    d["zscore_method"] = "internal"

    pkg = importlib.util.find_spec("who_growth_standard")
    if pkg is None:
        return add_internal_zscores(d), "Fallback: internal age-by-sex z-scores (who-growth-standard not installed)"

    try:
        import who_growth_standard as wgs
    except Exception:
        return add_internal_zscores(d), "Fallback: internal age-by-sex z-scores (who-growth-standard import failed)"

    # start with internal, then overwrite what can be computed
    d = add_internal_zscores(d)

    sex_map = {"Male": "M", "Female": "F"}
    d["sex_who"] = d["gender"].map(sex_map)
    d["age_days"] = np.round(d["age"] * 30.4375).astype("float")

    # Try a few possible interfaces because Python WHO packages vary.
    # If one path works, overwrite the corresponding z-scores.
    used_any = False

    # 1) Weight-for-age
    try:
        if hasattr(wgs, "wfa_zscore"):
            d["weight_z"] = [
                wgs.wfa_zscore(sex=s, age_in_days=a, weight=w) if pd.notna(s) and pd.notna(a) and pd.notna(w) else np.nan
                for s, a, w in zip(d["sex_who"], d["age_days"], d["weight"])
            ]
            used_any = True
    except Exception:
        pass

    # 2) Length/height-for-age
    try:
        if hasattr(wgs, "lhfa_zscore"):
            d["length_z"] = [
                wgs.lhfa_zscore(sex=s, age_in_days=a, length=l) if pd.notna(s) and pd.notna(a) and pd.notna(l) else np.nan
                for s, a, l in zip(d["sex_who"], d["age_days"], d["length"])
            ]
            used_any = True
        elif hasattr(wgs, "hfa_zscore"):
            d["length_z"] = [
                wgs.hfa_zscore(sex=s, age_in_days=a, height=l) if pd.notna(s) and pd.notna(a) and pd.notna(l) else np.nan
                for s, a, l in zip(d["sex_who"], d["age_days"], d["length"])
            ]
            used_any = True
    except Exception:
        pass

    # 3) MUAC-for-age if package exposes it
    try:
        if hasattr(wgs, "mfa_zscore"):
            d["muac_z"] = [
                wgs.mfa_zscore(sex=s, age_in_days=a, muac=m) if pd.notna(s) and pd.notna(a) and pd.notna(m) else np.nan
                for s, a, m in zip(d["sex_who"], d["age_days"], d["muac"])
            ]
            used_any = True
    except Exception:
        pass

    # 4) Head circumference-for-age if package exposes it
    try:
        if hasattr(wgs, "hcfa_zscore"):
            d["hc_z"] = [
                wgs.hcfa_zscore(sex=s, age_in_days=a, hc=h) if pd.notna(s) and pd.notna(a) and pd.notna(h) else np.nan
                for s, a, h in zip(d["sex_who"], d["age_days"], d["hc"])
            ]
            used_any = True
    except Exception:
        pass

    d = d.drop(columns=["sex_who", "age_days"], errors="ignore")
    if used_any:
        d["zscore_method"] = "WHO/reference-based where available; internal fallback for others"
        return d, "WHO/reference-based z-scores used where available; internal fallback for unsupported indicators"
    return d, "Fallback: internal age-by-sex z-scores (WHO package available but indicator functions not matched)"


def build_child_features(df):
    rows = []
    for cid, g in df.groupby("id"):
        g = g.sort_values("age")
        rows.append({
            "id": cid,
            "gender": g["gender"].iloc[0],
            "gender_num": 1 if g["gender"].iloc[0] == "Male" else 0,
            "birthweight": g.loc[g["age"] == 0, "weight"].iloc[0] if (g["age"] == 0).any() else np.nan,
            "weight_z_24": g.loc[g["age"] == 24, "weight_z"].iloc[0] if (g["age"] == 24).any() else np.nan,
            "muac_z_24": g.loc[g["age"] == 24, "muac_z"].iloc[0] if (g["age"] == 24).any() else np.nan,
            "length_z_24": g.loc[g["age"] == 24, "length_z"].iloc[0] if (g["age"] == 24).any() else np.nan,
            "hc_z_24": g.loc[g["age"] == 24, "hc_z"].iloc[0] if (g["age"] == 24).any() else np.nan,
            "wt_slope": safe_slope(g["age"], g["weight"]),
            "mu_slope": safe_slope(g["age"], g["muac"]),
            "len_slope": safe_slope(g["age"], g["length"]),
            "hc_slope": safe_slope(g["age"], g["hc"]),
            "wt_auc": safe_auc(g["age"], g["weight"]),
            "mu_auc": safe_auc(g["age"], g["muac"]),
            "len_auc": safe_auc(g["age"], g["length"]),
            "hc_auc": safe_auc(g["age"], g["hc"]),
            "gsed_24": g["gsed_24"].dropna().iloc[0] if g["gsed_24"].notna().any() else np.nan,
            "daz_24": g["daz_24"].dropna().iloc[0] if g["daz_24"].notna().any() else np.nan,
            "dscore_24": g["dscore_24"].dropna().iloc[0] if g["dscore_24"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def fit_models(features):
    xcols = ["gender_num", "birthweight", "weight_z_24", "muac_z_24", "length_z_24", "hc_z_24",
             "wt_slope", "mu_slope", "len_slope", "hc_slope", "wt_auc", "mu_auc", "len_auc", "hc_auc"]
    models = {}
    for target in ["gsed_24", "daz_24", "dscore_24"]:
        sub = features[xcols + [target]].dropna()
        if len(sub) >= 20:
            mdl = LinearRegression()
            mdl.fit(sub[xcols], sub[target])
            models[target] = (mdl, xcols)
        else:
            models[target] = (None, xcols)
    return models


def add_predictions(features, models):
    out = features.copy()
    for target in ["gsed_24", "daz_24", "dscore_24"]:
        pred_col = f"pred_{target}"
        out[pred_col] = np.nan
        mdl, xcols = models[target]
        if mdl is not None:
            ok = out[xcols].notna().all(axis=1)
            out.loc[ok, pred_col] = mdl.predict(out.loc[ok, xcols])
    return out


def population_curves(df):
    keep = MEASURES + [f"{m}_z" for m in MEASURES]
    return df.groupby(["age", "gender"])[keep].mean(numeric_only=True).reset_index()


def build_counterfactual(child_long, start_age=6, dwt=0, dmu=0, dlen=0, dhc=0):
    cf = child_long.copy()
    cf.loc[cf["age"] >= start_age, "weight"] += dwt
    cf.loc[cf["age"] >= start_age, "muac"] += dmu
    cf.loc[cf["age"] >= start_age, "length"] += dlen
    cf.loc[cf["age"] >= start_age, "hc"] += dhc
    return cf


def overall_risk(pred_gsed, pred_daz, row):
    red = 0
    yellow = 0
    if pd.notna(pred_daz) and pred_daz < -1.5:
        red += 1
    elif pd.notna(pred_daz) and pred_daz < -0.5:
        yellow += 1
    if pd.notna(pred_gsed) and pred_gsed < 85:
        red += 1
    elif pd.notna(pred_gsed) and pred_gsed < 92:
        yellow += 1
    for c in ["length_z_24", "weight_z_24", "hc_z_24"]:
        v = row.get(c, np.nan)
        if pd.notna(v) and v < -2:
            red += 1
        elif pd.notna(v) and v < -1:
            yellow += 1
    if red >= 2:
        return "High"
    if red >= 1 or yellow >= 2:
        return "Moderate"
    return "Low"


@dataclass
class Processed:
    raw: pd.DataFrame
    fmt: str
    long_imputed: pd.DataFrame
    features: pd.DataFrame
    pop: pd.DataFrame
    models: dict
    zscore_note: str


def process_input(df):
    fmt = detect_input_format(df)
    if fmt == "wide":
        long_df = wide_to_long(df)
    elif fmt == "long":
        long_df = canonicalize_long(df)
    else:
        raise ValueError("Could not detect input format. Use wide or long CSV.")
    long_df = clean_units(long_df)
    long_imp = impute_longitudinal(long_df)
    long_imp, zscore_note = try_add_who_zscores(long_imp)
    feats = build_child_features(long_imp)
    models = fit_models(feats)
    feats = add_predictions(feats, models)
    pop = population_curves(long_imp)
    return Processed(df, fmt, long_imp, feats, pop, models, zscore_note)


st.set_page_config(page_title="Child Digital Twin (Python WHO-style)", layout="wide")
st.title("Child Digital Twin Dashboard in Python")
st.caption("Wide/long CSV input • imputation • WHO-style z-score attempt • GSED/DAZ/D-score prediction • individual twin • counterfactual simulation")

with st.sidebar:
    st.header("Data")
    use_example = st.checkbox("Use example data", value=True)
    example_n = st.number_input("Example number of children", min_value=50, max_value=10000, value=500, step=50)
    example_fmt = st.radio("Example format", ["wide", "long"], horizontal=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    st.header("Twin controls")
    cutoff_age = st.slider("Observed visits up to age (months)", 0, 24, 12, 3)

    st.header("Counterfactual")
    cf_start_age = st.slider("Counterfactual starts at month", 3, 24, 6, 3)
    cf_dwt = st.slider("Weight change (kg)", -2.0, 2.0, 0.5, 0.1)
    cf_dmu = st.slider("MUAC change (cm)", -3.0, 3.0, 0.5, 0.1)
    cf_dlen = st.slider("Length change (cm)", -6.0, 6.0, 1.0, 0.1)
    cf_dhc = st.slider("Head circumference change (cm)", -3.0, 3.0, 0.3, 0.1)

try:
    if use_example:
        raw_df = make_example_data(int(example_n), fmt=example_fmt)
    else:
        if uploaded is None:
            st.info("Upload a CSV or enable example data.")
            st.stop()
        raw_df = pd.read_csv(uploaded)
    proc = process_input(raw_df)
except Exception as e:
    st.error(f"Processing failed: {e}")
    st.stop()

st.info(proc.zscore_note)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Children", proc.long_imputed["id"].nunique())
c2.metric("Input format", proc.fmt.upper())
c3.metric("Long rows", len(proc.long_imputed))
c4.metric("Feature columns", proc.features.shape[1])

with st.expander("Downloads"):
    st.download_button("Download imputed long CSV", data=to_csv_bytes(proc.long_imputed), file_name="child_digital_twin_imputed_long.csv", mime="text/csv")
    st.download_button("Download child features CSV", data=to_csv_bytes(proc.features), file_name="child_digital_twin_features.csv", mime="text/csv")

selected_child = st.selectbox("Select child", sorted(proc.features["id"].astype(str).tolist()))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Preview", "Population graphs", "Individual twin", "Counterfactual", "Correlations"])

with tab1:
    a, b = st.columns(2)
    a.subheader("Raw data preview")
    a.dataframe(proc.raw.head(20), use_container_width=True)
    b.subheader("Prediction preview")
    b.dataframe(proc.features[["id", "gender", "pred_gsed_24", "pred_daz_24", "pred_dscore_24", "weight_z_24", "length_z_24", "muac_z_24", "hc_z_24"]].head(20), use_container_width=True)

with tab2:
    st.subheader("Population trajectories")
    pop_long = proc.pop.melt(id_vars=["age", "gender"], value_vars=MEASURES, var_name="measure", value_name="value")
    fig = px.line(pop_long, x="age", y="value", color="gender", facet_col="measure", facet_col_wrap=2, markers=True)
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mean z-score curves")
    zcols = [f"{m}_z" for m in MEASURES]
    pop_z = proc.pop.melt(id_vars=["age", "gender"], value_vars=zcols, var_name="measure", value_name="value")
    fig = px.line(pop_z, x="age", y="value", color="gender", facet_col="measure", facet_col_wrap=2, markers=True)
    fig.update_layout(height=650)
    for y, dash in [(-2, "dash"), (-1, "dot"), (0, "solid"), (1, "dot"), (2, "dash")]:
        fig.add_hline(y=y, line_dash=dash, opacity=0.35)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Observed vs predicted outcomes")
    g1 = px.scatter(proc.features, x="gsed_24", y="pred_gsed_24", color="gender")
    st.plotly_chart(g1, use_container_width=True)
    g2 = px.scatter(proc.features, x="daz_24", y="pred_daz_24", color="gender")
    st.plotly_chart(g2, use_container_width=True)

with tab3:
    st.subheader("Individual digital twin")
    child_long = proc.long_imputed[proc.long_imputed["id"].astype(str) == selected_child].copy()
    expected = proc.long_imputed[proc.long_imputed["gender"] == child_long["gender"].iloc[0]].groupby("age")[MEASURES].mean().reset_index()

    fig = go.Figure()
    for measure in MEASURES:
        fig.add_trace(go.Scatter(
            x=child_long["age"], y=child_long[measure], mode="lines+markers",
            name=f"{measure}: observed"
        ))
        fig.add_trace(go.Scatter(
            x=expected["age"], y=expected[measure], mode="lines",
            line=dict(dash="dash"), name=f"{measure}: expected"
        ))
    fig.update_layout(height=650, xaxis_title="Age (months)", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    a.dataframe(child_long, use_container_width=True)
    b.dataframe(proc.features[proc.features["id"].astype(str) == selected_child], use_container_width=True)

with tab4:
    st.subheader("Counterfactual simulation")
    child_long = proc.long_imputed[proc.long_imputed["id"].astype(str) == selected_child].copy()
    cf = build_counterfactual(child_long, cf_start_age, cf_dwt, cf_dmu, cf_dlen, cf_dhc)
    cf, _ = try_add_who_zscores(cf)

    base_feat = build_child_features(child_long).iloc[0:1].copy()
    cf_feat = build_child_features(cf).iloc[0:1].copy()
    for target in ["gsed_24", "daz_24", "dscore_24"]:
        mdl, xcols = proc.models[target]
        base_feat[f"pred_{target}"] = np.nan
        cf_feat[f"pred_{target}"] = np.nan
        if mdl is not None and base_feat[xcols].notna().all(axis=1).iloc[0]:
            base_feat[f"pred_{target}"] = mdl.predict(base_feat[xcols])[0]
        if mdl is not None and cf_feat[xcols].notna().all(axis=1).iloc[0]:
            cf_feat[f"pred_{target}"] = mdl.predict(cf_feat[xcols])[0]

    risk = overall_risk(cf_feat["pred_gsed_24"].iloc[0], cf_feat["pred_daz_24"].iloc[0], cf_feat.iloc[0].to_dict())
    a, b, c, d, e = st.columns(5)
    a.metric("Overall risk", risk)
    b.metric("Predicted GSED", f'{cf_feat["pred_gsed_24"].iloc[0]:.2f}' if pd.notna(cf_feat["pred_gsed_24"].iloc[0]) else "NA")
    c.metric("Predicted DAZ", f'{cf_feat["pred_daz_24"].iloc[0]:.2f}' if pd.notna(cf_feat["pred_daz_24"].iloc[0]) else "NA")
    d.metric("Predicted D-score", f'{cf_feat["pred_dscore_24"].iloc[0]:.2f}' if pd.notna(cf_feat["pred_dscore_24"].iloc[0]) else "NA")
    e.metric("24m length z", f'{cf_feat["length_z_24"].iloc[0]:.2f}' if pd.notna(cf_feat["length_z_24"].iloc[0]) else "NA")

    obs_long = child_long[["age"] + MEASURES].melt(id_vars=["age"], value_vars=MEASURES, var_name="measure", value_name="value")
    obs_long["scenario"] = "Observed"
    cf_long = cf[["age"] + MEASURES].melt(id_vars=["age"], value_vars=MEASURES, var_name="measure", value_name="value")
    cf_long["scenario"] = "Counterfactual"
    plot_df = pd.concat([obs_long, cf_long], ignore_index=True)
    fig = px.line(plot_df, x="age", y="value", color="scenario", facet_col="measure", facet_col_wrap=2, markers=True)
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    compare = pd.DataFrame({
        "Metric": ["Predicted GSED at 24m", "Predicted DAZ at 24m", "Predicted D-score at 24m"],
        "Baseline": [base_feat["pred_gsed_24"].iloc[0], base_feat["pred_daz_24"].iloc[0], base_feat["pred_dscore_24"].iloc[0]],
        "Counterfactual": [cf_feat["pred_gsed_24"].iloc[0], cf_feat["pred_daz_24"].iloc[0], cf_feat["pred_dscore_24"].iloc[0]],
    })
    compare["Change"] = compare["Counterfactual"] - compare["Baseline"]
    st.dataframe(compare, use_container_width=True)

with tab5:
    st.subheader("Correlation heatmap")
    corr_df = proc.long_imputed[["weight", "muac", "length", "hc", "weight_z", "muac_z", "length_z", "hc_z", "gsed_24", "daz_24", "dscore_24"]].copy()
    corr = corr_df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

st.success("Python WHO-style version ready. Run locally with: streamlit run child_digital_twin_streamlit_who.py")

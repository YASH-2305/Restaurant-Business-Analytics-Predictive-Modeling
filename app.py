# app.py — corrected, robust version
# Run: streamlit run app.py

import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import streamlit.components.v1 as components
from urllib.parse import quote_plus
from math import exp

# ------------- Config (change names if necessary) -------------
CSV_FILENAME = "RESTAURANT DATASET.csv"
PIPELINE_FILENAME = "restaurant_pipeline_rf.pkl"   # optional rating regressor pipeline
CLASSIFIER_FILENAME = "new_success_model.pkl"      # optional success classifier
# ------------------------------------------------------------

st.set_page_config(page_title="RESTAURANT RATING FORECAST", layout="wide")

# ---------- Helpers ----------
def find_csv(prefer=CSV_FILENAME):
    if os.path.exists(prefer):
        return prefer
    for f in os.listdir("."):
        if f.lower().endswith(".csv") and "restaurant" in f.lower():
            return f
    return None

def load_if_exists(path):
    return joblib.load(path) if os.path.exists(path) else None

def maps_embed_url(place_query):
    q = quote_plus(place_query)
    return f"https://www.google.com/maps?q={q}&output=embed"

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def sigmoid(x):
    return 1 / (1 + exp(-x))

# Heuristic fallback for success probability (robust, uses dataset stats)
def heuristic_success_prob(row, medians, stdevs):
    # Prefer Estimated_Rating if present, else fallback to other features
    # Normalize rating to ~ N(0,1) using dataset median/std (robust)
    rating = row.get("Estimated_Rating", np.nan)
    cost = row.get("Estimated_Cost_for_Two_INR", np.nan)

    score_components = []
    weights = []

    # rating component (weight 0.7 if available)
    if not pd.isna(rating) and "Estimated_Rating" in medians and stdevs.get("Estimated_Rating", 0) > 0:
        z = (rating - medians["Estimated_Rating"]) / (stdevs["Estimated_Rating"] if stdevs["Estimated_Rating"]>0 else 1.0)
        # map z via sigmoid centered at 0
        rating_score = sigmoid(z)
        score_components.append(rating_score)
        weights.append(0.7)
    # cost component: lower cost often helps (weight 0.3)
    if not pd.isna(cost) and "Estimated_Cost_for_Two_INR" in medians and stdevs.get("Estimated_Cost_for_Two_INR", 0) > 0:
        # compute how cost compares to median: lower is better -> invert
        z_cost = (cost - medians["Estimated_Cost_for_Two_INR"]) / (stdevs["Estimated_Cost_for_Two_INR"] if stdevs["Estimated_Cost_for_Two_INR"]>0 else 1.0)
        cost_score = sigmoid(-z_cost)  # lower cost -> higher score
        score_components.append(cost_score)
        weights.append(0.3)

    # if we have both, weighted average; else fallback to single component or 0.5
    if weights:
        weighted = sum(c*w for c,w in zip(score_components, weights)) / sum(weights)
        return clamp(weighted)
    else:
        return 0.5

# ---------- Load dataset & models ----------
CSV_PATH = find_csv()
df = pd.read_csv(CSV_PATH) if CSV_PATH else None

pipeline = load_if_exists(PIPELINE_FILENAME)   # may be None
classifier = load_if_exists(CLASSIFIER_FILENAME)  # may be None

# compute medians and robust stdevs for useful columns
medians = {}
stdevs = {}
if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        medians[c] = df[c].median()
        # use MAD (median absolute deviation) scaled to approximate std to be robust to outliers
        mad = np.median(np.abs(df[c].dropna() - medians[c])) if df[c].dropna().size>0 else 0.0
        approx_std = mad * 1.4826 if mad>0 else (df[c].std() if df[c].std()>0 else 0.0)
        stdevs[c] = approx_std

# ---------- Styling ----------
st.markdown("""
<style>
h1 { text-align:center; font-weight:800; letter-spacing:1px; margin-bottom:8px; }
.label { font-weight:700; color:#0f172a; }
.value { color:#334155; }
.btn { padding:10px 20px; border-radius:10px; font-weight:700; }
.oval { background:#0ea5e9; color:white; padding:12px 28px; border-radius:999px; }
.reason-box { border-radius:20px; padding:16px; }
</style>
""", unsafe_allow_html=True)

# ---------- Header & selection ----------
st.markdown("<h1>RESTAURANT RATING FORECAST</h1>", unsafe_allow_html=True)
st.write("")

# selection stored in session_state so it persists
if "selected_name" not in st.session_state:
    st.session_state["selected_name"] = None

restaurant_options = df['Restaurant_Name'].tolist() if (df is not None and 'Restaurant_Name' in df.columns) else ["(No dataset)"]
sel_index = 0
if st.session_state["selected_name"] in restaurant_options:
    sel_index = restaurant_options.index(st.session_state["selected_name"])
st.session_state["selected_name"] = st.selectbox("RESTAURANT NAME", options=restaurant_options, index=sel_index, label_visibility="hidden")

# ---------- Get Info button ----------
if st.button("Get Restaurant Info"):
    if df is None:
        st.error("No dataset found in this folder. Put your 'RESTAURANT DATASET.csv' here.")
    else:
        sel = st.session_state["selected_name"]
        row = df[df['Restaurant_Name'] == sel].iloc[0].to_dict()
        st.session_state["selected_row"] = row
        # clear previous forecast and justification state
        st.session_state.pop("last_forecast", None)
        st.session_state.pop("show_justification", None)
        st.success(f"Loaded info for: {sel}")

# ---------- If a row is selected, show info & map & actions ----------
if "selected_row" in st.session_state:
    row = st.session_state["selected_row"]
    left_col, right_col = st.columns([2.2, 1])

    # LEFT: neat label/value pairs (2-column view)
    with left_col:
        info_pairs = [
            ("Title", row.get("Restaurant_Name","")),
            ("Location", row.get("Location","")),
            ("Category", row.get("Category","")),
            ("Primary Cuisine", row.get("Primary_Cuisine","")),
            ("Estimated Cost for Two (INR)", row.get("Estimated_Cost_for_Two_INR","")),
            ("Estimated Rating", row.get("Estimated_Rating","")),
            ("Specialty", row.get("Specialty","")),
        ]
        for label, val in info_pairs:
            c1, c2 = st.columns([1, 3])
            c1.markdown(f"<div class='label'>{label}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='value'>{'' if pd.isna(val) else val}</div>", unsafe_allow_html=True)

        st.write("")  # spacer

        # forecast & justification buttons that set state
        fc_col, jc_col = st.columns([1,1])
        if fc_col.button("GET FORECAST"):
            # prepare input_df for model/pipeline
            input_df = pd.DataFrame([{
                'Location': row.get('Location',''),
                'Category': row.get('Category',''),
                'Primary_Cuisine': row.get('Primary_Cuisine',''),
                'Specialty': row.get('Specialty',''),
                'Estimated_Cost_for_Two_INR': row.get('Estimated_Cost_for_Two_INR', np.nan)
            }])
            if 'Cost_per_person' not in input_df.columns and 'Estimated_Cost_for_Two_INR' in input_df.columns:
                input_df['Cost_per_person'] = input_df['Estimated_Cost_for_Two_INR'] / 2.0

            pred_rating = None
            success_prob = None

            # 1) Use classifier when available (prefer probability)
            if classifier is not None:
                try:
                    if hasattr(classifier, "predict_proba"):
                        success_prob = float(classifier.predict_proba(input_df)[0][1])
                    else:
                        # classifier label fallback
                        lbl = classifier.predict(input_df)[0]
                        success_prob = float(lbl)
                except Exception as e:
                    st.warning("Classifier prediction failed; falling back to heuristic. " + str(e))
                    success_prob = None

            # 2) Get predicted rating when available
            if pipeline is not None:
                try:
                    pred_rating = float(pipeline.predict(input_df)[0])
                except Exception as e:
                    st.warning("Rating pipeline predict failed; continuing. " + str(e))
                    pred_rating = None

            # 3) If classifier not available, use robust heuristic combining rating & cost
            if success_prob is None:
                # if pipeline predicted rating, prefer that in heuristic
                if pred_rating is not None:
                    # combine predicted rating with cost using medians/stdevs
                    fake_row = row.copy()
                    fake_row["Estimated_Rating"] = pred_rating
                    success_prob = heuristic_success_prob(fake_row, medians, stdevs)
                else:
                    success_prob = heuristic_success_prob(row, medians, stdevs)

            success_prob = clamp(success_prob)
            # store last forecast in session
            st.session_state["last_forecast"] = {"success_prob": success_prob, "predicted_rating": pred_rating}

        if jc_col.button("JUSTIFICATION"):
            if "last_forecast" not in st.session_state:
                st.warning("No forecast available — click GET FORECAST first.")
            else:
                st.session_state["show_justification"] = True

        # if a forecast exists show it (so users see results immediately)
        if "last_forecast" in st.session_state:
            lf = st.session_state["last_forecast"]
            sp = lf["success_prob"]
            pr = lf["predicted_rating"]

            banner_text = "Predicted: Successful" if sp >= 0.5 else "Predicted: Unsuccessful"
            banner_bg = "#e6f8ee" if sp >= 0.5 else "#fee2e2"
            banner_border = "#a7f3d0" if sp >= 0.5 else "#fca5a5"
            st.markdown(f"<div style='background:{banner_bg}; padding:12px; border-left:6px solid {banner_border}; border-radius:6px;'><strong>{banner_text}</strong></div>", unsafe_allow_html=True)
            st.write("")

            left_pct = int(round(sp*100)); right_pct = 100 - left_pct
            orange, red = "#ff7a18", "#ef4444"
            bar_html = (f"<div style='background:#f8fafc; padding:10px; border-radius:8px;'>"
                        f"<div style='display:flex; height:110px; border-radius:8px; overflow:hidden;'>"
                        f"<div style='width:{left_pct}%; background:{orange}; display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size:34px;'>{left_pct}%</div>"
                        f"<div style='width:{right_pct}%; background:{red}; display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size:34px;'>{right_pct}%</div></div>"
                        f"<div style='margin-top:8px; color:#334155; font-weight:600;'>Success probability: {sp:.2%} &nbsp;•&nbsp; Unsuccessful: {right_pct/100:.2%}</div>"
                        f"</div>")
            st.markdown(bar_html, unsafe_allow_html=True)
            st.write("")

            if pr is not None:
                st.markdown(f"<div style='background:#eef2ff; padding:10px; border-radius:6px;'><strong>Predicted Rating:</strong> {pr:.2f}</div>", unsafe_allow_html=True)
            else:
                # show dataset's Estimated_Rating if available
                ers = row.get("Estimated_Rating", None)
                st.markdown(f"<div style='background:#eef2ff; padding:10px; border-radius:6px;'><strong>Estimated Rating:</strong> {ers if ers is not None else 'N/A'}</div>", unsafe_allow_html=True)

            # show justification immediately if flagged
            if st.session_state.get("show_justification", False):
                sp = lf["success_prob"]
                lines = []
                top_feat_line = ""
                # try to use model feature importances (if pipeline exists)
                try:
                    if pipeline is not None:
                        model = pipeline.steps[-1][1]
                        if hasattr(model, "feature_importances_"):
                            pre = pipeline.named_steps['preprocessor']
                            num_cols = pre.transformers_[0][2]
                            cat_cols = pre.transformers_[1][2]
                            ohe = pre.transformers_[1][1].named_steps['onehot']
                            ohe_names = list(ohe.get_feature_names_out(cat_cols))
                            feat_names = list(num_cols) + ohe_names
                            fi = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False)
                            top = fi.index[0]
                            top_readable = top.replace("_"," ")
                            orig_val = row.get(top, None) if top in row else None
                            if orig_val is not None:
                                top_feat_line = f"The model's top factor is **{top_readable}** (value: {orig_val})."
                            else:
                                top_feat_line = f"The model's top factor is **{top_readable}**."
                except Exception:
                    top_feat_line = ""

                # rating vs median
                if 'Estimated_Rating' in row and not pd.isna(row.get('Estimated_Rating', np.nan)) and 'Estimated_Rating' in medians:
                    if float(row['Estimated_Rating']) >= medians['Estimated_Rating']:
                        lines.append(f"Estimated rating {row['Estimated_Rating']} ≥ median {medians['Estimated_Rating']:.2f} — positive sign.")
                    else:
                        lines.append(f"Estimated rating {row['Estimated_Rating']} < median {medians['Estimated_Rating']:.2f} — reduces expected success probability.")

                # cost vs median
                if 'Estimated_Cost_for_Two_INR' in row and not pd.isna(row.get('Estimated_Cost_for_Two_INR', np.nan)) and 'Estimated_Cost_for_Two_INR' in medians:
                    if float(row['Estimated_Cost_for_Two_INR']) <= medians['Estimated_Cost_for_Two_INR']:
                        lines.append("Cost for two below median — may attract broader customer base.")
                    else:
                        lines.append("Cost for two above median — high pricing may need stronger differentiation to succeed.")

                # specialty note if present
                if 'Specialty' in row and not pd.isna(row.get('Specialty', None)):
                    lines.append(f"Specialty: {row['Specialty']} — affects demand depending on local preferences.")

                if top_feat_line:
                    lines.insert(0, top_feat_line)

                verdict = "likely to be successful" if sp >= 0.5 else "likely to be unsuccessful"
                summary = f"Overall: model estimates this restaurant is **{verdict}** (probability {sp:.1%})."

                bg = "#e6f8ee" if sp >= 0.5 else "#fee2e2"
                border = "#a7f3d0" if sp >= 0.5 else "#fca5a5"
                reason_html = f"<div class='reason-box' style='background:{bg}; border:2px solid {border};'><div style='font-weight:700; margin-bottom:6px;'>{summary}</div>"
                for ln in lines:
                    reason_html += f"<div style='margin-bottom:6px;'>{ln}</div>"
                reason_html += "</div>"
                st.markdown(reason_html, unsafe_allow_html=True)
                # clear the flag so it doesn't persist
                st.session_state["show_justification"] = False

    # RIGHT: Map embed (maps?q=...&output=embed) + fallback link
    with right_col:
        st.markdown("<h4>LOCATION</h4>", unsafe_allow_html=True)
        if 'latitude' in row and 'longitude' in row and not pd.isna(row.get('latitude', None)):
            place_q = f"{row['latitude']},{row['longitude']}"
        else:
            place_q = f"{row.get('Restaurant_Name','')}, {row.get('Location','')}"
        embed_url = maps_embed_url(place_q)
        iframe = f'<iframe src="{embed_url}" width="100%" height="450" style="border:0;border-radius:8px;" allowfullscreen="" loading="lazy"></iframe>'
        components.html(iframe, height=460, scrolling=False)
        st.markdown(f"[Open in Google Maps]({embed_url})", unsafe_allow_html=True)

# ---------- End ----------

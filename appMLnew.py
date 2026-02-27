import streamlit as st
import pandas as pd
import io
import base64
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Bluetik Ads AI", layout="wide", page_icon="⚡")


# -------------------------------------------------------
# BACKGROUND + GLOBAL STYLING
# -------------------------------------------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    try:
        bin_str = get_base64_of_bin_file(png_file)
        bg_css = f"""
        background-image: linear-gradient(rgba(11,15,25,0.88), rgba(11,15,25,0.88)),
        url("data:image/jpg;base64,{bin_str}");
        """
    except:
        bg_css = "background-color: #0b0f19;"

    custom_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

    body {{
        font-family: 'Poppins', sans-serif;
    }}

    .stApp {{
        {bg_css}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #e0e0e0;
    }}

    /* ================= SIDEBAR STYLING ================= */

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f172a 0%, #0b0f19 100%);
        padding: 2rem 1.5rem;
        border-right: 1px solid rgba(255,255,255,0.08);
    }}

    [data-testid="stSidebar"] h3 {{
        color: white !important;
        font-weight: 600;
    }}

    [data-testid="stSidebar"] label {{
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }}

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select {{
        background-color: rgba(255,255,255,0.05) !important;
        color: white !important;
        border-radius: 10px !important;
    }}

    [data-testid="stSidebar"] .stSlider > div {{
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 6px;
    }}

    [data-testid="stSidebar"] .stButton > button {{
        background: linear-gradient(90deg, #0052cc 0%, #00d2ff 100%);
        color: white !important;
        border-radius: 30px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        width: 100%;
        border: none;
        margin-top: 1rem;
    }}

    [data-testid="stSidebar"] .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 210, 255, 0.4);
    }}

    /* ================= HEADERS ================= */

    h1, h2, h3 {{
        color: white !important;
        font-weight: 800 !important;
    }}

    .agency-blue {{
        background: linear-gradient(90deg, #0052cc 0%, #00d2ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    /* ================= FILE UPLOADER ================= */

    [data-testid="stFileUploader"] {{
        background-color: rgba(22,30,46,0.75);
        border: 1px dashed #00d2ff;
        border-radius: 15px;
        padding: 1rem;
    }}

    /* ================= METRICS ================= */

    div[data-testid="stMetricLabel"] {{
        color: white !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: #00d2ff !important;
        font-size: 1.8rem !important;
    }}

    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


set_background("istock-952679588-800x521.jpg")


# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown(
    "<h1>⚡ <span class='agency-blue'>BLUETIK</span> Campaign Intelligence</h1>",
    unsafe_allow_html=True,
)

st.write(
    "Upload your Meta Ads Excel report to isolate top performers and forecast future results."
)

st.markdown("<hr style='border:1px solid #1f2937;'>", unsafe_allow_html=True)


# -------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------
uploaded_file = st.file_uploader("Drop your Meta Ads Report (.xlsx) here", type=["xlsx"])


if uploaded_file:

    df = pd.read_excel(uploaded_file)

    amount_spent_col = next((col for col in df.columns if "amount spent" in col.lower()), None)
    objective_col = next((col for col in df.columns if "objective" in col.lower()), None)

    if not amount_spent_col or not objective_col:
        st.error("Required columns not found.")
        st.stop()

    tab1, tab2 = st.tabs(["📊 Campaign Filter", "🤖 AI Predictor"])

    # ===================================================
    # TAB 1 – FILTER SECTION
    # ===================================================
    with tab1:

        st.sidebar.markdown("### 🎯 Filter Settings")

        unique_objectives = df[objective_col].dropna().unique().tolist()

        objective_input = st.sidebar.selectbox(
            "Campaign Objective", unique_objectives
        )

        min_spend = st.sidebar.number_input("Min Amount spent (INR)", value=0.0)
        max_spend = st.sidebar.number_input("Max Amount spent (INR)", value=5000.0)

        top_n = st.sidebar.slider(
            "How many top campaigns?",
            min_value=1,
            max_value=50,
            value=5,
        )

        sort_metric = st.sidebar.selectbox(
            "Metric to sort by (lowest to highest)", df.columns.tolist()
        )

        if st.sidebar.button("Generate Report"):

            filtered_data = df[
                (df[objective_col].astype(str).str.contains(objective_input, case=False, na=False))
                & (df[amount_spent_col] >= min_spend)
                & (df[amount_spent_col] <= max_spend)
            ]

            if filtered_data.empty:
                st.warning("No campaigns found.")
            else:
                top_campaigns = (
                    filtered_data.sort_values(by=sort_metric).head(top_n)
                )
                st.success(f"Top {len(top_campaigns)} campaigns isolated!")
                st.dataframe(top_campaigns, use_container_width=True)

    # ===================================================
    # TAB 2 – AI PREDICTOR
    # ===================================================
    with tab2:

        results_col = next((col for col in df.columns if col.lower() == "results"), None)
        reach_col = next((col for col in df.columns if col.lower() == "reach"), None)
        impr_col = next((col for col in df.columns if col.lower() == "impressions"), None)
        clicks_col = next((col for col in df.columns if "clicks" in col.lower()), None)

        targets = {}
        if results_col: targets["Results"] = results_col
        if reach_col: targets["Reach"] = reach_col
        if impr_col: targets["Impressions"] = impr_col
        if clicks_col: targets["Clicks"] = clicks_col

        if not targets:
            st.warning("Required columns not found for prediction.")
        else:
            ml_df = df[[objective_col, amount_spent_col] + list(targets.values())].dropna()

            unique_objectives = df[objective_col].dropna().unique().tolist()
            ml_objective = st.selectbox("Train AI on Objective:", unique_objectives)

            train_data = ml_df[ml_df[objective_col] == ml_objective]

            if len(train_data) < 10:
                st.warning("Minimum 10 campaigns required.")
            else:
                X = train_data[[amount_spent_col]]
                y = train_data[list(targets.values())]

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)

                simulated_budget = st.slider(
                    "If I allocate this much budget...",
                    min_value=10.0,
                    max_value=float(X.max() * 2),
                    value=float(X.mean()),
                    step=50.0,
                )

                predictions = model.predict([[simulated_budget]])[0]
                pred_dict = dict(zip(targets.keys(), predictions))

                pred_results = pred_dict.get("Results", 0)
                pred_reach = pred_dict.get("Reach", 0)
                pred_impr = pred_dict.get("Impressions", 0)
                pred_clicks = pred_dict.get("Clicks", 0)

                est_cpa = simulated_budget / pred_results if pred_results else 0
                est_cpc = simulated_budget / pred_clicks if pred_clicks else 0
                est_ctr = (pred_clicks / pred_impr) * 100 if pred_impr else 0

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if "Reach" in targets:
                        st.metric("Predicted Reach", f"{int(pred_reach):,}")
                    if "Results" in targets:
                        st.metric("Predicted Results", f"{int(pred_results):,}")

                with col2:
                    if "Impressions" in targets:
                        st.metric("Predicted Impressions", f"{int(pred_impr):,}")
                    if "Results" in targets:
                        st.metric("Est. Cost per Result", f"₹{est_cpa:.2f}")

                with col3:
                    if "Clicks" in targets:
                        st.metric("Predicted Clicks", f"{int(pred_clicks):,}")
                        st.metric("Est. CPC", f"₹{est_cpc:.2f}")

                with col4:
                    if "Clicks" in targets and "Impressions" in targets:
                        st.metric("Est. CTR", f"{est_ctr:.2f}%")

                st.caption("Predictions are based on historical campaign trends.")
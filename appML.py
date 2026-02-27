import streamlit as st
import pandas as pd
import io
import base64
from sklearn.ensemble import RandomForestRegressor

# Set up the page layout
st.set_page_config(page_title="Bluetik Ads AI", layout="wide", page_icon="⚡")

# Function to convert the image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    try:
        bin_str = get_base64_of_bin_file(png_file)
        # If the image is found, use it with a dark overlay
        bg_css = f'background-image: linear-gradient(rgba(11, 15, 25, 0.85), rgba(11, 15, 25, 0.85)), url("data:image/jpg;base64,{bin_str}");'
    except FileNotFoundError:
        # If the image is missing (like before you upload it to GitHub), fall back to a sleek dark color
        bg_css = 'background-color: #0b0f19;'

    custom_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif !important;
    }}
    
    .stApp {{
        {bg_css}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #e0e0e0;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: rgba(17, 24, 39, 0.95);
        border-right: 1px solid #1f2937;
    }}

    h1, h2, h3 {{
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }}
    
    .agency-blue {{
        color: #00d2ff; 
        background: -webkit-linear-gradient(90deg, #0052cc 0%, #00d2ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .stButton>button {{
        background: linear-gradient(90deg, #0052cc 0%, #00d2ff 100%);
        color: white !important;
        border: none;
        border-radius: 30px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 210, 255, 0.4);
    }}

    .stDownloadButton>button {{
        background-color: transparent;
        color: #00d2ff !important;
        border: 1px solid #00d2ff;
        border-radius: 30px;
    }}
    .stDownloadButton>button:hover {{
        background-color: #00d2ff;
        color: #0b0f19 !important;
    }}

    .stFileUploader>div>div {{
        background-color: rgba(22, 30, 46, 0.7);
        border: 1px dashed #0052cc;
        border-radius: 15px;
    }}
    
    [data-testid="stDataFrame"] {{
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #1f2937;
        background-color: rgba(11, 15, 25, 0.9);
    }}
    
    /* Style the tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: rgba(0, 210, 255, 0.1);
        border-bottom: 2px solid #00d2ff !important;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Call the background setup using the cloud-ready relative path
set_background('istock-952679588-800x521.jpg')

# Custom Header
st.markdown("<h1>⚡ <span class='agency-blue'>BLUETIK</span> Campaign Intelligence</h1>", unsafe_allow_html=True)
st.write("Upload your Meta Ads Excel report to isolate top performers and forecast future results.")
st.markdown("<hr style='border: 1px solid #1f2937; margin-bottom: 2rem;'>", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader("Drop your Meta Ads Report (.xlsx) here", type=['xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    amount_spent_col = next((col for col in df.columns if 'amount spent' in col.lower()), None)
    objective_col = next((col for col in df.columns if 'objective' in col.lower()), None)
    results_col = next((col for col in df.columns if col.strip().lower() == 'results'), None)

    if not amount_spent_col or not objective_col:
        st.error("Could not find an 'Amount spent' or 'Objective' column in this file.")
        st.stop()

    # Create Tabs
    tab1, tab2 = st.tabs(["📊 Campaign Filter", "🤖 AI Predictor"])

    # --- TAB 1: DESCRIPTIVE FILTER ---
    with tab1:
        st.sidebar.markdown("<h3>🎯 Filter Settings</h3>", unsafe_allow_html=True)
        unique_objectives = df[objective_col].dropna().unique().tolist()
        objective_input = st.sidebar.selectbox("1. Campaign Objective", unique_objectives)
        
        min_spend = st.sidebar.number_input(f"2. Min {amount_spent_col}", value=0.0)
        max_spend = st.sidebar.number_input(f"3. Max {amount_spent_col}", value=5000.0)
        top_n = st.sidebar.slider("4. How many top campaigns?", min_value=1, max_value=50, value=5)
        sort_metric = st.sidebar.selectbox("5. Metric to sort by (lowest to highest)", df.columns.tolist(), index=df.columns.tolist().index('Cost per result') if 'Cost per result' in df.columns else 0)

        if st.sidebar.button("Generate Report", type="primary"):
            with st.spinner("Crunching the data..."):
                filtered_data = df[
                    (df[objective_col].astype(str).str.contains(objective_input, case=False, na=False)) & 
                    (df[amount_spent_col] >= min_spend) & 
                    (df[amount_spent_col] <= max_spend)
                ]

                if filtered_data.empty:
                    st.warning("No campaigns found matching these filters.")
                else:
                    top_campaigns = filtered_data.sort_values(by=sort_metric, ascending=True, na_position='last').head(top_n)
                    existing_columns = [col for col in df.columns if col in top_campaigns.columns]
                    # Limiting to first 28 cols to keep the UI clean
                    final_result = top_campaigns[existing_columns[:28]] 

                    st.success(f"Successfully isolated the top {len(final_result)} campaigns!")
                    st.dataframe(final_result, use_container_width=True)

                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        final_result.to_excel(writer, index=False, sheet_name='Filtered Campaigns')
                    
                    st.write("") 
                    st.download_button(
                        label="📥 Export to Excel",
                        data=buffer.getvalue(),
                        file_name=f"bluetik_filtered_{objective_input}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    # --- TAB 2: PREDICTIVE ML MODEL ---
    with tab2:
        st.markdown("### 🤖 Future Performance Simulator")
        st.write("Using a Machine Learning algorithm (Random Forest Regressor), we can learn from your historical campaigns to predict future outcomes based on budget changes.")
        
        if not results_col:
            st.warning("Could not find a 'Results' column in this dataset to train the AI on.")
        else:
            # Clean data for ML
            ml_df = df[[objective_col, amount_spent_col, results_col]].dropna()
            
            # Select objective to train on
            ml_objective = st.selectbox("Select Objective to Train the AI On:", unique_objectives, key="ml_obj")
            
            # Filter data for training
            train_data = ml_df[ml_df[objective_col] == ml_objective]
            
            if len(train_data) < 10:
                st.warning(f"Not enough historical data for '{ml_objective}'. The AI needs at least 10 campaigns to make accurate predictions.")
            else:
                st.success(f"✅ AI trained successfully on {len(train_data)} historical '{ml_objective}' campaigns.")
                
                # Setup ML X (Features) and y (Target)
                X = train_data[[amount_spent_col]]
                y = train_data[results_col]
                
                # Train the Random Forest
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                st.markdown("<hr style='border: 1px dashed #1f2937;'>", unsafe_allow_html=True)
                st.markdown("#### 🔮 Simulation Engine")
                
                # Interactive simulation slider
                avg_spend = float(X[amount_spent_col].mean())
                max_sim_spend = float(X[amount_spent_col].max() * 2) 
                
                simulated_budget = st.slider(
                    "If I allocate this much budget...", 
                    min_value=10.0, 
                    max_value=max_sim_spend, 
                    value=avg_spend, 
                    step=50.0
                )
                
                # Make the prediction
                prediction = model.predict([[simulated_budget]])[0]
                est_cpa = simulated_budget / prediction if prediction > 0 else 0
                
                # Display results nicely
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Predicted Results:**\n### {int(prediction)} results")
                with col2:
                    st.warning(f"**Estimated Cost per Result:**\n### ₹{est_cpa:.2f}")
                
                st.caption("Note: Predictions are based on historical algorithmic trends and assume market conditions remain stable.")

import streamlit as st
import pandas as pd
import io

# Set up the page layout
st.set_page_config(page_title="Meta Ads Filter", layout="wide")
st.title("📊 Meta Ads Campaign Filter")
st.write("Upload your Meta Ads Excel report to easily filter and export top performing campaigns.")

# 1. File Uploader (Much better for clients than hardcoding a file path!)
uploaded_file = st.file_uploader("Upload your Meta Ads Report (.xlsx)", type=['xlsx'])

if uploaded_file is not None:
    # Read the uploaded Excel file
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Auto-detect core columns
    amount_spent_col = next((col for col in df.columns if 'amount spent' in col.lower()), None)
    objective_col = next((col for col in df.columns if 'objective' in col.lower()), None)

    if not amount_spent_col or not objective_col:
        st.error("Could not find an 'Amount spent' or 'Objective' column in this file.")
        st.stop()

    st.sidebar.header("🎯 Filter Settings")

    # 2. Get unique objectives automatically for a dropdown menu!
    unique_objectives = df[objective_col].dropna().unique().tolist()
    objective_input = st.sidebar.selectbox("1. Campaign Objective", unique_objectives)
    
    # 3. Numeric inputs for spend
    min_spend = st.sidebar.number_input(f"2. Min {amount_spent_col}", value=0.0)
    max_spend = st.sidebar.number_input(f"3. Max {amount_spent_col}", value=5000.0)
    
    # 4. Top N slider
    top_n = st.sidebar.slider("4. How many top campaigns?", min_value=1, max_value=50, value=5)
    
    # 5. Metric dropdown
    sort_metric = st.sidebar.selectbox("5. Metric to sort by (lowest to highest)", df.columns.tolist(), index=df.columns.tolist().index('Cost per result') if 'Cost per result' in df.columns else 0)

    # Filter Button
    if st.sidebar.button("Run Filter", type="primary"):
        with st.spinner("Processing data..."):
            
            # Filter logic
            filtered_data = df[
                (df[objective_col].astype(str).str.contains(objective_input, case=False, na=False)) & 
                (df[amount_spent_col] >= min_spend) & 
                (df[amount_spent_col] <= max_spend)
            ]

            if filtered_data.empty:
                st.warning(f"No campaigns found matching Objective: '{objective_input}' and Spend: {min_spend}-{max_spend}.")
            else:
                # Sort logic
                top_campaigns = filtered_data.sort_values(by=sort_metric, ascending=True, na_position='last').head(top_n)
                
                # Define columns to output
                columns_to_show = [
                    'Campaign name', 'Ad set name', 'Page name', objective_col, amount_spent_col, 
                    'Ad set budget', 'Reach', 'Impressions', 'Frequency', 'CTR (All)', 
                    'Clicks (All)', 'CPC (All)', 'Attribution setting', 'Results', 
                    'Cost per result', 'Page engagement', 'Post comments', 'Post shares', 
                    'Post saves', 'Messaging conversations started', 
                    'Cost per messaging conversation started', 'Phone calls placed', 'Leads', 
                    'Cost per Lead', 'Reporting starts', 'Reporting ends', 'placement', 'media type'
                ]
                existing_columns = [col for col in columns_to_show if col in df.columns]
                final_result = top_campaigns[existing_columns]

                st.success(f"Found the top {len(final_result)} campaigns!")
                
                # Display the interactive table
                st.dataframe(final_result, use_container_width=True)

                # Prepare the Excel file for download in memory
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    final_result.to_excel(writer, index=False, sheet_name='Filtered Campaigns')
                
                # Download Button
                st.download_button(
                    label="📥 Download Results as Excel",
                    data=buffer.getvalue(),
                    file_name=f"filtered_{objective_input}_campaigns.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
import os
import streamlit as st
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai

# --- Milestone 2: Initialize API and Models ---
# Activity 2.2: Set up Google GenAI API credentials
genai.configure(api_key='AIzaSyAjTG9m37QZNBpnsHqKyID0wulMXpiZRfw')

# Activity 2.3: Print available models for verification
models = genai.list_models()
for model in models:
    print(model)

# Activity 2.4: Initialize the Google Generative AI model
llm = GoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    google_api_key='AIzaSyAjTG9m37QZNBpnsHqKyID0wulMXpiZRfw',
    temperature=1.0
)

# --- Milestone 3: Prompt Templates ---
prompt_templates = {
    "balance_sheet": PromptTemplate(
        input_variables=["balance_sheet_data"],
        template="""Given the balance sheet data: {balance_sheet_data}, provide a clear and concise summary highlighting key financial metrics and insights."""
    ),
    "profit_loss": PromptTemplate(
        input_variables=["profit_loss_data"],
        template="""Given the profit and loss statement data: {profit_loss_data}, provide a clear and concise summary highlighting key financial metrics and insights."""
    ),
    "cash_flow": PromptTemplate(
        input_variables=["cash_flow_data"],
        template="""Given the cash flow statement data: {cash_flow_data}, provide a clear and concise summary highlighting key financial metrics and insights."""
    )
}

# --- Milestone 4: File Upload and Processing ---
def upload_files():
    balance_sheet = st.file_uploader("Upload Balance Sheet", type=["csv", "xlsx"])
    profit_loss = st.file_uploader("Upload Profit and Loss Statement", type=["csv", "xlsx"])
    cash_flow = st.file_uploader("Upload Cash Flow Statement", type=["csv", "xlsx"])
    return balance_sheet, profit_loss, cash_flow

def load_file(file):
    if file is not None:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
    return None

# --- Milestone 5: Generate Summary ---
def generate_summary(prompt_type, data):
    if data is not None:
        data_dict = data.to_dict()
        # Use formatting logic based on prompt type
        if prompt_type == "balance_sheet":
            prompt = prompt_templates[prompt_type].format(balance_sheet_data=data_dict)
        elif prompt_type == "profit_loss":
            prompt = prompt_templates[prompt_type].format(profit_loss_data=data_dict)
        elif prompt_type == "cash_flow":
            prompt = prompt_templates[prompt_type].format(cash_flow_data=data_dict)
        
        response = llm.invoke(prompt) # Updated from llm(prompt) for newer LangChain versions
        return response
    return "No data provided for this section."

# --- Milestone 6: Visualization ---
def create_visuals(data, title):
    if data is not None:
        st.subheader(title)
        st.write(data)
        # Only plot numeric columns to avoid errors
        numeric_data = data.select_dtypes(include=['number'])
        if not numeric_data.empty:
            st.line_chart(numeric_data)
        else:
            st.info("No numeric data available for line chart.")

# --- Milestone 7: App Layout and Interaction ---
st.title("Gemini Pro Financial Decoder")

balance_sheet_file, profit_loss_file, cash_flow_file = upload_files()

if st.button("Generate Reports"):
    with st.spinner("Generating summaries and visualizations..."):
        # Load the data
        bs_data = load_file(balance_sheet_file)
        pl_data = load_file(profit_loss_file)
        cf_data = load_file(cash_flow_file)

        # 1. Balance Sheet Section
        if bs_data is not None:
            summary = generate_summary("balance_sheet", bs_data)
            st.subheader("Balance Sheet Summary")
            st.write(summary)
            create_visuals(bs_data, "Balance Sheet Visualization")
        
        # 2. Profit and Loss Section
        if pl_data is not None:
            summary = generate_summary("profit_loss", pl_data)
            st.subheader("Profit and Loss Summary")
            st.write(summary)
            create_visuals(pl_data, "Profit and Loss Visualization")

        # 3. Cash Flow Section
        if cf_data is not None:
            summary = generate_summary("cash_flow", cf_data)
            st.subheader("Cash Flow Summary")
            st.write(summary)
            create_visuals(cf_data, "Cash Flow Visualization")
            
        if all(x is None for x in [bs_data, pl_data, cf_data]):
            st.error("Please upload at least one file to generate a report.")
"""
Invictus — Smart Loader Agent (LLM-Native)
Uses LLM to surgically extract portfolio data from messy/arbitrary brokerage CSVs.
"""
import pandas as pd
import json
import streamlit as st
import io
from typing import Dict, Any, Optional, Tuple, List
from invictus.config import OPENAI_API_KEY, LLM_MODEL

def extract_portfolio_with_llm(raw_content: str) -> List[Dict[str, Any]]:
    """Sends raw CSV content to LLM to extract clean holdings."""
    if not OPENAI_API_KEY:
        return []

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # We take the first 150 lines which is enough for most portfolios
        content_sample = "\n".join(raw_content.splitlines()[:150])
        
        prompt = f"""You are an institutional data engineer. 
Extract the portfolio holdings from the following raw brokerage CSV content.
Ignore all summary headers, account info, 'CASH' rows, and totals. 
Identify the columns for Ticker, Quantity (Shares), and Cost Basis (Entry Price).

Raw Content:
{content_sample}

Output strictly a JSON list of objects:
[
  {{"Ticker": "SYMBOL", "Shares": 0.0, "CostBasis": 0.0}},
  ...
]
"""
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"} if "gpt-4o" in LLM_MODEL else None
        )
        
        raw_json = response.choices[0].message.content
        data = json.loads(raw_json)
        
        # Handle cases where LLM returns {"holdings": [...]} or just [...]
        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], list):
                    return data[key]
        return data if isinstance(data, list) else []
        
    except Exception as e:
        st.error(f"LLM Extraction failed: {e}")
        return []

def smart_load_portfolio(file_content: str, manual_mapping: Optional[Dict] = None) -> pd.DataFrame:
    """Intelligently loads a portfolio using LLM-native extraction."""
    try:
        # 1. Use LLM to extract data directly from the messy raw text
        holdings_list = extract_portfolio_with_llm(file_content)
        
        if not holdings_list:
            raise ValueError("LLM could not extract any holdings from the file.")
            
        df = pd.DataFrame(holdings_list)
        
        # 2. Standardize column names (LLM usually follows the prompt, but we check)
        rename_map = {}
        for col in df.columns:
            if col.lower() in ["ticker", "symbol", "instrument"]: rename_map[col] = "Ticker"
            if col.lower() in ["shares", "qty", "quantity"]: rename_map[col] = "Shares"
            if col.lower() in ["costbasis", "price", "basis", "cost"]: rename_map[col] = "CostBasis"
        
        df = df.rename(columns=rename_map)
        
        # 3. Final sanitization
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
        # Strip trailing text like 'AAPL Apple Inc'
        df["Ticker"] = df["Ticker"].apply(lambda x: x.split()[0] if " " in x else x)
        
        df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce")
        df["CostBasis"] = pd.to_numeric(df["CostBasis"], errors="coerce").fillna(0.0)
        
        df = df.dropna(subset=["Ticker", "Shares"])
        # Remove junk rows
        df = df[df["Ticker"].str.len() <= 6]
        df = df[df["Shares"] > 0]
        
        st.toast(f"Smart Loader: Extracted {len(df)} positions via LLM.")
        return df[["Ticker", "Shares", "CostBasis"]]
        
    except Exception as e:
        raise ValueError(f"Smart Loader Error: {str(e)}")

def analyze_csv_columns(file_content: str) -> Tuple[Dict[str, Optional[str]], float, list]:
    """Fallback for manual mapping if needed (uses basic pandas read)."""
    try:
        # Just to get the columns for the UI
        df = pd.read_csv(io.StringIO(file_content), sep=None, on_bad_lines='skip', engine='python', nrows=10)
        return {"Ticker": None, "Shares": None, "CostBasis": None}, 0.5, list(df.columns)
    except Exception:
        return {"Ticker": None, "Shares": None, "CostBasis": None}, 0.0, []

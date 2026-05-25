"""
Invictus — Smart Loader Agent (LLM-Native)
Uses LLM to surgically extract portfolio data from messy/arbitrary brokerage CSVs.
"""
import pandas as pd
import json
import streamlit as st
import io
from typing import Dict, Any, Optional, Tuple, List
from invictus.llm import call_llm_json, llm_available

def extract_portfolio_with_llm(raw_content: str) -> List[Dict[str, Any]]:
    """Sends raw CSV content to LLM to extract clean holdings."""
    if not llm_available():
        return []

    try:
        content_sample = "\n".join(raw_content.splitlines()[:150])

        prompt = f"""You are an institutional data engineer.
Extract the portfolio holdings from the following raw brokerage CSV content.
Ignore all summary headers, account info, 'CASH' rows, and totals.
Identify the columns for Ticker, Quantity (Shares), and Cost Basis (Entry Price).

Raw Content:
{content_sample}

Output strictly a JSON object with a "holdings" key containing a list:
{{"holdings": [{{"Ticker": "SYMBOL", "Shares": 0.0, "CostBasis": 0.0}}, ...]}}
"""
        data = call_llm_json(prompt)

        # Handle cases where LLM returns {"holdings": [...]} or just [...]
        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], list):
                    return data[key]
        return data if isinstance(data, list) else []

    except Exception as e:
        st.error(f"LLM Extraction failed: {e}")
        return []

def _fallback_csv_parse(file_content: str) -> pd.DataFrame:
    """
    Pandas-based fallback when LLM is unavailable.
    Auto-detects Ticker/Shares/CostBasis columns by name matching.
    """
    df = pd.read_csv(io.StringIO(file_content), sep=None, on_bad_lines='skip', engine='python')
    df.columns = [str(c).strip() for c in df.columns]

    # Column name matching — case-insensitive, partial match
    _ticker_hints = ["ticker", "symbol", "instrument", "stock", "scrip", "name"]
    _shares_hints = ["shares", "qty", "quantity", "units", "amount", "holdings"]
    _cost_hints = ["cost", "basis", "price", "avg", "entry", "purchase"]

    def _find_col(hints):
        for col in df.columns:
            cl = col.lower()
            for h in hints:
                if h in cl:
                    return col
        return None

    ticker_col = _find_col(_ticker_hints)
    shares_col = _find_col(_shares_hints)
    cost_col = _find_col(_cost_hints)

    if not ticker_col or not shares_col:
        # Last resort: assume first column is ticker, second is shares
        if len(df.columns) >= 2:
            ticker_col = df.columns[0]
            shares_col = df.columns[1]
            cost_col = df.columns[2] if len(df.columns) >= 3 else None
        else:
            raise ValueError(
                f"Cannot auto-detect columns. Found: {list(df.columns)}. "
                f"CSV must have at least Ticker and Shares columns."
            )

    result = pd.DataFrame()
    result["Ticker"] = df[ticker_col].astype(str).str.strip().str.upper()
    result["Ticker"] = result["Ticker"].apply(lambda x: x.split()[0] if " " in x else x)
    result["Shares"] = pd.to_numeric(df[shares_col], errors="coerce")
    result["CostBasis"] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0) if cost_col else 0.0

    result = result.dropna(subset=["Ticker", "Shares"])
    result = result[result["Ticker"].str.len() <= 6]
    result = result[result["Shares"] > 0]
    result = result[~result["Ticker"].isin(["CASH", "TOTAL", "NAN", ""])]

    return result[["Ticker", "Shares", "CostBasis"]]


def smart_load_portfolio(file_content: str, manual_mapping: Optional[Dict] = None) -> pd.DataFrame:
    """
    Intelligently loads a portfolio.
    Strategy: LLM first → pandas fallback if LLM unavailable/fails.
    """
    source = "unknown"

    # 1. Try LLM extraction
    try:
        holdings_list = extract_portfolio_with_llm(file_content)
        if holdings_list:
            df = pd.DataFrame(holdings_list)
            rename_map = {}
            for col in df.columns:
                cl = col.lower()
                if cl in ["ticker", "symbol", "instrument"]: rename_map[col] = "Ticker"
                if cl in ["shares", "qty", "quantity"]: rename_map[col] = "Shares"
                if cl in ["costbasis", "price", "basis", "cost"]: rename_map[col] = "CostBasis"
            df = df.rename(columns=rename_map)

            if "Ticker" in df.columns and "Shares" in df.columns:
                df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
                df["Ticker"] = df["Ticker"].apply(lambda x: x.split()[0] if " " in x else x)
                df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce")
                if "CostBasis" not in df.columns:
                    df["CostBasis"] = 0.0
                df["CostBasis"] = pd.to_numeric(df["CostBasis"], errors="coerce").fillna(0.0)
                df = df.dropna(subset=["Ticker", "Shares"])
                df = df[df["Ticker"].str.len() <= 6]
                df = df[df["Shares"] > 0]
                if len(df) > 0:
                    source = "LLM"
                    st.toast(f"Smart Loader: Extracted {len(df)} positions via AI.")
                    return df[["Ticker", "Shares", "CostBasis"]]
    except Exception:
        pass  # fall through to pandas

    # 2. Pandas fallback
    try:
        df = _fallback_csv_parse(file_content)
        if len(df) > 0:
            source = "auto-detect"
            st.toast(f"Smart Loader: Extracted {len(df)} positions via auto-detection.")
            return df
    except Exception as e:
        raise ValueError(
            f"Could not parse CSV. AI extraction unavailable, auto-detection failed: {str(e)}. "
            f"Ensure your CSV has columns for Ticker/Symbol and Shares/Quantity."
        )

def analyze_csv_columns(file_content: str) -> Tuple[Dict[str, Optional[str]], float, list]:
    """Fallback for manual mapping if needed (uses basic pandas read)."""
    try:
        # Just to get the columns for the UI
        df = pd.read_csv(io.StringIO(file_content), sep=None, on_bad_lines='skip', engine='python', nrows=10)
        return {"Ticker": None, "Shares": None, "CostBasis": None}, 0.5, list(df.columns)
    except Exception:
        return {"Ticker": None, "Shares": None, "CostBasis": None}, 0.0, []

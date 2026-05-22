"""
Invictus — Financial RAG Agent
Ingests 10-K text files, chunks them, builds a vector index,
and retrieves relevant sections for structured analysis.

For tickers without filings, returns a notice.
Uses TF-IDF similarity as fallback when FAISS/embeddings unavailable.
"""
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from invictus.agents.graph_state import PortfolioState
from invictus.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL


FILINGS_DIR = Path(__file__).parent / "filings"

# Queries for structured extraction
RAG_QUERIES = {
    "business_drivers": "What are the key business drivers and revenue sources?",
    "risk_factors": "What are the main risk factors and threats to the business?",
    "margin_signals": "What are the gross margin and profitability trends? Are margins expanding or compressing?",
    "competitive_moat": "What is the competitive advantage or moat? What differentiates this company?",
    "capital_allocation": "How does management allocate capital? Share buybacks, dividends, R&D, CapEx?",
    "management_tone": "What is management's tone and outlook? Are they optimistic or cautious?",
    "thesis_risks": "What could break the investment thesis? What are the biggest uncertainties?",
}


def _load_filings() -> Dict[str, str]:
    """Load all 10-K text files from the filings directory."""
    filings = {}
    if not FILINGS_DIR.exists():
        return filings
    for f in FILINGS_DIR.glob("*_10K.txt"):
        ticker = f.stem.replace("_10K", "")
        filings[ticker] = f.read_text(encoding="utf-8")
    return filings


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by paragraph then by size."""
    # Split on double newlines (paragraphs) first
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If paragraph itself is too long, split by sentences
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) <= chunk_size:
                        current_chunk += " " + sent if current_chunk else sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _build_tfidf_index(chunks: List[str]) -> tuple:
    """Build TF-IDF vectorizer and matrix for similarity search."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix


def _retrieve(query: str, vectorizer, tfidf_matrix, chunks: List[str], top_k: int = TOP_K_RETRIEVAL) -> List[str]:
    """Retrieve top-k most relevant chunks for a query."""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices if similarities[i] > 0.05]


def _extract_insights(ticker: str, text: str) -> Dict[str, Any]:
    """Extract structured insights from a 10-K filing using retrieval."""
    chunks = _chunk_text(text)
    if not chunks:
        return {"error": "No content to analyze"}

    vectorizer, tfidf_matrix = _build_tfidf_index(chunks)

    insights = {}
    for key, query in RAG_QUERIES.items():
        retrieved = _retrieve(query, vectorizer, tfidf_matrix, chunks, top_k=2)
        insights[key] = {
            "query": query,
            "context": "\n\n".join(retrieved) if retrieved else "No relevant context found.",
            "n_chunks": len(retrieved),
        }

    # Extract key financial metrics if present
    metrics = {}
    full_text = text.lower()
    patterns = {
        "revenue": r"net revenue[:\s]*\$?([\d,.]+)\s*billion",
        "net_income": r"net income[:\s]*\$?([\d,.]+)\s*billion",
        "gross_margin": r"gross margin[:\s]*([\d.]+)%",
        "operating_margin": r"operating margin[:\s]*([\d.]+)%",
        "free_cash_flow": r"free cash flow[:\s]*\$?([\d,.]+)\s*billion",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, full_text)
        if match:
            metrics[name] = match.group(1)

    return {
        "insights": insights,
        "metrics": metrics,
        "n_chunks": len(chunks),
        "text_length": len(text),
    }


def retrieve_10k_context(state: PortfolioState) -> PortfolioState:
    """
    Run RAG analysis on available 10-K filings for portfolio tickers.
    """
    holdings = state.holdings
    weights = state.weights

    if holdings is None:
        raise ValueError("Holdings required for RAG analysis.")

    tickers = holdings["Ticker"].tolist() if isinstance(holdings, pd.DataFrame) else list(weights.keys())

    # Load available filings
    filings = _load_filings()
    available = [t for t in tickers if t in filings]
    missing = [t for t in tickers if t not in filings]

    results = {}
    for ticker in available:
        results[ticker] = _extract_insights(ticker, filings[ticker])

    state.rag_insights = {
        "results": results,
        "available_tickers": available,
        "missing_tickers": missing,
        "coverage": f"{len(available)}/{len(tickers)} tickers with 10-K filings",
    }

    return state

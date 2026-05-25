"""
invictus.design.styles
=====================
CSS stylesheet and injection.
"""
import streamlit as st

_STYLESHEET = """
<style id="invictus-design">
    /* ============================================================
       INVICTUS — defensive, scoped styles.
       Goals:
         1. Never use negative margins to compensate for Streamlit chrome.
         2. Never use a "* { color: ... }" rule — only target text we own.
         3. Always leave room below content for the fixed footer (60px).
         4. Let cards/labels grow to fit content; no fixed heights.
       ============================================================ */
    /* ── Institutional Design Tokens ── */
    :root {
        --brand-navy-deep: #020617;
        --brand-navy: #1e293b;
        --brand-blue: #1d4ed8;
        --brand-blue-hover: #2563eb;
        --brand-blue-light: #60a5fa;
        --brand-blue-deep: #1e3a8a;
        --brand-silver: #cbd5e1;
        --brand-silver-bright: #f8fafc;
        --capsule-bg: #eef2ff;
        --capsule-text: #1e3a8a;
        --capsule-muted: #64748b;
        --success-green: #10b981;
        --danger-red: #ef4444;
        --slate-50:#f8fafc; --slate-100:#f1f5f9; --slate-200:#e2e8f0;
        --slate-300:#cbd5e1; --slate-400:#94a3b8; --slate-500:#64748b;
        --slate-600:#475569; --slate-700:#334155; --slate-800:#1e293b;
        --slate-900:#0f172a; --slate-950:#020617;
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    /* ── Global App Reset ──
       NOTE: do not use !important on font-family. It overrides Streamlit's
       Material Symbols icon font and makes icons render as raw text
       ("arrow_down", "keyboard_double_arrow_right", etc). */
    html, body, .stApp {
        background-color: #ffffff;
        color: var(--slate-900);
        font-family: var(--font-sans);
        -webkit-font-smoothing: antialiased;
    }

    /* ── Sidebar (Light Surface) ──
       overflow-y: visible suppresses the unnecessary inner scrollbar; the
       outer page scroll picks up anything that doesn't fit. */
    [data-testid='stSidebar'],
    [data-testid='stSidebarContent'],
    [data-testid='stSidebarUserContent'] {
        background-color: #ffffff !important;
        border-right: 1px solid var(--slate-200);
    }
    [data-testid='stSidebar'] > div:first-child { overflow-y: visible !important; }
    [data-testid='stSidebarContent'] { overflow-y: visible !important; }

    /* Hide the keyboard_double_arrow_right collapse-control row at the very
       top of the sidebar — it's noise and on some Streamlit versions the
       icon ligature fails to render as a glyph. */
    [data-testid='stSidebarCollapseButton'],
    [data-testid='stSidebarCollapsedControl'] { display: none !important; }

    /* Tighten all vertical gaps inside the sidebar */
    [data-testid='stSidebar'] [data-testid='stVerticalBlock'] {
        gap: 0.25rem !important;
    }

    /* Sidebar headings only (do NOT target every descendant — it breaks widgets). */
    [data-testid='stSidebar'] h1,
    [data-testid='stSidebar'] h2,
    [data-testid='stSidebar'] h3,
    [data-testid='stSidebar'] h4 { color: var(--slate-900) !important; }

    /* ── Main canvas: leave room for the fixed 28px footer + breathing space ── */
    .stApp [data-testid='stAppViewContainer'] .main .block-container {
        padding-bottom: 60px !important;
    }

    /* Sidebar Brand Block */
    .sidebar-brand {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 0 0 6px 0;
        border-bottom: 1px solid var(--slate-200);
        margin: 0 0 4px 0;
    }
    .sidebar-brand img {
        height: 115px !important;
        width: auto;
        margin: 0px 0 -4px 0;
        filter: drop-shadow(0 2px 8px rgba(29,78,216,0.3));
    }
    .sidebar-brand-name {
        color: var(--slate-900) !important;
        font-size: 22px !important;
        font-weight: 900 !important;
        letter-spacing: 4px !important;
        text-transform: uppercase !important;
        line-height: 1.1;
    }
    .sidebar-brand-sub {
        color: var(--capsule-muted) !important;
        font-size: 11px !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    /* ── Capsule pattern — ONLY for our explicit .agent-panel blocks. ──
       Do NOT auto-apply to every st.container(border=True) in the sidebar:
       that breaks the file-uploader's internal wrapper and clips its text. */
    .agent-panel {
        background-color: var(--capsule-bg) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
    }

    .agent-panel-title {
        color: var(--capsule-muted) !important;
        font-weight: 700 !important;
        font-size: 10px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-bottom: 8px !important;
        border-bottom: 1px solid rgba(0,0,0,0.05) !important;
        padding-bottom: 4px !important;
        display: block !important;
    }

    /* Text inside our capsules ONLY (do not touch widget internals). */
    .agent-panel label,
    .agent-panel p,
    .agent-panel span {
        color: var(--capsule-text) !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        line-height: 1.4;
    }

    .agent-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
        padding: 3px 0;
        font-size: 11px !important;
    }

    .snapshot-value {
        color: var(--capsule-text) !important;
        font-size: 13px !important;
        font-weight: 800 !important;
        font-variant-numeric: tabular-nums;
        font-family: var(--font-mono);
    }

    /* ── BUTTON (Primary, solid brand-blue) ── */
    [data-testid='stSidebar'] .stButton > button {
        background: var(--brand-blue) !important;
        color: #ffffff !important;
        border: 1px solid var(--brand-blue) !important;
        font-weight: 700 !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        height: 36px !important;
        line-height: 36px !important;
        padding: 0 16px !important;
        margin-top: 4px !important;
        border-radius: 6px !important;
        width: 100% !important;
        box-shadow: 0 1px 2px rgba(29,78,216,0.20) !important;
        transition: background-color .12s ease, box-shadow .12s ease !important;
    }
    [data-testid='stSidebar'] .stButton > button:hover {
        background: var(--brand-blue-hover) !important;
        border-color: var(--brand-blue-hover) !important;
        box-shadow: 0 2px 6px rgba(29,78,216,0.30) !important;
    }
    [data-testid='stSidebar'] .stButton > button:active {
        background: var(--brand-blue-deep) !important;
    }

    /* GLOBAL — Override ALL primary buttons to brand-blue (Streamlit defaults to red) */
    [data-testid='stBaseButton-primary'],
    button[kind='primary'] {
        background-color: var(--brand-blue) !important;
        border-color: var(--brand-blue) !important;
        color: #ffffff !important;
    }
    [data-testid='stBaseButton-primary']:hover,
    button[kind='primary']:hover {
        background-color: var(--brand-blue-hover) !important;
        border-color: var(--brand-blue-hover) !important;
    }
    [data-testid='stBaseButton-primary']:active,
    button[kind='primary']:active {
        background-color: var(--brand-blue-deep) !important;
        border-color: var(--brand-blue-deep) !important;
    }
    [data-testid='stBaseButton-primary']:disabled,
    button[kind='primary']:disabled {
        background-color: var(--slate-300) !important;
        border-color: var(--slate-300) !important;
        color: var(--slate-500) !important;
        cursor: not-allowed !important;
    }

    /* Sidebar inputs: file uploader, radios, selectboxes — light styling */
    [data-testid='stSidebar'] [data-testid='stFileUploader'] section {
        background: var(--slate-50) !important;
        border: 1px dashed var(--slate-300) !important;
        color: var(--slate-700) !important;
    }
    [data-testid='stSidebar'] .stRadio label {
        color: var(--slate-900) !important;
    }

    .agent-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; margin-right: 8px; flex-shrink: 0; }
    .agent-dot.active { background: #10b981; box-shadow: 0 0 6px #10b981; }
    .agent-dot.working { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; animation: agent-pulse 1.2s ease-in-out infinite; }
    .agent-dot.error { background: #ef4444; box-shadow: 0 0 6px #ef4444; }
    .agent-dot.inactive { background: #cbd5e1; }
    @keyframes agent-pulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.4); opacity: 0.5; } }

    /* Radios / tabs */
    div[data-testid='stRadio'] label[data-baseweb='radio'] div:first-child { border-color: var(--brand-blue) !important; }
    div[data-testid='stRadio'] label[data-baseweb='radio'] [aria-checked='true'] + div { background-color: var(--brand-blue) !important; }
    .stTabs [aria-selected='true'] { color: var(--brand-blue) !important; border-bottom: 3px solid var(--brand-blue) !important; }

    /* Main Canvas Section Headers */
    .section-header {
        font-size: 13px !important;
        font-weight: 800 !important;
        color: var(--brand-blue) !important;
        background-color: var(--brand-silver-bright) !important;
        border-left: 5px solid var(--brand-blue) !important;
        padding: 12px 20px !important;
        margin: 16px 0 12px 0 !important;
        border-radius: 0 8px 8px 0 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        line-height: 1.2;
    }

    /* Stale content: let Streamlit fade old content during transitions
       so tab switches look clean. Only metric-card values stay crisp. */
    .metric-card[data-stale="true"] { opacity: 1 !important; }

    /* Fixed Bottom Footer Bar — always above all Streamlit chrome. */
    .inv-footer {
        position: fixed;
        bottom: 0; left: 0; right: 0;
        z-index: 2147483647;
        height: 28px;
        background: var(--brand-silver-bright) !important;
        border-top: 1px solid var(--brand-blue) !important;
        display: flex !important;
        align-items: center;
        justify-content: space-between;
        padding: 0 16px !important;
        font-size: 9px;
        color: var(--capsule-muted) !important;
        text-transform: uppercase;
        font-weight: 700 !important;
        letter-spacing: 0.08em;
        pointer-events: none;
    }
    .inv-footer span { color: var(--capsule-muted) !important; }
    .inv-footer .sep { color: var(--brand-blue) !important; font-weight: 900; }

    /* ── Sidebar Navigation — Copilot-style tree ── */
    /* Zero out Streamlit gaps inside the nav tree */
    [data-testid='stSidebar'] [data-testid='stVerticalBlock']:has(.snav-tree) {
        gap: 0 !important;
    }
    .snav-tree { display: none !important; }
    .snav-heading-wrap, .snav-item-wrap { display: none !important; }

    /* ── HEADING buttons ── */
    [data-testid='stElementContainer']:has(.snav-heading-wrap) + [data-testid='stElementContainer'] .stButton > button,
    [data-testid='stElementContainer']:has(.snav-heading-wrap) + [data-testid='stElementContainer'] .stButton > button > div,
    [data-testid='stElementContainer']:has(.snav-heading-wrap) + [data-testid='stElementContainer'] .stButton > button p {
        text-align: left !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid='stElementContainer']:has(.snav-heading-wrap) + [data-testid='stElementContainer'] .stButton > button {
        background: transparent !important;
        border: none !important;
        border-bottom: 1px solid var(--slate-200) !important;
        box-shadow: none !important;
        padding: 10px 0 3px 0 !important;
        margin: 0 !important;
        font-size: 10px !important;
        font-weight: 800 !important;
        color: var(--slate-400) !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        min-height: 0 !important;
        height: auto !important;
        border-radius: 0 !important;
        line-height: 1.2 !important;
        width: 100% !important;
    }
    [data-testid='stElementContainer']:has(.snav-heading-wrap) + [data-testid='stElementContainer'] .stButton > button:hover {
        color: var(--brand-blue) !important;
        background: transparent !important;
    }
    [data-testid='stElementContainer']:has(.snav-heading-wrap.active) + [data-testid='stElementContainer'] .stButton > button {
        color: var(--brand-blue) !important;
        border-bottom-color: var(--brand-blue) !important;
    }

    /* ── SUB-ITEM buttons ── */
    [data-testid='stElementContainer']:has(.snav-item-wrap) + [data-testid='stElementContainer'] .stButton > button,
    [data-testid='stElementContainer']:has(.snav-item-wrap) + [data-testid='stElementContainer'] .stButton > button > div,
    [data-testid='stElementContainer']:has(.snav-item-wrap) + [data-testid='stElementContainer'] .stButton > button p {
        text-align: left !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid='stElementContainer']:has(.snav-item-wrap) + [data-testid='stElementContainer'] .stButton > button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 4px 8px 4px 20px !important;
        margin: 0 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: var(--slate-700) !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        min-height: 0 !important;
        height: auto !important;
        border-radius: 6px !important;
        line-height: 1.3 !important;
        width: 100% !important;
    }
    [data-testid='stElementContainer']:has(.snav-item-wrap) + [data-testid='stElementContainer'] .stButton > button:hover {
        background: var(--slate-50) !important;
        color: var(--slate-900) !important;
    }
    [data-testid='stElementContainer']:has(.snav-item-wrap.active) + [data-testid='stElementContainer'] .stButton > button {
        color: var(--brand-blue) !important;
        font-weight: 700 !important;
        background: rgba(29,78,216,0.06) !important;
    }

    /* ============================================================
       FLUSH-TO-TOP + TOP BAR STYLING
       ============================================================ */
    [data-testid='stHeader'] { display: none !important; }
    .stApp .main .block-container,
    .stApp [data-testid='stMain'] .block-container {
        padding-top: 0.75rem !important;
        padding-bottom: 60px !important;
    }

    /* ── Fixed Header ── */
    .inv-header-pin { display: none !important; }

    /* Fix the header container to the top of the main canvas */
    [data-testid='stVerticalBlock']:has(> [data-testid='stElementContainer'] > [data-testid='stMarkdownContainer'] > .inv-header-pin) {
        position: fixed !important;
        top: 0 !important;
        right: 0 !important;
        left: var(--sidebar-width, 245px) !important;
        z-index: 999 !important;
        background: #ffffff !important;
        padding: 8px 1rem 0 1rem !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    }

    /* Push main content below the fixed header */
    .stApp .main .block-container,
    .stApp [data-testid='stMain'] .block-container {
        padding-top: 70px !important;
    }

    /* Adapt to sidebar collapsed state */
    [data-testid='stSidebar'][aria-expanded='false'] ~ [data-testid='stMain']
    [data-testid='stVerticalBlock']:has(> [data-testid='stElementContainer'] > [data-testid='stMarkdownContainer'] > .inv-header-pin) {
        left: 0 !important;
    }
    /* Header bar — tighter gaps */
    [data-testid='stVerticalBlock']:has(.inv-header-pin) [data-testid='stHorizontalBlock'] {
        gap: 6px !important;
        align-items: center !important;
    }

    /* Load Portfolio — secondary buttons in header get silver treatment */
    [data-testid='stVerticalBlock']:has(.inv-header-pin) button[kind='secondary'] {
        background: var(--brand-silver-bright) !important;
        border: 1px solid var(--brand-silver) !important;
        color: var(--slate-700) !important;
        font-weight: 600 !important;
        font-size: 12px !important;
    }
    [data-testid='stVerticalBlock']:has(.inv-header-pin) button[kind='secondary']:hover {
        background: var(--slate-100) !important;
        border-color: var(--slate-400) !important;
    }

    /* Header primary buttons — brand blue */
    [data-testid='stVerticalBlock']:has(.inv-header-pin) button[kind='primary'] {
        background: var(--brand-blue) !important;
        border: 1px solid var(--brand-blue) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 12px !important;
    }
    [data-testid='stVerticalBlock']:has(.inv-header-pin) button[kind='primary']:hover {
        background: var(--brand-blue-hover) !important;
        border-color: var(--brand-blue-hover) !important;
    }

    [data-testid='stSidebar'] > div:first-child { padding-top: 0 !important; }
    [data-testid='stSidebarUserContent']         { padding-top: 0 !important; }
    .sidebar-brand                                 { margin-top: -18px !important; padding-top: 0 !important; }

    /* ── Commentary Box ── */
    .commentary-box {
        border-left: 3px solid var(--brand-blue);
        padding: 16px 20px;
        background: var(--brand-silver-bright);
        border-radius: 6px;
        color: #0f172a;
        font-size: 14px;
        line-height: 1.8;
        white-space: pre-wrap;
    }

    /* ── Concentration / Severity Banners ── */
    .conc-banner {
        background: #f8fafc;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .conc-banner .lvl {
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.05em;
    }
    .conc-banner .body { color: #475569; font-size: 13px; margin-top: 4px; }

    /* ── Metric Cards ──
       No fixed min-height — cards size to content so values never collide. */
    .metric-card {
        background: #ffffff !important;
        border: 1px solid var(--slate-200) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        display: flex;
        flex-direction: column;
        gap: 8px;
        overflow: hidden;
    }
    .metric-label {
        color: var(--capsule-muted) !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-value {
        color: var(--slate-900) !important;
        font-weight: 800 !important;
        font-size: 22px !important;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.01em;
        line-height: 1.15;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-value.pos { color: var(--success-green) !important; }
    .metric-value.neg { color: var(--danger-red) !important; }
    .metric-delta {
        font-size: 13px;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        line-height: 1.2;
        white-space: nowrap;
    }
    .metric-delta.pos { color: var(--success-green) !important; }
    .metric-delta.neg { color: var(--danger-red) !important; }
    .pos { color: var(--success-green) !important; }
    .neg { color: var(--danger-red) !important; }

    /* Bump metric-value coloring above any inline styles. */
    .metric-card .metric-value.pos { color: var(--success-green) !important; }
    .metric-card .metric-value.neg { color: var(--danger-red) !important; }
    .metric-card .metric-delta.pos { color: var(--success-green) !important; }
    .metric-card .metric-delta.neg { color: var(--danger-red) !important; }

    /* Material Symbols / icon spans — hard pin to the icon font so they
       never inherit our sans stack. Streamlit uses these in expanders,
       dropdowns, the sidebar collapse, and elsewhere. */
    span[data-testid*='Icon'],
    span[class*='material-symbols'],
    span[class*='MaterialIcons'],
    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        font-feature-settings: 'liga';
        letter-spacing: 0 !important;
    }
</style>
"""


def inject_styles() -> None:
    """
    Inject the full Invictus stylesheet AND mount the fixed footer into
    <body> via JS. Mounting via JS (instead of `st.markdown` at script
    bottom) keeps the footer visible during Streamlit reruns — it persists
    on the document, not inside Streamlit's rerun-able container.
    Call ONCE near the top of `app.py`.
    """
    st.markdown(_STYLESHEET, unsafe_allow_html=True)



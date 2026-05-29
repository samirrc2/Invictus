"""
Invictus — LangGraph Orchestrator
Central workflow that coordinates all portfolio intelligence agents.

Graph topology (7 stages, 15 nodes):
    load_portfolio
         |
    ┌────┴─────────────────────────────┐
    │  Stage 1: Portfolio Intelligence │  (fan-out, parallel)
    ├──────────────────────────────────┤
    │  compute_risk                    │
    │  run_pca                         │
    │  detect_vol_regime               │
    │  run_stress_tests                │
    │  compute_greeks                  │
    │  attribute_pnl                   │
    └────┬─────────────────────────────┘
         |  (fan-in barrier)
    ┌────┴─────────────────────────────┐
    │  Stage 2: Conviction Intelligence│  (fan-out, parallel)
    ├──────────────────────────────────┤
    │  analyze_flows                   │
    │  retrieve_10k_context            │
    │  run_filing_intel                │
    │  run_earnings_intel              │
    └────┬─────────────────────────────┘
         |  (fan-in barrier)
    run_accumulation_model
         |
    run_conviction_synthesis
         |
    generate_commentary
         |
    evaluate_commentary

Built on LangGraph StateGraph with fan-out/fan-in edges.
"""
from typing import Any, Dict, List, Optional, Callable, Annotated
from functools import wraps
import operator
import traceback

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from invictus.agents.graph_state import PortfolioState


# ── Node Registry ──────────────────────────────────────────────────────
# Each node is a function: (state: PortfolioState) -> PortfolioState
# Nodes are registered here and wired into the graph below.

_node_registry: Dict[str, Callable] = {}


def _gather_node_context(state: PortfolioState, name: str) -> dict:
    """Capture lightweight input context for error diagnostics."""
    ctx = {}
    try:
        if state.weights:
            ctx["tickers"] = list(state.weights.keys())
        if state.returns is not None and hasattr(state.returns, "shape"):
            ctx["returns_shape"] = list(state.returns.shape)
            ctx["returns_date_range"] = [
                str(state.returns.index.min()),
                str(state.returns.index.max()),
            ]
        if state.prices is not None and hasattr(state.prices, "shape"):
            ctx["prices_shape"] = list(state.prices.shape)
    except Exception:
        pass  # context gathering must never fail the node
    return ctx


def register_node(name: str):
    """Decorator to register a function as a graph node.

    Captures full traceback + input context on failure so the Dev
    Console can show exactly which node failed, why, and what data
    it was operating on.
    """
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(state: PortfolioState) -> PortfolioState:
            state.current_node = name
            try:
                state = fn(state)
                state.mark_complete(name)
            except Exception as e:
                tb = traceback.format_exc()
                ctx = _gather_node_context(state, name)
                # Structured error with full traceback
                state.add_node_error(name, e, tb, ctx)
                # Legacy string (backward compat with existing consumers)
                state.add_error(name, f"{type(e).__name__}: {str(e)}")
                state.mark_complete(name)  # continue graph even on error
            return state
        _node_registry[name] = wrapper
        return wrapper
    return decorator


def get_node(name: str) -> Callable:
    """Retrieve a registered node by name."""
    if name not in _node_registry:
        raise KeyError(f"Node '{name}' not registered. Available: {list(_node_registry.keys())}")
    return _node_registry[name]


# ── Placeholder Nodes ──────────────────────────────────────────────────
# Each will be replaced with real implementations in their respective steps.

@register_node("load_portfolio")
def load_portfolio_node(state: PortfolioState) -> PortfolioState:
    """Load portfolio data — typically pre-loaded from Streamlit."""
    # In practice, the Streamlit app populates state before the graph runs.
    # This node validates that portfolio data exists.
    if state.holdings is None:
        raise ValueError("No portfolio data loaded. Load holdings first.")
    if state.prices is None:
        raise ValueError("No price data loaded. Fetch prices first.")
    return state


@register_node("compute_risk")
def compute_risk_node(state: PortfolioState) -> PortfolioState:
    """Compute portfolio risk metrics — implemented in Step 4."""
    from invictus.agents.risk_agent import compute_risk
    return compute_risk(state)


@register_node("run_pca")
def run_pca_node(state: PortfolioState) -> PortfolioState:
    """Run PCA factor decomposition — implemented in Step 5."""
    from invictus.agents.pca_agent import run_pca
    return run_pca(state)


@register_node("detect_vol_regime")
def detect_vol_regime_node(state: PortfolioState) -> PortfolioState:
    """Detect volatility regime — implemented in Step 6."""
    from invictus.agents.vol_regime_agent import detect_vol_regime
    return detect_vol_regime(state)


@register_node("run_stress_tests")
def run_stress_tests_node(state: PortfolioState) -> PortfolioState:
    """Run stress tests — implemented in Step 7."""
    from invictus.agents.stress_agent import run_stress_tests
    return run_stress_tests(state)


@register_node("compute_greeks")
def compute_greeks_node(state: PortfolioState) -> PortfolioState:
    """Compute portfolio Greeks — implemented in Step 8."""
    from invictus.agents.greeks_agent import compute_greeks
    return compute_greeks(state)


@register_node("analyze_flows")
def analyze_flows_node(state: PortfolioState) -> PortfolioState:
    """Analyze institutional flows — implemented in Step 9."""
    from invictus.agents.flow_agent import analyze_flows
    return analyze_flows(state)


@register_node("run_accumulation_model")
def run_accumulation_model_node(state: PortfolioState) -> PortfolioState:
    """Run ML accumulation classifier — implemented in Step 10."""
    from invictus.agents.ml_agent import run_accumulation_model
    return run_accumulation_model(state)


@register_node("run_filing_intel")
def run_filing_intel_node(state: PortfolioState) -> PortfolioState:
    """Run Filing & Guidance Intelligence — implemented in Step 15."""
    from invictus.agents.filing_agent import run_filing_intel
    return run_filing_intel(state)


@register_node("run_earnings_intel")
def run_earnings_intel_node(state: PortfolioState) -> PortfolioState:
    """Run Earnings & Management Intelligence — implemented in Step 15."""
    from invictus.agents.earnings_agent import run_earnings_intel
    return run_earnings_intel(state)


@register_node("retrieve_10k_context")
def retrieve_10k_context_node(state: PortfolioState) -> PortfolioState:
    """Retrieve 10-K context via RAG — implemented in Step 11."""
    from invictus.rag.rag_agent import retrieve_10k_context
    return retrieve_10k_context(state)


@register_node("attribute_pnl")
def attribute_pnl_node(state: PortfolioState) -> PortfolioState:
    """Attribute P&L to factors — implemented in Step 12."""
    from invictus.agents.pnl_agent import attribute_pnl
    return attribute_pnl(state)


@register_node("generate_commentary")
def generate_commentary_node(state: PortfolioState) -> PortfolioState:
    """Generate AI commentary — implemented in Step 13."""
    from invictus.agents.commentary_agent import generate_commentary
    return generate_commentary(state)


@register_node("evaluate_commentary")
def evaluate_commentary_node(state: PortfolioState) -> PortfolioState:
    """Evaluate commentary quality — implemented in Step 14."""
    from invictus.evaluation.eval_agent import evaluate_commentary
    return evaluate_commentary(state)


@register_node("run_conviction_synthesis")
def run_conviction_synthesis_node(state: PortfolioState) -> PortfolioState:
    """Synthesize all intelligence into conviction scores — implemented in Step 15."""
    from invictus.agents.synthesis_agent import run_conviction_synthesis
    return run_conviction_synthesis(state)


# ── LangGraph State Schema ─────────────────────────────────────────────
# TypedDict state for LangGraph. List fields use operator.add reducer
# so fan-out branches accumulate entries instead of overwriting.

class PipelineState(TypedDict, total=False):
    """State dict flowing through the LangGraph StateGraph."""
    # Portfolio data
    holdings: Any
    prices: Any
    returns: Any
    weights: Optional[Dict[str, float]]
    total_value: Optional[float]
    total_daily_pnl: Optional[float]
    daily_return_pct: Optional[float]
    total_cost: Optional[float]
    total_unrealized_pnl: Optional[float]
    unrealized_pnl_pct: Optional[float]
    summary: Any
    # Agent outputs
    risk_metrics: Optional[Dict[str, Any]]
    ticker_risk: Any
    correlation_matrix: Any
    pca_results: Optional[Dict[str, Any]]
    vol_regime: Optional[Dict[str, Any]]
    stress_results: Optional[Dict[str, Any]]
    greeks_results: Optional[Dict[str, Any]]
    flow_signals: Optional[Dict[str, Any]]
    ml_predictions: Optional[Dict[str, Any]]
    rag_insights: Optional[Dict[str, Any]]
    pnl_attribution: Optional[Dict[str, Any]]
    commentary: Optional[Dict[str, Any]]
    eval_results: Optional[Dict[str, Any]]
    selected_horizon: str
    filing_intel: Optional[Dict[str, Any]]
    earnings_intel: Optional[Dict[str, Any]]
    conviction_synthesis: Optional[Dict[str, Any]]
    # Orchestration tracking — use add reducer for fan-out merge
    errors: Annotated[List[str], operator.add]
    node_errors: Annotated[List[Dict[str, Any]], operator.add]
    completed_nodes: Annotated[List[str], operator.add]
    current_node: Optional[str]


def _wrap_node(node_fn: Callable) -> Callable:
    """Wrap a registered node function for LangGraph state dict protocol.

    LangGraph nodes receive and return dicts. This wrapper hydrates a
    PortfolioState, runs the agent, and returns a partial state update.
    List fields (errors, completed_nodes) return only NEW entries so the
    operator.add reducer concatenates correctly during fan-out merges.
    """
    def wrapper(state: dict) -> dict:
        # Hydrate PortfolioState from state dict
        fields = {}
        for k in PortfolioState.model_fields:
            if k in state and state[k] is not None:
                fields[k] = state[k]
        pstate = PortfolioState(**fields)

        prev_completed_len = len(pstate.completed_nodes)
        prev_errors_len = len(pstate.errors)
        prev_node_errors_len = len(pstate.node_errors)

        # Run the actual agent (with error handling from register_node)
        pstate = node_fn(pstate)

        # Build return dict — all scalar fields + list deltas
        result: Dict[str, Any] = {}
        for k in PortfolioState.model_fields:
            if k == "completed_nodes":
                result[k] = pstate.completed_nodes[prev_completed_len:]
            elif k == "errors":
                result[k] = pstate.errors[prev_errors_len:]
            elif k == "node_errors":
                result[k] = pstate.node_errors[prev_node_errors_len:]
            else:
                result[k] = getattr(pstate, k)
        return result
    return wrapper


# ── Graph Topology ─────────────────────────────────────────────────────
# Defines the DAG structure: fan-out parallel stages + sequential tail.

# Stage groupings for topology queries
STAGES = [
    ["load_portfolio"],
    ["compute_risk", "run_pca", "detect_vol_regime", "run_stress_tests", "compute_greeks", "attribute_pnl"],
    ["analyze_flows", "retrieve_10k_context", "run_filing_intel", "run_earnings_intel"],
    ["run_accumulation_model"],
    ["run_conviction_synthesis"],
    ["generate_commentary"],
    ["evaluate_commentary"],
]

PI_NODES = STAGES[1]   # Portfolio Intelligence — parallel fan-out
CI_NODES = STAGES[2]   # Conviction Intelligence — parallel fan-out
SEQ_TAIL = ["run_accumulation_model", "run_conviction_synthesis",
            "generate_commentary", "evaluate_commentary"]


def _build_graph() -> StateGraph:
    """Build the LangGraph StateGraph with fan-out/fan-in edges."""
    graph = StateGraph(PipelineState)

    # ── Add all agent nodes ──
    for name, fn in _node_registry.items():
        graph.add_node(name, _wrap_node(fn))

    # ── Barrier (passthrough) nodes for fan-in synchronization ──
    def _passthrough(state: dict) -> dict:
        return {}

    graph.add_node("pi_barrier", _passthrough)
    graph.add_node("ci_barrier", _passthrough)

    # ── Wire edges ──
    # Entry → load
    graph.add_edge(START, "load_portfolio")

    # Load → fan-out to Portfolio Intelligence (parallel)
    for node in PI_NODES:
        graph.add_edge("load_portfolio", node)

    # PI fan-in → barrier
    for node in PI_NODES:
        graph.add_edge(node, "pi_barrier")

    # Barrier → fan-out to Conviction Intelligence (parallel)
    for node in CI_NODES:
        graph.add_edge("pi_barrier", node)

    # CI fan-in → barrier
    for node in CI_NODES:
        graph.add_edge(node, "ci_barrier")

    # Sequential tail: ML → Synthesis → Commentary → Eval → END
    graph.add_edge("ci_barrier", "run_accumulation_model")
    graph.add_edge("run_accumulation_model", "run_conviction_synthesis")
    graph.add_edge("run_conviction_synthesis", "generate_commentary")
    graph.add_edge("generate_commentary", "evaluate_commentary")
    graph.add_edge("evaluate_commentary", END)

    return graph


# ── Graph Executor ─────────────────────────────────────────────────────

class InvictusGraph:
    """
    LangGraph-backed executor with a .run() convenience API.

    Compiles the StateGraph and exposes .run(state, progress_callback)
    matching the interface app.py expects. Also provides topology
    introspection for the Dev Console architecture tab.
    """

    def __init__(self):
        self._graph = _build_graph()
        self._compiled = self._graph.compile()
        self.stages = STAGES

    def run(
        self,
        state: PortfolioState,
        skip_nodes: list = None,
        only_nodes: list = None,
        progress_callback: Callable = None,
    ) -> PortfolioState:
        """Execute the LangGraph pipeline.

        Invokes the compiled StateGraph. Progress callbacks fire by
        streaming node start events from the graph execution.
        """
        skip_nodes = skip_nodes or []

        # Convert PortfolioState → dict for LangGraph
        init_state: Dict[str, Any] = {}
        for k in PortfolioState.model_fields:
            init_state[k] = getattr(state, k)
        # Ensure list fields are initialized for the add reducer
        init_state.setdefault("errors", [])
        init_state.setdefault("node_errors", [])
        init_state.setdefault("completed_nodes", [])

        if progress_callback or skip_nodes or only_nodes:
            # Use step-by-step execution for progress reporting / filtering
            return self._run_stepwise(state, skip_nodes, only_nodes, progress_callback)

        # Fast path: invoke the full compiled graph
        result = self._compiled.invoke(init_state)

        # Merge result back into PortfolioState
        for k in PortfolioState.model_fields:
            if k in result and result[k] is not None:
                setattr(state, k, result[k])
        return state

    def _run_stepwise(
        self,
        state: PortfolioState,
        skip_nodes: list,
        only_nodes: list,
        progress_callback: Callable,
    ) -> PortfolioState:
        """Sequential stage-by-stage execution with progress callbacks.

        Falls back to stage-ordered execution when we need progress
        reporting or node filtering (skip/only).
        """
        total_stages = len(self.stages)

        for stage_idx, stage_nodes in enumerate(self.stages):
            for node_name in stage_nodes:
                if node_name in skip_nodes:
                    continue
                if only_nodes and node_name not in only_nodes:
                    continue

                if progress_callback:
                    progress_callback(node_name, stage_idx, total_stages)

                node_fn = get_node(node_name)
                state = node_fn(state)

        return state

    def run_single_node(self, state: PortfolioState, node_name: str) -> PortfolioState:
        """Run a single node independently."""
        node_fn = get_node(node_name)
        return node_fn(state)

    def get_topology(self) -> Dict[str, Any]:
        """Return the graph topology for visualization."""
        return {
            "stages": [
                {"stage": i, "nodes": nodes, "parallel": len(nodes) > 1}
                for i, nodes in enumerate(self.stages)
            ],
            "total_nodes": sum(len(s) for s in self.stages),
        }

    def get_graph(self) -> StateGraph:
        """Return the underlying StateGraph for inspection."""
        return self._graph


# ── Convenience ────────────────────────────────────────────────────────

def create_graph() -> InvictusGraph:
    """Create the LangGraph-backed pipeline executor.

    Returns an InvictusGraph instance wrapping a compiled LangGraph
    StateGraph with fan-out/fan-in edges for parallel stages.
    """
    return InvictusGraph()

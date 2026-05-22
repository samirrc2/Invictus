"""
Invictus — LangGraph Orchestrator
Central workflow that coordinates all portfolio intelligence agents.

Graph topology:
    load_portfolio
         |
    ┌────┴─────────────────────┐
    │  parallel risk branch    │
    ├──────────────────────────┤
    │  compute_risk            │
    │  run_pca                 │
    │  detect_vol_regime       │
    │  run_stress_tests        │
    │  compute_greeks          │
    └────┬─────────────────────┘
         |
    ┌────┴─────────────────────┐
    │  parallel intel branch   │
    ├──────────────────────────┤
    │  analyze_flows           │
    │  run_accumulation_model  │
    │  retrieve_10k_context    │
    └────┬─────────────────────┘
         |
    attribute_pnl
         |
    generate_commentary
         |
    evaluate_commentary
         |
    produce_final_report
"""
from typing import Any, Dict, Callable
from functools import wraps
import traceback

from invictus.agents.graph_state import PortfolioState


# ── Node Registry ──────────────────────────────────────────────────────
# Each node is a function: (state: PortfolioState) -> PortfolioState
# Nodes are registered here and wired into the graph below.

_node_registry: Dict[str, Callable] = {}


def register_node(name: str):
    """Decorator to register a function as a graph node."""
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(state: PortfolioState) -> PortfolioState:
            state.current_node = name
            try:
                state = fn(state)
                state.mark_complete(name)
            except Exception as e:
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


@register_node("produce_final_report")
def produce_final_report_node(state: PortfolioState) -> PortfolioState:
    """Assemble the final report — implemented in Step 15."""
    # This will be a report assembly step, not a separate agent module
    return state


# ── Graph Builder ──────────────────────────────────────────────────────

class InvictusGraph:
    """
    Simple sequential/parallel graph executor.
    Can be upgraded to full LangGraph StateGraph when langgraph is available.

    This gives us the orchestration pattern now; the LangGraph wiring
    is a drop-in replacement once the agents are built.
    """

    def __init__(self):
        self.stages = [
            # Stage 0: Validate portfolio
            ["load_portfolio"],
            # Stage 1: Risk analytics (parallel)
            ["compute_risk", "run_pca", "detect_vol_regime", "run_stress_tests", "compute_greeks"],
            # Stage 2: Intelligence gathering — flows, filings, earnings (parallel)
            ["analyze_flows", "retrieve_10k_context", "run_filing_intel", "run_earnings_intel"],
            # Stage 3: ML model (depends on flows + filing + earnings for features)
            ["run_accumulation_model"],
            # Stage 4: Conviction Synthesis (depends on all intelligence)
            ["run_conviction_synthesis"],
            # Stage 5: Attribution
            ["attribute_pnl"],
            # Stage 6: Commentary
            ["generate_commentary"],
            # Stage 7: Evaluation
            ["evaluate_commentary"],
            # Stage 8: Final report
            ["produce_final_report"],
        ]

    def run(
        self,
        state: PortfolioState,
        skip_nodes: list = None,
        only_nodes: list = None,
        progress_callback: Callable = None,
    ) -> PortfolioState:
        """
        Execute the graph sequentially (stages run in order).
        Within each stage, nodes run sequentially for simplicity.

        Args:
            state: The portfolio state to process
            skip_nodes: List of node names to skip
            only_nodes: If set, only run these specific nodes
            progress_callback: Optional callback(node_name, stage_idx, total_stages)
        """
        skip_nodes = skip_nodes or []
        total_stages = len(self.stages)

        for stage_idx, stage_nodes in enumerate(self.stages):
            for node_name in stage_nodes:
                # Skip logic
                if node_name in skip_nodes:
                    continue
                if only_nodes and node_name not in only_nodes:
                    continue

                # Progress reporting
                if progress_callback:
                    progress_callback(node_name, stage_idx, total_stages)

                # Execute node
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


# ── LangGraph Native Builder (used when langgraph is installed) ────────

def build_langgraph():
    """
    Build a native LangGraph StateGraph.
    This is the production-grade version using langgraph's graph primitives.
    Falls back to InvictusGraph if langgraph is not installed.
    """
    try:
        from langgraph.graph import StateGraph, END

        # Define the graph with our state schema
        workflow = StateGraph(PortfolioState)

        # Add all nodes
        for name, fn in _node_registry.items():
            workflow.add_node(name, fn)

        # Define edges
        workflow.set_entry_point("load_portfolio")

        # After load → risk branch
        risk_nodes = ["compute_risk", "run_pca", "detect_vol_regime", "run_stress_tests", "compute_greeks"]
        workflow.add_edge("load_portfolio", "compute_risk")
        workflow.add_edge("load_portfolio", "run_pca")
        workflow.add_edge("load_portfolio", "detect_vol_regime")
        workflow.add_edge("load_portfolio", "run_stress_tests")
        workflow.add_edge("load_portfolio", "compute_greeks")

        # After risk → intel branch
        intel_nodes = ["analyze_flows", "run_accumulation_model", "retrieve_10k_context"]
        for risk_node in risk_nodes:
            for intel_node in intel_nodes:
                workflow.add_edge(risk_node, intel_node)

        # After intel → attribution
        for intel_node in intel_nodes:
            workflow.add_edge(intel_node, "attribute_pnl")

        # Sequential tail
        workflow.add_edge("attribute_pnl", "generate_commentary")
        workflow.add_edge("generate_commentary", "evaluate_commentary")
        workflow.add_edge("evaluate_commentary", "produce_final_report")
        workflow.add_edge("produce_final_report", END)

        return workflow.compile()

    except ImportError:
        # Fallback to our simple executor
        return InvictusGraph()


# ── Convenience ────────────────────────────────────────────────────────

def create_graph() -> InvictusGraph:
    """Create an InvictusGraph instance."""
    return InvictusGraph()

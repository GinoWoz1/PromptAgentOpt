# prompt_optimizer/workflow.py

import logging
from typing import List
import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

# Import the State definition (Updated for simplified workflow)
from .models import OptimizationState, EvaluationTask

# Import the nodes (Updated for simplified workflow)
from .nodes import (
    generate_prompts,
    evaluate_prompt_node, # New combined node
    aggregate_and_score
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Routers (Map/Reduce and Loop Control)
# -----------------------------------------------------------------------

def router_to_evaluation(state: OptimizationState) -> List[Send]:
    """
    MAP Dispatcher: Execute and Evaluate all candidates across the mini-batch in parallel.
    Parallelism Degree: N_CANDIDATES * MINI_BATCH_SIZE
    """
    actor_model = state["actor_model"]
    voter_ensemble = state["voter_ensemble"]
    task_description = state["target_task_description"]
    # The mini-batch was selected during the generate_prompts node
    mini_batch = state["current_mini_batch"]
    candidates = state["current_candidates"].values()
    sends = []

    if not mini_batch:
        logger.warning("Mini-batch is empty. Skipping evaluation dispatch.")
        return []

    if not candidates:
        logger.warning("No candidates generated. Skipping evaluation dispatch.")
        return []

    # N * K dispatch
    for candidate in candidates:
        for input_example in mini_batch:
            # Create the combined task
            task = EvaluationTask(
                candidate=candidate,
                input_example=input_example,
                actor_model=actor_model,
                voter_ensemble=voter_ensemble,
                target_task_description=task_description
            )
            sends.append(
                Send(
                    "evaluate_prompt", # Target the combined node
                    task
                )
            )
    logger.info(f"Dispatching {len(sends)} evaluation tasks (Map).")
    return sends


def iteration_router(state: OptimizationState) -> str:
    """
    LOOP CONTROL: Determines whether to continue iterating or finish, implementing Early Stopping.
    """
    current_iteration = state["current_iteration"]
    max_iterations = state["max_iterations"]

    # 1. Check Max Iterations
    if current_iteration >= max_iterations:
        logger.info(f"Reached max iterations ({max_iterations}). Finishing.")
        return "finish"

    # 2. Early Stopping Logic
    # Check if performance improvement has stalled based on configuration.
    min_iterations = state.get("es_min_iterations", 0)
    patience = state.get("es_patience", 0)
    threshold_percentage = state.get("es_threshold_percentage", 0.0)
    # We use the history of the *best score achieved in each iteration*
    score_history = state.get("iteration_best_score_history", [])

    # Ensure we have enough data and ES is enabled (patience > 0)
    # We need at least 'patience' iterations after a baseline to check for improvement.
    if current_iteration >= min_iterations and patience > 0 and len(score_history) > patience:

        # Sanitize history: replace NaN/Inf with very low numbers for comparison
        sanitized_history = [s if np.isfinite(s) else -np.inf for s in score_history]

        # Determine the baseline score: The best score achieved *before* the patience window started.
        # The patience window includes the last 'patience' iterations.
        history_before_window = sanitized_history[:-patience]

        # Calculate the baseline (historical max)
        if history_before_window:
            baseline_score = max(history_before_window)
        else:
            # Should not happen if len(score_history) > patience, but defensive check.
            baseline_score = -np.inf

        # Determine the best score achieved *within* the patience window
        window_scores = sanitized_history[-patience:]
        best_in_window = max(window_scores) if window_scores else -np.inf

        # Calculate improvement
        improvement = best_in_window - baseline_score

        # Calculate percentage improvement relative to the baseline
        if abs(baseline_score) > 1e-9 and np.isfinite(baseline_score): # Avoid division by zero or inf
            percentage_improvement = (improvement / abs(baseline_score)) * 100
        else:
            # Handle division by zero or infinite baseline
            # If baseline is near 0 or -inf, improvement is significant if best_in_window is positive and there was improvement
            percentage_improvement = float('inf') if best_in_window > 0 and improvement > 0 else 0.0

        logger.info(f"Early Stopping Check (Iter {current_iteration}): Patience={patience}, Threshold={threshold_percentage:.2f}%.")
        logger.info(f"  Baseline (Historical Max)={baseline_score:.4f}, Best in Window={best_in_window:.4f}. Improvement={percentage_improvement:.2f}%.")

        if percentage_improvement < threshold_percentage:
            logger.info(f"Improvement ({percentage_improvement:.2f}%) is below threshold ({threshold_percentage:.2f}%) "
                        f"for {patience} iterations. Stopping early.")
            return "finish"

    # 3. Continue Iterating
    if state.get('best_result'):
        # Log the global best score found so far
        score = state['best_result']['aggregate_score']
        logger.info(f"Iteration {current_iteration} complete. Best global score so far: {score:.4f}. Continuing.")
    else:
         logger.info(f"Iteration {current_iteration} complete. No best result yet. Continuing.")

    return "iterate"

# -----------------------------------------------------------------------
# Graph Construction
# -----------------------------------------------------------------------

def build_optimizer_graph() -> StateGraph:
    """
    Defines the structure of the optimization workflow.
    The architecture uses a simple Generate -> Map(Evaluate) -> Reduce(Aggregate) pattern.
    Returns the uncompiled StateGraph.
    """
    graph = StateGraph(OptimizationState)

    # Add Nodes
    graph.add_node("generate_prompts", generate_prompts)
    graph.add_node("evaluate_prompt", evaluate_prompt_node) # Combined Execution + Voting
    graph.add_node("aggregate_and_score", aggregate_and_score)

    # Define Edges
    graph.add_edge(START, "generate_prompts")

    # Map: Generation -> Parallel Evaluation
    graph.add_conditional_edges("generate_prompts", router_to_evaluation)

    # Reduce: Evaluation -> Aggregation
    # LangGraph implicitly synchronizes and aggregates 'current_votes' before this step.
    graph.add_edge("evaluate_prompt", "aggregate_and_score")

    # The Loop: Aggregation -> Iteration Control (with Early Stopping)
    graph.add_conditional_edges(
        "aggregate_and_score",
        iteration_router,
        {"iterate": "generate_prompts", "finish": END}
    )

    # Return the uncompiled graph definition
    return graph
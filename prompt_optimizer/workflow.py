# prompt_optimizer/workflow.py

import logging
from typing import List
from langgraph.graph import END, START, StateGraph
# Use the standard import for Send
from langgraph.types import Send

# Import the State definition
from .models import OptimizationState, ExecutionTask, VotingTask

# Import the nodes
from .nodes import (
    generate_prompts,
    execute_prompt_node,
    synchronize_executions,  # Ensure this is imported
    vote_on_result_node,
    aggregate_and_score
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Routers (Map/Reduce and Loop Control)
# -----------------------------------------------------------------------

def router_to_execution(state: OptimizationState) -> List[Send]:
    """
    Dispatcher for Map 1: Execute all candidates across the mini-batch.
    Parallelism: N_CANDIDATES * MINI_BATCH_SIZE
    """
    actor_model = state["actor_model"]
    # The mini-batch was selected during the generate_prompts node
    mini_batch = state["current_mini_batch"]
    sends = []

    if not mini_batch:
        # If the batch is empty, we skip execution.
        logger.warning("Mini-batch is empty. Skipping execution dispatch.")
        return []

    # N * K dispatch
    for candidate in state["current_candidates"].values():
        for input_example in mini_batch:
            sends.append(
                Send(
                    "execute_prompt",
                    ExecutionTask(
                        candidate=candidate,
                        input_example=input_example,
                        actor_model=actor_model
                    )
                )
            )
    logger.info(f"Dispatching {len(sends)} execution tasks.")
    return sends

def router_to_voting(state: OptimizationState) -> List[Send]:
    """
    Dispatcher for Map 2: Evaluate execution results only for current candidates.
    """
    sends = []
    task_description = state["target_task_description"]
    all_prompts = state["all_tested_prompts"]

    # Filter to current iteration candidate IDs
    valid_candidate_ids = set(state["current_candidates"].keys())

    filtered_results = []
    seen_pairs = set()
    for result in state["current_execution_results"]:
        cid = result.get("candidate_id")
        iid = result.get("input_example_id")
        if cid in valid_candidate_ids:
            pair = (cid, iid)
            if pair not in seen_pairs:
                filtered_results.append(result)
                seen_pairs.add(pair)

    if len(filtered_results) != len(state["current_execution_results"]):
        logger.info(
            f"Voting filter kept {len(filtered_results)}/{len(state['current_execution_results'])} "
            f"based on current_candidates."
        )

    for result in filtered_results:
        prompt_text = all_prompts.get(result["candidate_id"], {}).get("prompt_text", "[Missing]")
        # Do not rely on current_mini_batch; pass empty or result-provided context
        input_data = result.get("input_example_data", {})

        for voter_id in state["voter_ensemble"]:
            task_data = VotingTask(
                execution_result=result,
                voter_id=voter_id,
                target_task_description=task_description,
                prompt_text=prompt_text,
                input_example_data=input_data
            )
            sends.append(Send("vote_on_result", task_data))

    logger.info(f"Dispatching {len(sends)} voting tasks.")
    return sends

def iteration_router(state: OptimizationState) -> str:
    """Step 6: Loop control."""
    if state["current_iteration"] >= state["max_iterations"]:
        logger.info(f"Reached max iterations ({state['max_iterations']}). Finishing.")
        return "finish"

    if state.get('best_result'):
        score = state['best_result']['aggregate_score']
        logger.info(f"Iteration {state['current_iteration']} complete. Best score so far: {score:.4f}")
    return "iterate"

# -----------------------------------------------------------------------
# Graph Construction
# -----------------------------------------------------------------------

def compile_optimizer_graph():
    """Compiles the optimization workflow into a LangGraph executable."""
    graph = StateGraph(OptimizationState)

    # Add Nodes
    graph.add_node("generate_prompts", generate_prompts)
    graph.add_node("execute_prompt", execute_prompt_node)
    # Ensure the synchronization node is added
    graph.add_node("synchronize_executions", synchronize_executions)
    graph.add_node("vote_on_result", vote_on_result_node)
    graph.add_node("aggregate_and_score", aggregate_and_score)

    # Define Edges
    graph.add_edge(START, "generate_prompts")

    # Map 1: Generation -> Execution (Across Mini-Batch)
    graph.add_conditional_edges("generate_prompts", router_to_execution)

    # --- Synchronization Wiring ---

    # Reduce 1: Execution -> Synchronization
    # All parallel executions must converge here.
    graph.add_edge("execute_prompt", "synchronize_executions")

    # Map 2: Synchronization -> Voting
    # Dispatch the next parallel step from the synchronized state.
    graph.add_conditional_edges("synchronize_executions", router_to_voting)

    # ------------------------------

    # Reduce 2: Voting -> Aggregation
    graph.add_edge("vote_on_result", "aggregate_and_score")

    # The Loop
    graph.add_conditional_edges(
        "aggregate_and_score",
        iteration_router,
        {"iterate": "generate_prompts", "finish": END}
    )

    app = graph.compile()
    return app
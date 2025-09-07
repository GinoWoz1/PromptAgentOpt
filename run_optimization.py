# run_optimization.py

import logging
import uuid
import json
import argparse
import sys
from typing import List, Dict, Any

# NOTE: Pandas/Numpy imports are removed as analysis is now integrated into the workflow (nodes.py).

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),             # console
        logging.FileHandler('run_optimization.log', encoding='utf-8')  # file
    ],
)
logger = logging.getLogger(__name__)


# Import necessary components
try:
    # NEW: Import the callback handler for Langfuse
    from callbacks import get_langfuse_handler
except ImportError:
    logger.warning("Could not import get_langfuse_handler from callbacks.py. Tracing might be disabled if requested.")
    get_langfuse_handler = None

try:
    from prompt_optimizer.workflow import compile_optimizer_graph
    # Updated import to include PerformanceExample for type hinting if desired, though not strictly necessary here.
    from prompt_optimizer.models import OptimizationState, InputExample
    # Ensure nodes are imported so the LLM manager initializes (if required by environment setup)
    from prompt_optimizer.nodes import generate_prompts
    import os
    from datetime import datetime
except ImportError as e:
    print(f"Error importing prompt_optimizer modules: {e}")
    print("Please ensure the project structure is correct and the necessary packages are installed.")
    sys.exit(1)


def load_json_file(filepath: str) -> Any:
    """Loads a JSON file with robust error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {filepath}")
        sys.exit(1)


def format_dataset(raw_dataset: List[Dict[str, Any]]) -> List[InputExample]:
    """Formats the raw dataset into the required InputExample structure."""
    input_examples = []

    if not isinstance(raw_dataset, list):
        logger.error("Error: Dataset JSON must be a list of objects.")
        sys.exit(1)

    for item in raw_dataset:
        if not isinstance(item, dict):
            logger.warning(f"Skipping invalid item in dataset (not a dictionary): {item}")
            continue

        # We need a unique ID for tracking, use the provided 'id' or generate one
        example_id = str(item.get("id", str(uuid.uuid4())[:8]))

        # The 'data' field contains the actual inputs that will be formatted into the prompt template
        # We exclude the 'id' key from the data payload itself, as it's metadata.
        data_payload = {k: v for k, v in item.items() if k != 'id'}

        if not data_payload:
            logger.warning(f"Skipping empty data item with id {example_id}")
            continue

        input_examples.append(InputExample(id=example_id, data=data_payload))

    if not input_examples:
        logger.error("Error: The dataset is empty or contains no valid examples after formatting.")
        sys.exit(1)

    return input_examples


# -----------------------------------------------------------------------
# Post-Hoc Analysis Helpers (REMOVED)
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------

# Updated main function signature to include enable_tracing
def main(config_path: str, dataset_path: str, output_path: str, enable_tracing: bool):
    logger.info(f"Starting optimization process.")
    logger.info(f"Loading configuration from: {config_path}")
    logger.info(f"Loading dataset from: {dataset_path}")

    # 1. Load Configuration and Data
    config = load_json_file(config_path)
    raw_dataset = load_json_file(dataset_path)

    # 2. Format Dataset
    input_dataset = format_dataset(raw_dataset)
    logger.info(f"Successfully formatted {len(input_dataset)} input examples.")

    # Extract parameters from config
    task_description = config.get("task_description")
    opt_params = config.get("optimization_parameters", {})
    model_config = config.get("model_configuration", {})
    # NEW: Extract early stopping parameters (if provided in config under "early_stopping" key)
    es_config = config.get("early_stopping", {})

    if not task_description or not opt_params or not model_config:
        logger.error("Error: Configuration file is missing required sections (task_description, optimization_parameters, or model_configuration).")
        sys.exit(1)

    # 3. Initialize Optimization State
    initial_state = OptimizationState(
        # Core Configuration
        max_iterations=opt_params.get("max_iterations", 10),
        N_CANDIDATES=opt_params.get("N_CANDIDATES", 5),
        MINI_BATCH_SIZE=opt_params.get("MINI_BATCH_SIZE", 2),
        target_task_description=task_description,
        input_dataset=input_dataset,

        # Early Stopping Configuration
        es_min_iterations=es_config.get("min_iterations", 3),
        es_patience=es_config.get("patience", 2),
        es_threshold_percentage=es_config.get("threshold_percentage", 5.0),

        # Model Configuration
        optimizer_model=model_config.get("optimizer_model"),
        actor_model=model_config.get("actor_model"),
        voter_ensemble=model_config.get("voter_ensemble", []),

        # Initial state variables (boilerplate)
        current_iteration=0,
        current_candidates={},
        current_mini_batch=[],
        current_execution_results=[],
        current_votes=[],
        history=[],
        all_tested_prompts={},
        best_result=None,
        synthesized_critiques="",
        iteration_best_score_history=[],
        global_best_example=None,
        global_worst_example=None,
        
        # NEW (Q2): Initialize the detailed trace history
        execution_trace_history=[],
    )

    # Validate models are specified
    if not initial_state['optimizer_model'] or not initial_state['actor_model'] or not initial_state['voter_ensemble']:
        logger.error("Error: Model configuration is incomplete in the config file.")
        sys.exit(1)

    # 4. Compile and Run the Graph
    logger.info("Compiling optimizer graph...")
    app = compile_optimizer_graph()

    # Configure Callbacks for Langfuse
    # Initialize the configuration with required settings
    runnable_config = {"recursion_limit": 150}

    if enable_tracing:
        if get_langfuse_handler:
            try:
                langfuse_handler = get_langfuse_handler()
                # LangGraph (like LangChain) expects callbacks under the "callbacks" key
                runnable_config["callbacks"] = [langfuse_handler]
                logger.info("Langfuse tracing enabled.")
            except Exception as e:
                # Catch potential initialization errors (e.g., missing API keys in .env)
                logger.warning(f"Could not initialize Langfuse handler. Tracing will be disabled. Error: {e}")
        else:
            logger.warning("Tracing requested but Langfuse handler could not be imported.")

    logger.info("Running optimization workflow...")
    try:
        # Using invoke to get the final state. Pass the configuration.
        final_state = app.invoke(initial_state, config=runnable_config)

        logger.info("\n--- Optimization Finished ---")

        # Extract results from final state
        best_result = final_state.get('best_result')
        final_history = final_state.get('history', [])
        global_best_example = final_state.get('global_best_example')
        global_worst_example = final_state.get('global_worst_example')
        # NEW (Q2): Extract the detailed trace history
        execution_trace_history = final_state.get('execution_trace_history', [])

        if best_result:

            # Display Summary (Updated for Q3: Interpretability)
            print(f"\nTask Name: {config.get('task_name', 'N/A')}")
            print(f"Iterations Completed: {final_state.get('current_iteration', 'N/A')}")

            # NEW (Q3): Clarified Metrics
            print(f"\n--- Performance Metrics ---")
            raw_score = best_result.get('raw_average_score', 0.0)
            agg_score = best_result.get('aggregate_score', 0.0)

            print(f"Absolute Quality Score (Avg 1-10): {raw_score:.2f}")
            print(f"  (Average score given by voters. Best indicator of real-world quality.)")
           
            print(f"\nRelative Performance Score (Z-avg): {agg_score:.4f}")
            print(f"  (Normalized score used for optimization. Indicates performance relative to other candidates.)")
            print("-----------------------------\n")

            print("\n--- Best Prompt Template ---")
            print(best_result.get('prompt_text', '[Missing]'))
            print("----------------------------\n")

            # Display Global Best Example Snippet (Updated for Q3)
            if global_best_example:
                print("\n--- Global Best Execution Example (Highest Consensus Score Observed) ---")
                # Show both scores
                print(f"Consensus Raw Score (1-10): {global_best_example.get('consensus_raw_score', 0.0):.2f}")
                print(f"Consensus Normalized Score: {global_best_example.get('consensus_normalized_score', 0.0):.4f}")
                print(f"Candidate ID: {global_best_example.get('candidate_id')}")
                print(f"Executed Prompt Snippet: {global_best_example.get('executed_prompt_text', '')[:500]}...")
                print(f"\nOutput Snippet: {global_best_example.get('output', '')[:500]}...")
                print("---------------------------------------------------------------\n")

            # Save results (Updated structure for Q2)
            results_output = {
                "task_name": config.get('task_name'),
                "configuration": config,
                "best_prompt_result": best_result,
                "summary_traceability": { # Renamed from "traceability" for clarity
                    "overall_best_execution": global_best_example,
                    "overall_worst_execution": global_worst_example,
                    "iteration_best_score_history": final_state.get('iteration_best_score_history', []),
                },
                "prompt_history": final_history, # Renamed from "history"
                # NEW (Q2): Detailed history of every execution and evaluation
                "execution_details": execution_trace_history,
            }
            with open(output_path, "w", encoding='utf-8') as f:
                # ensure_ascii=False is crucial for saving non-English characters correctly
                json.dump(results_output, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
            logger.info(f"Total detailed execution traces captured: {len(execution_trace_history)}")

        else:
            logger.warning("Optimization finished without finding a best result. Check logs for errors during the run.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Generalized Prompt Optimization Workflow.")
    parser.add_argument("--config", type=str, default="optimization_config.json", help="Path to the optimization configuration JSON file.")
    parser.add_argument("--dataset", type=str, default="input_dataset.json", help="Path to the input dataset JSON file.")
    parser.add_argument("--output", type=str, help="Path to save the optimization results JSON file. If not specified, will be generated based on config and timestamp.")
    parser.add_argument("--trace", action="store_true", help="Enable Langfuse tracing for observability.")

    args = parser.parse_args()

    # Generate dynamic output filename if not provided
    if not args.output:

        # Load config to get task name
        config = load_json_file(args.config)
        task_name = config.get('task_name', 'optimization')

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate filename: taskname_timestamp_results.json
        safe_task_name = "".join(c for c in task_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_task_name = safe_task_name.replace(' ', '_')

        # Create optimization_results folder if it doesn't exist
        results_dir = "optimization_results"
        os.makedirs(results_dir, exist_ok=True)

        args.output = os.path.join(results_dir, f"{safe_task_name}_{timestamp}_results.json")

        # Pass the trace argument to main
        main(args.config, args.dataset, args.output, args.trace)
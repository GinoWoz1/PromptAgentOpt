# run_optimization.py

import logging
import uuid
import json
import argparse
import sys
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),                      # console
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
    from prompt_optimizer.models import OptimizationState, InputExample
    # Ensure nodes are imported so the LLM manager initializes (if required by environment setup)
    from prompt_optimizer.nodes import generate_prompts 
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

    if not task_description or not opt_params or not model_config:
        logger.error("Error: Configuration file is missing required sections (task_description, optimization_parameters, or model_configuration).")
        sys.exit(1)

    # 3. Initialize Optimization State
    initial_state = OptimizationState(
        # Core Configuration
        max_iterations=opt_params.get("max_iterations", 3),
        N_CANDIDATES=opt_params.get("N_CANDIDATES", 5),
        MINI_BATCH_SIZE=opt_params.get("MINI_BATCH_SIZE", 2),
        target_task_description=task_description,
        input_dataset=input_dataset,

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
        synthesized_critiques=""
    )

    # Validate models are specified
    if not initial_state['optimizer_model'] or not initial_state['actor_model'] or not initial_state['voter_ensemble']:
        logger.error("Error: Model configuration is incomplete in the config file.")
        sys.exit(1)

    # 4. Compile and Run the Graph
    logger.info("Compiling optimizer graph...")
    app = compile_optimizer_graph()

    # NEW: Configure Callbacks for Langfuse
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
        best_result = final_state.get('best_result')
        if best_result:
            print(f"\nTask Name: {config.get('task_name', 'N/A')}")
            print(f"Best Aggregate Score (Normalized): {best_result['aggregate_score']:.4f}")
            print(f"Best Raw Average Score: {best_result['raw_average_score']:.4f}")
            print("\n--- Best Prompt Template ---")
            print(best_result['prompt_text'])
            print("----------------------------\n")
            
            # Save results
            results_output = {
                "task_name": config.get('task_name'),
                "configuration": config,
                "best_result": best_result,
                "history": final_state['history']
            }
            with open(output_path, "w", encoding='utf-8') as f:
                # ensure_ascii=False is crucial for saving non-English characters correctly
                json.dump(results_output, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")

        else:
            logger.warning("Optimization finished without finding a best result. Check logs for errors during the run.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Generalized Prompt Optimization Workflow.")
    parser.add_argument("--config", type=str, default="optimization_config.json", help="Path to the optimization configuration JSON file.")
    parser.add_argument("--dataset", type=str, default="input_dataset.json", help="Path to the input dataset JSON file.")
    parser.add_argument("--output", type=str, default="optimization_results.json", help="Path to save the optimization results JSON file.")
    # NEW: Add argument for tracing
    parser.add_argument("--trace", action="store_true", help="Enable Langfuse tracing for observability.")

    args = parser.parse_args()
    
    # Pass the trace argument to main
    main(args.config, args.dataset, args.output, args.trace)
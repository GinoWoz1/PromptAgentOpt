# run.py
import logging
import json
import sys
import os
from typing import Dict, Any, Optional

# Conditionally import typer; provide feedback if missing.
try:
    import typer
except ImportError:
    print("Error: Typer is required to run the CLI. Please install it: pip install typer[all]")
    sys.exit(1)

from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# Import necessary components from the application structure
try:
    # Import the registry which holds the compiled workflows
    from prompt_optimizer.registry import workflow_registry
    # Import the initializer to handle state preparation
    from prompt_optimizer.initializer import initialize_optimization_state
    # Import utilities for setup
    from prompt_optimizer.utils import (
        configure_logging, setup_run_directory, load_json_file
    )
    from prompt_optimizer.models import OptimizationState
    # Ensure LLM manager is initialized (it's initialized when nodes import it)
    from prompt_optimizer import nodes
    # Import callback handler
    from callbacks import get_langfuse_handler
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure the project structure is correct and the necessary packages are installed.")
    # Specific check for Langfuse handler if it failed during the main block
    get_langfuse_handler = None

# Initialize logger (will be configured properly once run_dir is known)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Prompt Optimization Workflow Runner")

# -----------------------------------------------------------------------
# Presentation Logic
# -----------------------------------------------------------------------

def display_summary(final_state: OptimizationState, config: Dict[str, Any]):
    """Prints the final summary of the optimization run to the console."""
    best_result = final_state.get('best_result')
    global_best_example = final_state.get('global_best_example')

    if not best_result:
        typer.echo("\nOptimization finished without finding a best result. Check logs for errors.")
        return

    typer.echo(f"\n--- Optimization Finished ---")
    typer.echo(f"Task Name: {config.get('task_name', 'N/A')}")
    typer.echo(f"Iterations Completed: {final_state.get('current_iteration', 'N/A')}")

    # Metrics
    typer.echo(f"\n--- Performance Metrics ---")
    raw_score = best_result.get('raw_average_score', 0.0)
    agg_score = best_result.get('aggregate_score', 0.0)

    typer.echo(f"Absolute Quality Score (Avg 1-10): {raw_score:.2f}")
    typer.echo(f"  (Average score given by voters. Best indicator of real-world quality.)")

    typer.echo(f"\nRelative Performance Score (Z-avg): {agg_score:.4f}")
    typer.echo(f"  (Normalized score used for optimization.)")
    typer.echo("-----------------------------\n")

    typer.echo("\n--- Best Prompt Template ---")
    typer.echo(best_result.get('prompt_text', '[Missing]'))
    typer.echo("----------------------------\n")

    # Global Best Example Snippet
    if global_best_example:
        typer.echo("\n--- Global Best Execution Example (Highest Consensus Score Observed) ---")
        typer.echo(f"Consensus Raw Score (1-10): {global_best_example.get('consensus_raw_score', 0.0):.2f}")
        typer.echo(f"Candidate ID: {global_best_example.get('candidate_id')}")
        typer.echo(f"Output Snippet: {global_best_example.get('output', '')[:500]}...")
        typer.echo("---------------------------------------------------------------\n")

def save_results(results_output_path: str, final_state: OptimizationState, config: Dict[str, Any]):
    """Saves the detailed results to the results.json file."""

    if not final_state.get('best_result'):
        return

    # Prepare the results dictionary structure
    results_output = {
        "task_name": config.get('task_name'),
        "configuration": config,
        "best_prompt_result": final_state.get('best_result'),
        "summary_traceability": {
            "overall_best_execution": final_state.get('global_best_example'),
            "overall_worst_execution": final_state.get('global_worst_example'),
            "iteration_best_score_history": final_state.get('iteration_best_score_history', []),
        },
        "prompt_history": final_state.get('history', []),
        "execution_details": final_state.get('execution_trace_history', []),
    }

    try:
        with open(results_output_path, "w", encoding='utf-8') as f:
            # ensure_ascii=False is crucial for non-English characters
            json.dump(results_output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {results_output_path}")
        logger.info(f"Total detailed execution traces captured: {len(results_output['execution_details'])}")
    except IOError as e:
        logger.error(f"Error saving results to {results_output_path}: {e}")


# -----------------------------------------------------------------------
# Main CLI Command
# -----------------------------------------------------------------------

@app.command()
def optimize(
    workflow_name: str = typer.Argument("prompt_optimizer_v1", help="The name of the workflow to run from the registry."),
    config_path: str = typer.Option("optimization_config.json", "--config", help="Path to the optimization configuration JSON file."),
    dataset_path: str = typer.Option("input_dataset.json", "--dataset", help="Path to the input dataset JSON file."),
    trace: bool = typer.Option(False, "--trace", help="Enable Langfuse tracing for observability."),
    run_name: Optional[str] = typer.Option(None, "--run_name", help="Optional name for the run directory. If not specified, generated from config and timestamp.")
):
    """
    Run the Generalized Prompt Optimization Workflow.
    """
    # 1. Setup Directory and Logging
    try:
        # We need to load the config preliminarily just for the directory setup helper.
        # suppress_logging=True because the logger isn't configured yet.
        config_preliminary = load_json_file(config_path, suppress_logging=True)
        run_dir = setup_run_directory(config_preliminary, run_name)
        configure_logging(run_dir)
    except (FileNotFoundError, ValueError, OSError) as e:
        # Errors during critical setup
        typer.echo(f"Error during setup: {e}", err=True)
        raise typer.Exit(code=1)

    results_output_path = os.path.join(run_dir, "results.json")
    prompt_trace_path = os.path.join(run_dir, "prompt_generation_trace.jsonl")

    logger.info(f"Starting optimization process.")
    logger.info(f"Workflow: {workflow_name} | Config: {config_path} | Dataset: {dataset_path}")
    logger.info(f"Results directory: {run_dir}")

    # 2. Initialize State (Handles data loading and validation)
    try:
        initial_state, config = initialize_optimization_state(config_path, dataset_path, prompt_trace_path)
    except (FileNotFoundError, ValueError, TypeError) as e:
        logger.error(f"Failed to initialize optimization state: {e}")
        typer.echo(f"Error during initialization. See logs for details.", err=True)
        raise typer.Exit(code=1)

    # 3. Retrieve Compiled Workflow from Registry
    try:
        workflow_app = workflow_registry.get(workflow_name)
        logger.info(f"Successfully retrieved compiled workflow '{workflow_name}' from registry.")
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)

    # 4. Configure Runnable (Callbacks and Limits)
    # Set recursion limit dynamically based on max iterations.
    runnable_config = {"recursion_limit": initial_state['max_iterations'] * 5 + 50}

    if trace:
        if get_langfuse_handler:
            try:
                langfuse_handler = get_langfuse_handler()
                runnable_config["callbacks"] = [langfuse_handler]
                logger.info("Langfuse tracing enabled.")
            except Exception as e:
                logger.warning(f"Could not initialize Langfuse handler. Tracing disabled. Error: {e}")
        else:
             logger.info("Tracing requested, but Langfuse handler could not be initialized (check dependencies/config).")

    # 5. Execute Workflow
    logger.info("Running optimization workflow...")
    try:
        # Invoke the compiled graph
        final_state = workflow_app.invoke(initial_state, config=runnable_config)

        # 6. Display Summary and Save Results
        display_summary(final_state, config)
        save_results(results_output_path, final_state, config)

    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}", exc_info=True)
        typer.echo(f"Error during execution. See {os.path.join(run_dir, 'execution.log')} for details.", err=True)
        sys.exit(1)

if __name__ == "__main__":
    # Entry point for the Typer application
    app()
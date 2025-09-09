# prompt_optimizer/initializer.py

"""
This module handles the initialization of the optimization process, including
loading configuration, formatting the dataset, and preparing the initial state
for the LangGraph workflow.
"""

import logging
from typing import Dict, Any, Tuple

# Import the core models and utilities using relative imports
from .models import OptimizationState
from .utils import load_json_file, format_dataset

# Import the configuration validator (absolute import as config.py is at root)
# We rely on the manual validation implemented in config.py now.
try:
    from config import validate_optimization_config
except ImportError as e:
    # This should only happen if config.py itself is missing, which is a critical failure.
    logging.getLogger().error(f"CRITICAL: Could not import configuration validator (config.py missing?): {e}")
    # Raise ImportError so the application stops execution if the config module is missing.
    raise ImportError(f"Failed to import necessary configuration module: {e}")


logger = logging.getLogger(__name__)

def initialize_optimization_state(config_path: str, dataset_path: str, prompt_trace_path: str) -> Tuple[OptimizationState, Dict[str, Any]]:
    """
    Loads configuration and data, formats the dataset, validates configuration, and creates the initial OptimizationState.

    Returns:
        A tuple containing the initial OptimizationState and the loaded configuration dictionary.
    Raises:
        FileNotFoundError, ValueError, TypeError: If loading or validation fails.
    """
    logger.info(f"Initializing state. Config: {config_path} | Dataset: {dataset_path}")

    # 1. Load Configuration and Data
    # Exceptions (FileNotFoundError, ValueError) are propagated to the caller.
    raw_config = load_json_file(config_path)
    raw_dataset = load_json_file(dataset_path)

    # 1b. Validate Configuration Structure and Apply Defaults
    try:
        # This will raise ValueError if validation fails (handled by the manual validator).
        config = validate_optimization_config(raw_config)
    except ValueError as e:
        # The error is already logged in the validator.
        # We re-raise it for consistency in error handling in the caller (run.py).
        # The message from the validator is already formatted correctly.
        raise e

    # The fallback logic (manual checks and defaults) previously present here is now removed,
    # as it is fully handled within validate_optimization_config.

    # 2. Format Dataset
    # Exceptions (ValueError, TypeError) are propagated to the caller.
    input_dataset = format_dataset(raw_dataset)

    logger.info(f"Successfully formatted {len(input_dataset)} input examples.")

    # 3. Extract parameters from config
    # We rely on the structure and defaults established during validation (step 1b).
    # We can safely access keys as validation ensures they exist.
    task_description = config["task_description"]
    opt_params = config["optimization_parameters"]
    model_config = config["model_configuration"]
    # Early stopping is guaranteed to exist due to the default handling in the validator
    es_config = config["early_stopping"]

    # 4. Initialize Optimization State
    initial_state = OptimizationState(
        # Core Configuration
        max_iterations=opt_params["max_iterations"],
        N_CANDIDATES=opt_params["N_CANDIDATES"],
        MINI_BATCH_SIZE=opt_params["MINI_BATCH_SIZE"],
        target_task_description=task_description,
        input_dataset=input_dataset,

        # Early Stopping Configuration
        es_min_iterations=es_config["min_iterations"],
        es_patience=es_config["patience"],
        es_threshold_percentage=es_config["threshold_percentage"],

        # Model Configuration
        optimizer_model=model_config["optimizer_model"],
        actor_model=model_config["actor_model"],
        voter_ensemble=model_config["voter_ensemble"],

        # Initial state variables (boilerplate)
        current_iteration=0,
        current_candidates={},
        current_mini_batch=[],
        current_votes=[],
        history=[],
        all_tested_prompts={},
        best_result=None,
        synthesized_critiques="",
        iteration_best_score_history=[],
        global_best_example=None,
        global_worst_example=None,
        execution_trace_history=[],
        prompt_trace_file_path=prompt_trace_path,
    )

    # Final safeguard check (redundant as validation covers this, but safe practice)
    if not initial_state['optimizer_model'] or not initial_state['actor_model'] or not initial_state['voter_ensemble']:
        msg = "Error: Model configuration is incomplete after initialization (Safeguard Check)."
        logger.error(msg)
        raise ValueError(msg)

    # Return both the state and the validated/defaulted config
    return initial_state, config
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

logger = logging.getLogger(__name__)

def initialize_optimization_state(config_path: str, dataset_path: str, prompt_trace_path: str) -> Tuple[OptimizationState, Dict[str, Any]]:
    """
    Loads configuration and data, formats the dataset, and creates the initial OptimizationState.

    Returns:
        A tuple containing the initial OptimizationState and the loaded configuration dictionary.
    Raises:
        FileNotFoundError, ValueError, TypeError: If loading or validation fails.
    """
    logger.info(f"Initializing state. Config: {config_path} | Dataset: {dataset_path}")

    # 1. Load Configuration and Data
    # Exceptions (FileNotFoundError, ValueError) are propagated to the caller.
    config = load_json_file(config_path)
    raw_dataset = load_json_file(dataset_path)

    # 2. Format Dataset
    # Exceptions (ValueError, TypeError) are propagated to the caller.
    input_dataset = format_dataset(raw_dataset)

    logger.info(f"Successfully formatted {len(input_dataset)} input examples.")

    # 3. Extract parameters from config
    task_description = config.get("task_description")
    opt_params = config.get("optimization_parameters", {})
    model_config = config.get("model_configuration", {})
    es_config = config.get("early_stopping", {})

    if not task_description or not opt_params or not model_config:
        msg = "Error: Configuration file is missing required sections (task_description, optimization_parameters, or model_configuration)."
        logger.error(msg)
        raise ValueError(msg)

    # 4. Initialize Optimization State
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

    # Validate models are specified
    if not initial_state['optimizer_model'] or not initial_state['actor_model'] or not initial_state['voter_ensemble']:
        msg = "Error: Model configuration is incomplete (optimizer_model, actor_model, or voter_ensemble missing)."
        logger.error(msg)
        raise ValueError(msg)

    # Return both the state and the config (needed for saving results later)
    return initial_state, config
# ./config.py
# centralizes all configuration management and validation (Manual Implementation)

import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Manual Validation Helpers
# -----------------------------------------------------------------------

def _validate_required(config: Dict[str, Any], key: str, parent_key: str = None):
    """Helper to validate required keys (presence and non-emptiness)."""
    full_key_name = f"{parent_key}.{key}" if parent_key else key

    # Determine the target dictionary and ensure parent structure is valid
    if parent_key:
        if parent_key not in config or not isinstance(config.get(parent_key), dict):
             # If the parent section is missing or not a dict, the required child key cannot exist.
             raise ValueError(f"Missing or invalid parent configuration section: '{parent_key}' required by '{full_key_name}'")
        target_dict = config[parent_key]
    else:
        target_dict = config

    # Check for presence
    if key not in target_dict:
        raise ValueError(f"Missing required configuration: '{full_key_name}'")

    value = target_dict[key]

    # Check if the value is empty (None, empty string, empty list, empty dict)
    if value is None or (isinstance(value, (str, list, dict)) and not value):
         raise ValueError(f"Required configuration '{full_key_name}' cannot be empty or None.")

    return value

def _apply_default(config: Dict[str, Any], key: str, default: Any, expected_type: type, constraint_check: Callable = None, parent_key: str = None):
    """Helper to apply defaults, check types, attempt type conversion, and validate constraints."""
    full_key_name = f"{parent_key}.{key}" if parent_key else key

    # Determine the target dictionary and ensure parent structure exists
    if parent_key:
        # Ensure parent dictionary exists and is a dictionary
        if parent_key not in config or config.get(parent_key) is None:
             config[parent_key] = {}
        if not isinstance(config[parent_key], dict):
             raise ValueError(f"Parent configuration section '{parent_key}' must be a dictionary.")
        target_dict = config[parent_key]
    else:
        target_dict = config

    # Apply default if missing or None
    if key not in target_dict or target_dict.get(key) is None:
        target_dict[key] = default
        return default

    value = target_dict[key]

    # Type check and conversion
    if not isinstance(value, expected_type):
        # Attempt conversion for simple types (e.g., if config file provided numbers as strings, or int for float)
        try:
            if expected_type == float and isinstance(value, (int, float, str)):
                 converted_value = float(value)
            elif expected_type == int and isinstance(value, (int, str)):
                 # Ensure string doesn't represent a float before converting to int
                 if isinstance(value, str) and '.' in value: raise TypeError()
                 converted_value = int(value)
            elif expected_type == bool and isinstance(value, (bool, str)):
                 if isinstance(value, str):
                      if value.lower() == 'true': converted_value = True
                      elif value.lower() == 'false': converted_value = False
                      else: raise TypeError()
                 else:
                      converted_value = value
            else:
                 raise TypeError()

            target_dict[key] = converted_value
            value = converted_value

        except (ValueError, TypeError):
            # If conversion fails or type is fundamentally incompatible
            raise ValueError(f"Invalid type for '{full_key_name}'. Expected {expected_type.__name__}, got {type(value).__name__} (Value: {value}).")

    # Constraint check
    if constraint_check and not constraint_check(value):
        raise ValueError(f"Configuration '{full_key_name}' fails constraint check. Value: {value}")

    return value

# -----------------------------------------------------------------------
# Main Validation Function
# -----------------------------------------------------------------------

def validate_optimization_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manually validates the structure and content of an optimization configuration dictionary
    and applies default values.

    Args:
        config_data: The dictionary loaded from the optimization JSON file.

    Returns:
        The validated configuration data (with defaults applied).

    Raises:
        ValueError: If the configuration is invalid (missing keys, invalid types, or failed constraints).
    """
    logger.info("Validating optimization configuration structure (Manual)...")

    # Work on a copy to ensure the original data is not mutated if validation fails midway
    config = config_data.copy()

    try:
        # 1. Core Requirements
        # Apply default for task_name first
        _apply_default(config, "task_name", "unnamed_optimization", str)

        # Validate critical required fields
        _validate_required(config, "task_description")
        # We don't strictly validate the presence of the parent sections here,
        # as the nested validation helpers handle it gracefully.

        # 2. Optimization Parameters Details
        opt_params = "optimization_parameters"
        # Constraints: gt=0 (Greater than 0)
        _apply_default(config, "max_iterations", 10, int, lambda x: x > 0, parent_key=opt_params)
        _apply_default(config, "N_CANDIDATES", 5, int, lambda x: x > 0, parent_key=opt_params)
        _apply_default(config, "MINI_BATCH_SIZE", 2, int, lambda x: x > 0, parent_key=opt_params)

        # 3. Model Configuration Details (Required section, no defaults for models)
        model_config = "model_configuration"
        _validate_required(config, "optimizer_model", parent_key=model_config)
        _validate_required(config, "actor_model", parent_key=model_config)
        voter_ensemble = _validate_required(config, "voter_ensemble", parent_key=model_config)

        # Specific type and constraint check for voter_ensemble (must be list, len_min=1)
        if not isinstance(voter_ensemble, list) or len(voter_ensemble) < 1:
             raise ValueError("model_configuration.voter_ensemble must be a non-empty list.")
        if not all(isinstance(m, str) and m for m in voter_ensemble):
             raise ValueError("model_configuration.voter_ensemble must contain only non-empty strings.")

        # 4. Early Stopping (Optional section)
        es_config = "early_stopping"
        # Constraints: gte=0 (Greater than or equal to 0)
        _apply_default(config, "min_iterations", 3, int, lambda x: x >= 0, parent_key=es_config)
        _apply_default(config, "patience", 2, int, lambda x: x >= 0, parent_key=es_config)
        _apply_default(config, "threshold_percentage", 5.0, float, lambda x: x >= 0.0, parent_key=es_config)

        logger.info("Optimization configuration validated successfully.")
        return config

    except ValueError as e:
        # Catch validation errors, log them, and re-raise with a standardized prefix
        logger.error(f"Configuration validation failed: {e}")
        # Re-raise as ValueError for consistency with the caller's expectations
        raise ValueError(f"Invalid optimization configuration structure: {e}")
# prompt_optimizer/utils.py
import logging
import json
import os
import uuid
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

# Relative import for models within the package
from .models import InputExample

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------

def configure_logging(run_dir: str):
    """Configures centralized logging for the optimization run."""

    # Ensure the directory exists
    os.makedirs(run_dir, exist_ok=True)

    # Define the standard log format
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # 1. Console Handler
    # Use RichHandler if available for better formatting, otherwise standard StreamHandler
    try:
        # Optional dependency: pip install rich
        from rich.logging import RichHandler
        console_handler = RichHandler(rich_tracebacks=True, show_path=False)
        # RichHandler uses its own format string; we just need the message.
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    except ImportError:
        # Fallback to standard console output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)

    console_handler.setLevel(logging.INFO)


    # 2. Execution Log File Handler
    execution_log_path = os.path.join(run_dir, 'execution.log')
    # Use mode='w' for a fresh log
    execution_file_handler = logging.FileHandler(execution_log_path, encoding='utf-8', mode='w')
    execution_file_handler.setLevel(logging.INFO)
    execution_file_handler.setFormatter(log_format)

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs if run in sequence
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(execution_file_handler)

    # We use the specific module logger (__name__) to log initialization confirmation.
    logging.getLogger(__name__).info(f"--- Logging Initialized ---")
    logging.getLogger(__name__).info(f"Execution log: {execution_log_path}")

# -----------------------------------------------------------------------
# File and Data Handling
# -----------------------------------------------------------------------

def load_json_file(filepath: str, suppress_logging: bool = False) -> Any:
    """Loads a JSON file with robust error handling. Raises exceptions on failure."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        msg = f"Error: File not found at {filepath}"
        if not suppress_logging:
            logger.error(msg)
        raise FileNotFoundError(msg)
    except json.JSONDecodeError:
        msg = f"Error: Could not decode JSON from {filepath}"
        if not suppress_logging:
            logger.error(msg)
        raise ValueError(msg) # Raise ValueError for bad JSON format
    except Exception as e:
        msg = f"An unexpected error occurred loading JSON from {filepath}: {e}"
        if not suppress_logging:
            logger.error(msg)
        raise

def format_dataset(raw_dataset: List[Dict[str, Any]]) -> List[InputExample]:
    """Formats the raw dataset. Raises exceptions on critical failures."""
    input_examples = []

    if not isinstance(raw_dataset, list):
        msg = "Error: Dataset JSON must be a list of objects."
        logger.error(msg)
        raise TypeError(msg)

    for item in raw_dataset:
        if not isinstance(item, dict):
            logger.warning(f"Skipping invalid item in dataset (not a dictionary): {item}")
            continue

        example_id = str(item.get("id", str(uuid.uuid4())[:8]))
        data_payload = {k: v for k, v in item.items() if k != 'id'}

        if not data_payload:
            logger.warning(f"Skipping empty data item with id {example_id}")
            continue

        input_examples.append(InputExample(id=example_id, data=data_payload))

    if not input_examples:
        msg = "Error: The dataset is empty or contains no valid examples after formatting."
        logger.error(msg)
        raise ValueError(msg)

    return input_examples

# -----------------------------------------------------------------------
# Directory Management
# -----------------------------------------------------------------------

def setup_run_directory(config: Dict[str, Any], run_name: Optional[str]) -> str:
    """Creates a unique directory for the run results."""
    results_base_dir = "optimization_results"
    os.makedirs(results_base_dir, exist_ok=True)

    if run_name:
        run_dir_name = run_name
    else:
        # Determine name from the provided config dictionary
        task_name = config.get('task_name', 'optimization')

        # Create timestamp and sanitize names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_task_name = "".join(c for c in task_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_task_name = safe_task_name.replace(' ', '_')
        run_dir_name = f"{safe_task_name}_{timestamp}"

    run_dir_path = os.path.join(results_base_dir, run_dir_name)

    try:
        os.makedirs(run_dir_path, exist_ok=False)
    except FileExistsError:
        # Handle collisions
        if not run_name:
            # Auto-generated name collision, add suffix
            new_run_dir_path = os.path.join(results_base_dir, f"{run_dir_name}_{str(uuid.uuid4())[:4]}")
            print(f"Info: Run directory {run_dir_path} exists. Creating new directory: {new_run_dir_path}")
            os.makedirs(new_run_dir_path)
            run_dir_path = new_run_dir_path
        else:
            # User specified name collision, raise error
            raise FileExistsError(f"Run directory {run_dir_path} exists. Please specify a unique --run_name.")
    except OSError as e:
        raise OSError(f"Could not create results directory {run_dir_path}. {e}")

    return run_dir_path
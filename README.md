# Generalized Prompt Optimization Framework

## Overview

This repository contains an advanced, iterative prompt optimization framework built using [LangGraph](https://langchain-ai.github.io/langgraph/). The goal of this framework is to automatically discover the best prompt template for a specific task by leveraging Large Language Models (LLMs) to both generate candidate prompts and evaluate their performance.

The system operates in a loop using distinct roles for LLMs:

1.  **Generate (The Optimizer):** An LLM generates diverse prompt candidates based on the task description and historical performance data.
2.  **Execute (The Actor):** An LLM executes these candidate prompts against a mini-batch of input data.
3.  **Evaluate (The Voters):** An ensemble of LLMs evaluates the outputs (LLM-as-a-Judge).
4.  **Aggregate & Synthesize:** Scores are normalized (Z-score) to find consensus, and critiques are synthesized into actionable insights for the next iteration.

## Key Features

  * **LLM Agnostic Architecture:** Easily switch between different model providers, including OpenAI, Google (Gemini), Anthropic, and local models via Ollama.
  * **Iterative Refinement:** The system learns over time, using synthesized feedback to improve prompt strategies.
  * **Ensemble Voting and Normalization:** Automated evaluation using an ensemble of Voter LLMs, utilizing normalized scoring (Z-scores) for robust comparison and to mitigate bias from individual voters.
  * **Elitism:** Ensures the best-performing prompt from the previous iteration is carried forward, preventing performance regression.
  * **Novelty Checking:** Uses embeddings to ensure newly generated prompts are sufficiently different from previously tested ones, encouraging exploration of the prompt space.
  * **Dynamic Interpolation:** Supports robust Dollar-style (`$placeholder`) template rendering for generalized input data.
  * **Detailed Traceability:** Generates comprehensive logs (`execution.log`), result summaries (`results.json`), and detailed traces of the optimizer's decisions and rejections (`prompt_generation_trace.jsonl`).
  * **Observability:** Optional integration with [Langfuse](https://langfuse.com/) for detailed visualization of the workflow execution.

## Prerequisites

  * Python 3.10+
  * API keys for the LLM providers you intend to use.
  * (Optional) A running [Ollama](https://ollama.com/) instance if using local models via Ollama.

## Installation

1.  **Install dependencies using uv**:

    ```bash
    uv sync
    ```

    This will automatically create a virtual environment and install all required dependencies.

2.  **Activate the virtual environment**:

    ```bash
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

    Alternatively, you can run commands directly with uv without activating:

    ```bash
    uv run python run_optimization.py --config config.json --dataset data.json
    ```
3.  **(Optional) Install additional LLM providers** if needed:

    ```bash
    uv add openai anthropic google-generativeai
    ```

4.  **(Optional) Install packages for specific features**:

    ```bash
    # For Langfuse observability
    uv add langfuse

    ```

## Configuration

### 1\. Environment Variables (.env)

The system uses a `.env` file to manage API keys and endpoints. Create a `.env` file in the root directory:

```dotenv
# --- Required for Cloud LLMs ---
# Required if using OpenAI models. Also used by default for embeddings (novelty checks).
OPENAI_API_KEY=sk-xxxxxxxxxxxx

# Required if using Google Gemini models
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxx

# Required if using Anthropic Claude models
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx

# --- Optional: For Local Models (Ollama) ---
# Defaults to http://localhost:11434/v1 if not set.
# Note: The URL must end in /v1 for compatibility with the OpenAI SDK.
# OLLAMA_BASE_URL=http://localhost:11434/v1

# --- Optional: For Langfuse Tracing (if using --trace) ---
# LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxx
# LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxx
```

### 2\. Input Data Format

The input dataset must be a JSON file containing a list of objects. Each object represents a single input example.

The keys in the dataset objects will be dynamically inserted into the prompt templates during execution.

**Example: `dataset_codegen.json`**

Let's optimize a prompt for generating Python code based on a description and signature. This will be used an input as part of a task where the result of the LLM will be judged by an actor model.

```json
[
    {
        "id": "ex1_reverse_string",
        "description": "A function that reverses a string using slicing.",
        "signature": "def reverse_string(s: str) -> str:"
    },
    {
        "id": "ex2_factorial",
        "description": "A function that calculates the factorial of a non-negative integer iteratively.",
        "signature": "def calculate_factorial(n: int) -> int:"
    },
    {
        "id": "ex3_fibonacci",
        "description": "A function that returns the nth Fibonacci number.",
        "signature": "def fibonacci(n: int) -> int:"
    }
]
```

### 3\. Optimization Configuration (JSON)

This file defines the optimization task, the models to use, and the parameters for the run.

**Example: `config_codegen.json`**

```json
{
    "task_name": "Python Function Generator",
    "task_description": "Generate a complete Python function based on a description and a function signature. The output should be clean, executable Python code. The inputs will be provided via $description and $signature placeholders.",
    "optimization_parameters": {
        "max_iterations": 5,
        "N_CANDIDATES": 8,
        "MINI_BATCH_SIZE": 10
    },
    "model_configuration": {
        "optimizer_model": "gpt-4o",
        "actor_model": "ollama/codellama:7b",
        "voter_ensemble": [
            "gpt-4o-mini",
            "gemini-1.5-flash"
        ]
    },
    "early_stopping": {
        "min_iterations": 3,
        "patience": 2,
        "threshold_percentage": 5.0
    }
}
```

#### Configuration Fields

  * `task_name`: A friendly name for the task. Used for organizing output directories.
  * `task_description`: The ground truth instruction for the task. This is what the Voters use to evaluate the Actor's output. **Crucially, mention the input placeholders (e.g., `$description`) here.**
  * `optimization_parameters`:
      * `max_iterations`: How many cycles the optimization will run.
      * `N_CANDIDATES`: How many new prompts to generate per iteration.
      * `MINI_BATCH_SIZE`: How many data examples to test each prompt against in a single iteration (sampling from the dataset).
  * `model_configuration`: Defines the LLMs for each role.
  * `early_stopping` (Optional): Parameters to stop the optimization if the score improvement stalls (`patience` defines how many iterations to wait for improvement).

#### Model Naming Conventions

  * **OpenAI, Google, Anthropic:** Use their standard model names (e.g., `gpt-4o`, `gemini-1.5-flash`, `claude-3-haiku-20240307`).
  * **Ollama:** Prefix the model name with `ollama/` (e.g., `ollama/llama3:8b`). Ensure Ollama is running and the model is pulled.

## Usage

To run the optimization workflow, use the `run_optimization.py` script.

```bash
python run_optimization.py --config <path_to_config.json> --dataset <path_to_dataset.json> [options]
```

### Arguments

  * `--config`: Path to the optimization configuration JSON file. (Default: `optimization_config.json`)
  * `--dataset`: Path to the input dataset JSON file. (Default: `input_dataset.json`)
  * `--run_name`: Optional name for the output directory. If not specified, it is generated from the task name and timestamp.
  * `--trace`: Enable Langfuse tracing for observability. Requires Langfuse keys in the `.env` file.

### Examples

**Basic Run:**

```bash
python run_optimization.py --config config_codegen.json --dataset dataset_codegen.json
```

**Run with Tracing and a Custom Name:**

```bash
python run_optimization.py --config config_codegen.json --dataset dataset_codegen.json --run_name codegen_v1_codellama --trace
```

## Interpreting the Output

The script automatically organizes the results into a dedicated directory within `optimization_results/`.

```
optimization_results/
└── Python_Function_Generator_20250908_042752/
    ├── execution.log                 # Detailed log of the run.
    ├── prompt_generation_trace.jsonl # Step-by-step trace of the Optimizer LLM calls.
    └── results.json                  # The final output file.
```

### Key Output Files

1.  **`results.json`**: The primary output file.

      * `best_prompt_result`: The highest-scoring prompt found during the optimization.
          * `prompt_text`: **The optimized prompt template.**
          * `raw_average_score`: The average score (1-10) given by the voters. (Best indicator of real-world quality).
          * `aggregate_score`: The normalized score (Z-avg). (Used internally for optimization decisions).
      * `configuration`: A copy of the configuration used for the run.
      * `summary_traceability`: Includes the single best and worst individual executions observed globally during the run.
      * `prompt_history`: A history of all prompts tested and their scores, sorted by performance.
      * `execution_details`: A detailed list of every single execution (prompt + input example) including the output generated and the individual votes/critiques received.

2.  **`execution.log`**: A detailed log file of the entire optimization process.

3.  **`prompt_generation_trace.jsonl`**: A JSON Lines file providing deep insight into the Optimizer LLM's behavior. It tracks every attempt to generate prompts, including the inputs provided to the LLM, the generated output, and reasons why specific prompts were rejected (e.g., failing the novelty check).
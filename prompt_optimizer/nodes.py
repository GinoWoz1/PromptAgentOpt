# prompt_optimizer/nodes.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
import logging
import json
import re
import os
from string import Template
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity

# Import models and templates (using relative imports)
from .models import (
    PromptCandidate, ExecutionResult, Vote, IterationResult, ExecutionTask,
    VotingTask, OptimizationState, PerformanceExample, ExecutionTrace,
    PromptGenerationTrace # Data models for tracing generation and execution
)
from .templates import (
    format_optimizer_prompt, format_synthesis_prompt,
    OPTIMIZER_SYSTEM_PROMPT, format_voting_prompt, VOTING_SYSTEM_PROMPT
)

# Import the LLM Manager
# Assumes intelligence_assets.llm.client is available in the environment
try:
    from intelligence_assets.llm.client import llm_manager
except ImportError:
    # Set up basic logging if the dependency is missing
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.warning("LLM Manager import failed. LLM calls will not work. Using mock manager for compatibility.")
    
    # Define a mock llm_manager if the dependency is missing to allow the module to load
    class MockLLMManager:
        def call_model(self, *args, **kwargs):
            return "LLM_CALL_ERROR: LLM Manager not available."
        def get_embeddings(self, texts):
            # Return dummy embeddings
            return [[0.0]*10 for _ in texts]
    llm_manager = MockLLMManager()
else:
    # If the import succeeds, set up the logger normally
    logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------

# Attempt to import dirtyjson for robust JSON parsing
try:
    import dirtyjson
except ImportError:
    dirtyjson = None
    # Ensure the logger is available before attempting to log the warning
    if 'logger' in locals() or 'logger' in globals():
        logger.warning("dirtyjson not installed. JSON parsing will be less robust. Install with 'uv pip install dirtyjson'.")


def call_llm(model_name: str, prompt: str, system_prompt: Optional[str] = None, response_format="text", **kwargs) -> str:
    """
    Centralized handler for calling the specified LLM model via the LLMClientManager.

    Args:
        model_name: The identifier of the LLM model to use.
        prompt: The user prompt input.
        system_prompt: Optional system prompt to guide the model's behavior.
        response_format: The desired output format (e.g., "text", "json").
        **kwargs: Additional keyword arguments passed to the LLM manager (e.g., temperature).

    Returns:
        The model's response as a string, or an error string prefixed with "LLM_CALL_ERROR:".
    """
    try:
        return llm_manager.call_model(model_name=model_name, prompt=prompt, system_prompt=system_prompt, response_format=response_format, **kwargs)
    except Exception as e:
        logger.error(f"Error calling LLM {model_name}: {e}")
        return f"LLM_CALL_ERROR: {e}"

def parse_json_output(output: str) -> Any:
    """
    Parses the LLM output which is expected to be a JSON string.

    Implements robust extraction by finding the outermost JSON structure (object or array),
    handling cases where the LLM includes surrounding text (e.g., markdown). It also uses
    lenient parsing (via dirtyjson if available) to handle common formatting errors
    like trailing commas or incorrect escaping.

    Args:
        output: The raw string output from the LLM.

    Returns:
        The parsed Python object (typically a dict or list).

    Raises:
        ValueError: If the output is empty, contains an LLM error, has no JSON structure,
                    is severely truncated, or fails parsing even with lenient methods.
    """
    output = output.strip()
    if output.startswith("LLM_CALL_ERROR"):
        raise ValueError(output)

    # 1. Robust Extraction: Find the start and end of the main JSON object/array.
    start_match = re.search(r'[\{\[]', output)
    if not start_match:
        if not output:
            raise ValueError("LLM output was empty.")
        raise ValueError(f"No JSON structure found in LLM output. Snippet: {output[:200]}")

    start_index = start_match.start()
    
    # Determine the expected closing character.
    if output[start_index] == '{':
        end_index = output.rfind('}')
    else: # output[start_index] == '['
        end_index = output.rfind(']')

    if end_index == -1 or end_index < start_index:
        # Handle cases where the closing marker is missing (severe truncation)
        raise ValueError(f"Incomplete JSON structure (truncation likely). Snippet: {output[start_index:start_index+200]}...")

    json_str = output[start_index:end_index+1]

    # 2. Parsing: Try standard json.loads first, fallback to dirtyjson if installed.
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # If standard parsing fails, attempt recovery if dirtyjson is available.
        if dirtyjson:
            try:
                # Use dirtyjson to handle internal errors (escaping, trailing commas, etc.)
                logger.info("Standard JSON parsing failed. Retrying with dirtyjson.")
                return dirtyjson.loads(json_str)
            except Exception as de:
                # If both standard and dirtyjson fail, log the errors and raise an exception.
                logger.error(f"JSON parsing failed (standard and dirtyjson). Errors: {e} | {de}. Attempted to parse: {json_str[:500]}...")
                raise ValueError(f"Failed to parse JSON output. Snippet: {json_str[:200]}")
        else:
            # Fail if standard parsing failed and dirtyjson is not available.
            logger.error(f"JSON parsing error: {e}. Attempted to parse: {json_str[:500]}...")
            raise ValueError(f"Failed to parse JSON output. Snippet: {json_str[:200]}")

def write_prompt_trace(trace: PromptGenerationTrace, file_path: Optional[str]):
    """
    Appends a prompt generation trace record to the specified JSONL file.

    This function ensures that detailed logs of the prompt generation process,
    including inputs, outputs, acceptance/rejection status, and errors, are saved
    for later analysis. It handles file operations safely and logs errors without
    interrupting the main optimization workflow.

    Args:
        trace: A dictionary representing the PromptGenerationTrace record.
        file_path: The path to the JSONL file. If None, the write operation is skipped.
    """
    if not file_path:
        return
    try:
        # Open in append mode ('a') to add new lines to the log file.
        with open(file_path, 'a', encoding='utf-8') as f:
            # ensure_ascii=False is crucial for saving non-English characters (e.g., Chinese text) correctly.
            json_record = json.dumps(trace, ensure_ascii=False)
            f.write(json_record + '\n')
    except Exception as e:
        # Log the error but do not stop the optimization process if logging fails.
        logger.error(f"CRITICAL: Failed to write prompt generation trace to {file_path}: {e}")

# -----------------------------------------------------------------------
# Node Definitions
# -----------------------------------------------------------------------

# Synchronization Barrier (Reduce 1)
def synchronize_executions(state: OptimizationState) -> Dict[str, Any]:
    """
    Acts as a synchronization barrier (Reduce step) after parallel executions (Map 1).

    This node waits for all execution tasks in the current iteration to complete
    before proceeding to the voting phase. It primarily serves as a logging and
    control flow mechanism.

    Args:
        state: The current optimization state, containing 'current_execution_results'.

    Returns:
        An empty dictionary, as this node ensures synchronization rather than state modification.
    """
    num_results = len(state['current_execution_results'])
    logger.info(f"--- Synchronization Point (Reduce 1) ---")
    logger.info(f"Accumulated {num_results} execution results. Proceeding to voting.")
    return {}

def generate_prompts(state: OptimizationState) -> Dict[str, Any]:
    """
    Generates a new batch of prompt candidates for the current iteration (Generate Step).

    This function handles several key responsibilities in the optimization loop:
    1. Determines the inputs for the optimizer LLM (synthesized critiques, performance history).
    2. Implements Elitism by carrying forward the best-performing prompt from previous iterations.
    3. Calls the optimizer LLM to generate novel prompt suggestions.
    4. Ensures Novelty by checking the semantic similarity of new prompts against all previously tested prompts.
    5. Selects a mini-batch of input data for evaluation.
    6. Maintains detailed tracing of the generation process, including attempts and failures, for diagnostics.

    Args:
        state: The current optimization state.

    Returns:
        A dictionary updating the state with the new iteration number, candidates,
        updated list of all tested prompts, the current mini-batch, and resetting
        results/votes for the new iteration.

    Raises:
        RuntimeError: If the optimizer fails to generate any novel prompts after maximum retries and the fallback mechanism also fails.
        ValueError: If the input dataset is empty.
    """
    iteration = state['current_iteration'] + 1
    N_CANDIDATES = state['N_CANDIDATES']
    OPTIMIZER_MODEL = state['optimizer_model']
    # Similarity threshold for novelty check. Prompts above this threshold are rejected.
    # A relaxed threshold (0.90) encourages focused optimization while still enforcing some diversity.
    SIMILARITY_THRESHOLD = 0.90
    MAX_RETRIES = 3
    # Path for saving detailed generation traces, if enabled.
    TRACE_FILE_PATH = state.get('prompt_trace_file_path')

    logger.info(f"--- Starting Iteration {iteration} --- Model: {OPTIMIZER_MODEL} ---")

    # 1. Prepare input data and initialize structures
    # Performance history provides structured guidance to the optimizer.
    performance_history = state['history']
    synthesized_critiques = state['synthesized_critiques']
    # We work on a copy of all_tested_prompts, updating it as we accept new prompts.
    all_tested_prompts = state['all_tested_prompts'].copy()

    new_candidates = {}
    # Initialize list to track elite prompts specifically for the trace log.
    elite_tracking = []
    
    # --- Elitism Implementation ---
    # Ensure the best prompt is automatically included in the next iteration (if space allows).
    if state['best_result'] and N_CANDIDATES > 1:
        elite_id = state['best_result']['candidate_id']
        elite_candidate = all_tested_prompts.get(elite_id)
        
        if elite_candidate:
            logger.info(f"Elitism: Carrying forward Elite prompt (ID: {elite_id})")
            # Ensure the dictionary is copied so we don't modify the historical record inadvertently.
            elite_copy = elite_candidate.copy() 
            # Mark it as elite for this iteration's tracking.
            elite_copy['is_elite'] = True 
            new_candidates[elite_id] = elite_copy
            # Track elite prompts for the trace log.
            elite_tracking.append({
                "candidate_id": elite_id,
                "prompt_text": elite_copy['prompt_text'],
                "is_elite": True,
                "similarity_score": 1.0 # Elite prompts are definitionally identical to themselves.
            })
        else:
            logger.warning(f"Elite candidate ID {elite_id} not found in all_tested_prompts.")

    # Prepare a summary of the performance history for the trace log (Top 10).
    MAX_HISTORY_FOR_TRACE = 10
    history_summary = [
        {"candidate_id": h['candidate_id'], "aggregate_score": h['aggregate_score'], "iteration": h['iteration']}
        for h in performance_history[:MAX_HISTORY_FOR_TRACE]
    ]

    # 2. Generate and Validate Prompts (The Novelty Loop)
    attempts = 0
    generated_texts = []

    # Continue generating until the required number of candidates is met or max retries are exhausted.
    while len(new_candidates) < N_CANDIDATES and attempts < MAX_RETRIES:
        attempts += 1

        # Initialize trace dictionary for this specific generation attempt.
        trace = {
            "iteration": iteration,
            "attempt": attempts,
            "timestamp": datetime.now().isoformat(),
            "optimizer_model": OPTIMIZER_MODEL,
            "input_synthesized_critiques": synthesized_critiques if synthesized_critiques else "[No critiques available]",
            "input_performance_history_summary": history_summary,
            "num_requested": 0,
            "num_generated": 0,
            "num_accepted": 0,
            "num_rejected_similarity": 0,
            "rejected_prompts": [],
            "accepted_prompts": [],
            "llm_call_successful": False,
            "error_message": None
        }

        # Calculate how many NEW prompts we need (accounts for elites already added).
        needed = N_CANDIDATES - len(new_candidates)
        trace['num_requested'] = needed

        if needed <= 0:
            break

        logger.info(f"Attempt {attempts}/{MAX_RETRIES} to generate {needed} prompts")

        # Format the prompt for the optimizer LLM, including task description, critiques, history, and count.
        optimizer_prompt = format_optimizer_prompt(
            state['target_task_description'],
            synthesized_critiques,
            performance_history, # Passing structured history
            needed,
            iteration
        )

        # Call the optimizer LLM.
        raw_output = call_llm(
            model_name=OPTIMIZER_MODEL,
            prompt=optimizer_prompt,
            system_prompt=OPTIMIZER_SYSTEM_PROMPT,
            response_format="json",
            temperature=0.7 # Moderate temperature for creative but relevant generation.
        )

        if raw_output.startswith("LLM_CALL_ERROR"):
            logger.error(f"LLM call failed: {raw_output}")
            # Record failure in trace and write immediately.
            trace['error_message'] = raw_output
            write_prompt_trace(trace, TRACE_FILE_PATH)
            continue

        # Mark LLM call success in the trace.
        trace['llm_call_successful'] = True

        # Parse the LLM output.
        try:
            parsed_output = parse_json_output(raw_output)

            # Validate the structure of the parsed output.
            if isinstance(parsed_output, dict) and "prompts" in parsed_output and isinstance(parsed_output["prompts"], list):
                generated_texts = parsed_output["prompts"]
                logger.info(f"Successfully parsed {len(generated_texts)} prompts")
                trace['num_generated'] = len(generated_texts)
            else:
                logger.warning(f"Unexpected output format. Output: {raw_output[:200]}")
                generated_texts = []
                # Record error and write trace if the format is incorrect.
                trace['error_message'] = "Unexpected JSON format (missing 'prompts' key or not a list)."
                write_prompt_trace(trace, TRACE_FILE_PATH)
                continue
        except ValueError as e:
            logger.warning(f"JSON parsing failed: {str(e)}")
            generated_texts = []
            # Record error and write trace if parsing fails.
            trace['error_message'] = f"JSON parsing failed: {str(e)}"
            write_prompt_trace(trace, TRACE_FILE_PATH)
            continue

        if not generated_texts:
            # Write trace if the list was empty but parsing succeeded (e.g., LLM returned []).
            if trace['llm_call_successful'] and trace['num_generated'] == 0:
                trace['error_message'] = "LLM successfully generated an empty list of prompts."
                write_prompt_trace(trace, TRACE_FILE_PATH)
            continue

        # Generate embeddings for the new prompts to check for novelty.
        try:
            generated_embeddings = llm_manager.get_embeddings(generated_texts)
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            # Record error and write trace if embedding generation fails.
            trace['error_message'] = f"Embedding generation failed: {str(e)}"
            write_prompt_trace(trace, TRACE_FILE_PATH)
            continue

        # Comparison set includes embeddings of all historical prompts.
        comparison_embeddings = [p['embedding'] for p in all_tested_prompts.values() if p.get('embedding')]

        rejected_count = 0
        accepted_count = 0

        # Initialize lists for detailed tracking of accepted/rejected prompts within this attempt.
        attempt_accepted = []
        attempt_rejected = []

        # Iterate through generated prompts and apply the novelty check.
        for text, embedding in zip(generated_texts, generated_embeddings):
            if len(new_candidates) >= N_CANDIDATES:
                break

            is_novel = True
            # Initialize max_similarity for tracing purposes.
            max_similarity = 0.0

            # Check similarity only if there are historical prompts to compare against.
            if comparison_embeddings:
                try:
                    # Calculate cosine similarity between the new prompt and all existing prompts.
                    embedding_np = np.array(embedding).reshape(1, -1)
                    comparison_embeddings_np = np.array(comparison_embeddings)
                    similarities = cosine_similarity(embedding_np, comparison_embeddings_np)[0]
                    max_similarity = np.max(similarities)

                    # Reject if the maximum similarity exceeds the threshold.
                    if max_similarity > SIMILARITY_THRESHOLD:
                        rejected_count += 1
                        is_novel = False
                except Exception as e:
                    logger.error(f"Similarity check failed: {str(e)}")
                    is_novel = True # Proceed if similarity check fails as a fallback.

            if is_novel:
                # Accept the novel prompt.
                candidate_id = str(uuid.uuid4())[:8]
                candidate = {
                    "candidate_id": candidate_id,
                    "prompt_text": text,
                    "iteration": iteration,
                    "embedding": embedding,
                    "is_elite": False
                }
                new_candidates[candidate_id] = candidate
                # Crucial: Add to all_tested_prompts immediately so subsequent prompts in this batch are compared against it.
                all_tested_prompts[candidate_id] = candidate
                comparison_embeddings.append(embedding)
                accepted_count += 1
                # Track acceptance details for the trace log.
                attempt_accepted.append({
                    "candidate_id": candidate_id,
                    "prompt_text": text,
                    "is_elite": False,
                    # Cast numpy float to native Python float for JSON serialization compatibility.
                    "similarity_score": float(max_similarity)
                })
            else:
                # Track rejection details for the trace log.
                attempt_rejected.append({
                    "prompt_text": text,
                    "reason": f"Similarity threshold ({SIMILARITY_THRESHOLD}) exceeded",
                    "similarity_score": float(max_similarity)
                })

        logger.info(f"After attempt {attempts}: {accepted_count} accepted, {rejected_count} rejected")

        # Finalize the trace for this attempt.
        # We track the count of *newly generated* accepted prompts in this attempt.
        trace['num_accepted'] = accepted_count
        trace['num_rejected_similarity'] = rejected_count
        trace['accepted_prompts'] = attempt_accepted
        trace['rejected_prompts'] = attempt_rejected

        # If this is the first attempt, include the elite prompts in the trace for completeness of the iteration's batch.
        if attempts == 1 and elite_tracking:
            # We combine elites with newly accepted prompts.
            trace['accepted_prompts'] = elite_tracking + trace['accepted_prompts']
            # Update the total accepted count shown in this trace entry to include elites.
            trace['num_accepted'] += len(elite_tracking)


        # Write the completed trace record for the attempt.
        write_prompt_trace(trace, TRACE_FILE_PATH)

    # 3. Fallback Mechanism (if generation fails completely)
    if not new_candidates:
        logger.error("CRITICAL ERROR - No novel prompts were generated after all attempts. Using fallback.")

        # Initialize trace for the fallback mechanism activation.
        fallback_trace = {
            "iteration": iteration,
            "attempt": "FALLBACK", # Indicate this is the fallback mechanism
            "timestamp": datetime.now().isoformat(),
            "optimizer_model": OPTIMIZER_MODEL,
            "input_synthesized_critiques": "N/A (Fallback mechanism activated due to failure in standard generation)",
            "input_performance_history_summary": history_summary,
            "num_requested": 1,
            "num_generated": 0,
            "num_accepted": 0,
            "num_rejected_similarity": 0,
            "rejected_prompts": [],
            "accepted_prompts": [],
            "llm_call_successful": False,
            "error_message": "Failed to generate novel prompts after max retries. Initiating fallback."
        }

        try:
            # Attempt a simple, direct generation as a last resort.
            # CRITICAL: Ensure placeholders use $ syntax for consistency with the execution engine (string.Template).
            raw_output = call_llm(
                model_name=OPTIMIZER_MODEL,
                # Example fallback prompt tailored to a potential use case.
                prompt="Generate one simple prompt for analyzing Chinese Buddhist text translation. Include placeholders for $chinese_context, $target_info, and $english_translation_block.",
                system_prompt="You are a helpful assistant. Generate a prompt template. Use $ for placeholders.",
                response_format="text",
                temperature=0.9 # High temperature for maximum chance of generation.
            )

            if not raw_output.startswith("LLM_CALL_ERROR"):
                # Update trace if the fallback LLM call succeeded.
                fallback_trace['llm_call_successful'] = True
                fallback_trace['num_generated'] = 1
                fallback_trace['error_message'] = None # Clear the initiation message.

                candidate_id = str(uuid.uuid4())[:8]
                try:
                    # Attempt to embed the fallback prompt.
                    fallback_embedding = llm_manager.get_embeddings([raw_output.strip()])[0]
                except:
                    # Use a dummy embedding if embedding fails.
                    fallback_embedding = [0.0] * 10 
                candidate = {
                    "candidate_id": candidate_id,
                    "prompt_text": raw_output.strip(),
                    "iteration": iteration,
                    "embedding": fallback_embedding,
                    "is_elite": False
                }
                new_candidates[candidate_id] = candidate
                all_tested_prompts[candidate_id] = candidate

                # Update trace with acceptance details of the fallback prompt.
                fallback_trace['num_accepted'] = 1
                fallback_trace['accepted_prompts'].append({
                    "candidate_id": candidate_id,
                    "prompt_text": candidate['prompt_text'],
                    "is_elite": False,
                    "similarity_score": 0.0 # Fallback bypasses the similarity check.
                })
                write_prompt_trace(fallback_trace, TRACE_FILE_PATH)

            else:
                # Record failure and write trace before raising a fatal error.
                fallback_trace['error_message'] = f"Fallback LLM call failed: {raw_output}"
                write_prompt_trace(fallback_trace, TRACE_FILE_PATH)
                raise RuntimeError(f"Optimizer LLM failed completely: {raw_output}")
        except Exception as e:
            # Ensure the trace is written if an unexpected error occurs during fallback handling.
            # Check if the trace hasn't already captured a specific failure reason before overwriting the error message.
            if fallback_trace['error_message'] == "Failed to generate novel prompts after max retries. Initiating fallback.":
                 fallback_trace['error_message'] = f"Unexpected error during fallback execution: {str(e)}"
                 write_prompt_trace(fallback_trace, TRACE_FILE_PATH)
            # Raise the underlying exception after tracing.
            raise RuntimeError(f"Optimizer LLM failed to generate novel prompts after multiple attempts and fallback failed: {str(e)}")

    # 4. Mini-Batch Selection
    dataset = state['input_dataset']
    # Determine the batch size, capped by the available dataset size.
    batch_size = min(state['MINI_BATCH_SIZE'], len(dataset))

    if batch_size > 0:
        # Randomly sample indices for the mini-batch without replacement.
        indices = np.random.choice(len(dataset), batch_size, replace=False)
        current_mini_batch = [dataset[i] for i in indices]
    else:
        current_mini_batch = []
        if len(dataset) == 0:
            logger.error("Input dataset is empty")
            raise ValueError("Input dataset cannot be empty")

    # 5. Return Updated State
    return {
        "current_iteration": iteration,
        "current_candidates": new_candidates,
        # Pass the updated dictionary containing the new prompts.
        "all_tested_prompts": all_tested_prompts, 
        "current_mini_batch": current_mini_batch,
        # Reset results and votes for the new iteration.
        "current_execution_results": [],
        "current_votes": [],
    }


def execute_prompt_node(task: ExecutionTask) -> Dict[str, List[ExecutionResult]]:
    """
    Executes a prompt candidate (template) on a specific input example (Map 1 - Execute).

    This node handles:
    1. Template Rendering: Dynamically interpolates input data into the prompt template using string.Template ($ placeholders) for robustness.
    2. Execution: Calls the Actor LLM to generate an output based on the rendered prompt.
    3. Result Packaging: Structures the execution details (input, rendered prompt, output) into an ExecutionResult.

    Args:
        task: An ExecutionTask containing the candidate, input example, and actor model ID.

    Returns:
        A dictionary containing a list with the single ExecutionResult, ready for accumulation in the state.
    """
    candidate = task['candidate']
    input_example = task['input_example']
    ACTOR_MODEL = task['actor_model']

    input_data = input_example.get('data', {})
    input_example_id = input_example.get('id', 'N/A')

    logger.debug(f"Executing prompt {candidate['candidate_id']} on input {input_example_id} with {ACTOR_MODEL}")

    # Initialize with a default failure message in case rendering fails.
    prompt_to_execute = "[Template Rendering Failed]"
    output = ""

    # -----------------------------------------------------------------------
    # DYNAMIC INTERPOLATION AND EXECUTION
    # -----------------------------------------------------------------------
    try:
        # Use string.Template for interpolation. It expects $placeholders.
        template = Template(candidate['prompt_text'])
        # safe_substitute prevents errors if a placeholder is missing in the input data and ignores standard {} formatting.
        prompt_to_execute = template.safe_substitute(input_data)
    except Exception as e:
        # Catches potential errors during template rendering, such as invalid $ syntax (e.g., ValueError for a bare $).
        error_msg = f"Template rendering failed (string.Template). Check for invalid syntax (e.g., bare $): {e}."
        logger.error(f"Prompt format error for {candidate['candidate_id']}: {error_msg}")
        output = f"LLM_CALL_ERROR: {error_msg}"
    else:
        # If formatting succeeds, execute the LLM call using the Actor model.
        output = call_llm(ACTOR_MODEL, prompt_to_execute)

    # -----------------------------------------------------------------------
    # RESULT PACKAGING
    # -----------------------------------------------------------------------
    if output.startswith("LLM_CALL_ERROR"):
        logger.warning(f"Execution summary: Failed for candidate {candidate['candidate_id']}. Error: {output[:200]}...")

    # Structure the result.
    result = {
        "execution_id": str(uuid.uuid4())[:8],
        "candidate_id": candidate['candidate_id'],
        "input_example_id": input_example_id,
        "input_example_data": input_data,
        "executed_prompt_text": prompt_to_execute,
        "output": output
    }

    # Return the result wrapped in a list for state accumulation.
    return {"current_execution_results": [result]}


def vote_on_result_node(task: VotingTask) -> Dict[str, List[Vote]]:
    """
    Evaluates an execution result using a specified Voter LLM (Map 2 - Vote).

    This node handles:
    1. Error Handling: Assigns a score of 0.0 if the execution resulted in an error or empty output.
    2. Evaluation Prompting: Formats the input, prompt, and output into a prompt for the Voter LLM.
    3. LLM Call: Calls the Voter LLM to get a score and critique in JSON format.
    4. Robust Parsing: Parses the JSON output and validates the presence and types of 'score' and 'critique'.

    Args:
        task: A VotingTask containing the execution result, voter model ID, and context.

    Returns:
        A dictionary containing a list with the single Vote, ready for accumulation in the state.
    """
    voter_model_id = task['voter_id']
    exec_result = task['execution_result']
    output = exec_result['output']

    logger.debug(f"Voting with {voter_model_id} on result {exec_result['execution_id']}")

    # Handle execution failures or empty outputs immediately.
    if output.startswith("LLM_CALL_ERROR") or output.strip() == "":
        vote = {
            "vote_id": str(uuid.uuid4())[:8],
            "execution_result_id": exec_result['execution_id'],
            "candidate_id": exec_result['candidate_id'],
            "input_example_id": exec_result['input_example_id'],
            "voter_id": voter_model_id,
            "score": 0.0,
            "critique": f"Automatic failure due to execution error or empty output: {output[:100]}"
        }
        return {"current_votes": [vote]}

    # Format the prompt for the Voter LLM.
    voting_prompt = format_voting_prompt(
        task_description=task['target_task_description'],
        prompt_text=task['prompt_text'],
        input_context=task['input_example_data'],
        output=output
    )

    # Call the Voter LLM, expecting a JSON response.
    raw_vote_output = call_llm(
        voter_model_id,
        voting_prompt,
        system_prompt=VOTING_SYSTEM_PROMPT,
        response_format="json"
    )

    # Parse and validate the Voter LLM output.
    try:
        vote_data = parse_json_output(raw_vote_output)

        # Validate the structure of the parsed data.
        if not isinstance(vote_data, dict):
            raise ValueError(f"Voter response was valid JSON but not a dictionary. Received Type: {type(vote_data)}")

        if "score" not in vote_data or "critique" not in vote_data:
            raise ValueError("Voter response missing required keys ('score', 'critique').")

        # Validate the types of the required fields.
        score = float(vote_data.get("score"))
        critique = str(vote_data.get("critique"))

    except (ValueError, TypeError, AttributeError) as e:
        # Handle parsing or validation failures by assigning a default score and recording the error.
        logger.error(f"Failed to parse or validate vote from {voter_model_id}: {e}")
        score = 0.0
        critique = f"Voter response parsing/validation failed: {e}"

    # Structure the vote.
    vote = {
        "vote_id": str(uuid.uuid4())[:8],
        "execution_result_id": exec_result['execution_id'],
        "candidate_id": exec_result['candidate_id'],
        "input_example_id": exec_result['input_example_id'],
        "voter_id": voter_model_id,
        "score": score,
        "critique": critique
    }

    # Return the vote wrapped in a list for state accumulation.
    return {"current_votes": [vote]}

def aggregate_and_score(state: OptimizationState) -> Dict[str, Any]:
    """
    Aggregates votes, calculates scores, updates history, and generates feedback (Reduce 2 - Aggregate and Optimize).

    This node is the final step in an iteration and performs several critical functions:
    1. Normalization: Applies Z-score normalization to the raw votes per voter to mitigate voter bias.
    2. Aggregation: Calculates consensus scores (average normalized scores) per execution and aggregate scores per prompt candidate.
    3. History Update: Updates the comprehensive performance history, ensuring elite prompts reflect their performance on the current batch.
    4. Best/Worst Tracking: Identifies the global best and worst performing examples (input/output pairs).
    5. Execution Tracing: Generates detailed execution traces for the iteration, linking inputs, outputs, scores, and individual votes.
    6. Critique Synthesis: Synthesizes individual critiques from the top-performing prompts into actionable feedback for the Optimizer LLM.

    Args:
        state: The current optimization state containing votes, execution results, and history.

    Returns:
        A dictionary updating the state with the new history, best result, synthesized critiques, global examples, iteration score history, and execution traces.
    """
    votes = state['current_votes']
    execution_results = state['current_execution_results']
    OPTIMIZER_MODEL = state['optimizer_model']
    all_prompts = state['all_tested_prompts']
    current_iteration = state['current_iteration']

    # Retrieve global tracking variables from the state.
    global_best_example = state.get('global_best_example')
    global_worst_example = state.get('global_worst_example')
    iteration_score_history = state.get('iteration_best_score_history', []).copy()

    # 1. Filter and Prepare Data
    # Ensure we only process votes relevant to the current iteration's candidates.
    valid_candidate_ids = set(state.get('current_candidates', {}).keys())
    filtered_votes = []
    seen = set()
    # Filter out potential duplicate votes (e.g., from restarts or retries if implemented).
    for v in votes or []:
        cid = v.get('candidate_id')
        key = (cid, v.get('input_example_id'), v.get('voter_id'))
        if cid in valid_candidate_ids and key not in seen:
            filtered_votes.append(v)
            seen.add(key)

    # Handle the edge case where no votes were recorded.
    if not filtered_votes:
        logger.warning("No votes recorded for current candidates in this iteration.")
        iteration_score_history.append(-1.0) # Indicate failure or baseline in the score history.
        
        return {
            "synthesized_critiques": "No valid votes recorded for synthesis.",
            "history": state['history'],
            "best_result": state['best_result'],
            "global_best_example": global_best_example,
            "global_worst_example": global_worst_example,
            "iteration_best_score_history": iteration_score_history,
            "execution_trace_history": [], # Return empty traces.
        }

    df_votes = pd.DataFrame(filtered_votes)

    # 2. Z-Score Normalization (Mitigating Voter Bias)
    def normalize(x):
        """Helper function for Z-score normalization within a group (voter)."""
        x = pd.to_numeric(x, errors='coerce')
        mean = x.mean()
        std = x.std()
        # If standard deviation is zero (all scores are the same) or calculation fails, return 0.0 (average).
        if std == 0 or np.isnan(std) or np.isnan(mean):
            return pd.Series(0.0, index=x.index)
        return (x - mean) / std

    # Define system errors that should not be included in normalization calculations.
    SYSTEM_ERRORS = ["Automatic failure", "Voter response parsing/validation failed"]
    # Create a mask for valid votes (score > 0 and not a system error).
    valid_votes_mask = (df_votes['score'] > 0) & (~df_votes['critique'].str.startswith(tuple(SYSTEM_ERRORS)))
    df_votes['normalized_score'] = 0.0 # Initialize normalized score column.
    
    if valid_votes_mask.any():
        # Apply normalization grouped by 'voter_id' only to the valid votes.
        normalized_values = df_votes.loc[valid_votes_mask].groupby('voter_id')['score'].transform(normalize)
        df_votes.loc[valid_votes_mask, 'normalized_score'] = normalized_values
        # Fill any NaNs resulting from the normalization process (e.g., if a voter only had one valid vote) with 0.0.
        df_votes['normalized_score'] = df_votes['normalized_score'].fillna(0.0)
    else:
        logger.warning("No valid votes (non-system errors and score > 0) in this iteration.")

    # 3. Aggregation
    # Calculate consensus scores: Average scores per specific execution (prompt + input example).
    consensus_scores = df_votes.groupby(['candidate_id', 'input_example_id', 'execution_result_id']).agg(
        avg_normalized_score=('normalized_score', 'mean'),
        avg_raw_score=('score', 'mean'),
        critiques=('critique', list) # Collect all critiques for the execution.
    ).reset_index()

    # Calculate aggregate scores: Average consensus scores per prompt candidate across all input examples in the batch.
    aggregate_scores = consensus_scores.groupby('candidate_id').agg(
        aggregate_score=('avg_normalized_score', 'mean'),
        raw_average_score=('avg_raw_score', 'mean')
    ).reset_index()
    
    # 4. Update GLOBAL Best and Worst Examples AND Generate Detailed Dataframe
    
    # Prepare df_details by merging execution results with the consensus scores.
    filtered_executions = [e for e in execution_results if e.get('candidate_id') in valid_candidate_ids]
    
    if filtered_executions:
        df_executions = pd.DataFrame(filtered_executions)
        # Select relevant columns from execution results.
        df_executions_subset = df_executions[['execution_id', 'executed_prompt_text', 'output']]
        # Merge based on the execution ID.
        df_details = pd.merge(consensus_scores, df_executions_subset, left_on='execution_result_id', right_on='execution_id', how='left')
    else:
        # Handle case where execution results might be missing (should be rare if votes exist).
        df_details = consensus_scores
        df_details['executed_prompt_text'] = '[Execution data missing]'
        df_details['output'] = '[Execution data missing]'

    def format_example(row) -> PerformanceExample:
        """Helper function to format a row from df_details into a PerformanceExample dictionary."""
        candidate_id = row['candidate_id']
        # Retrieve the original prompt template text.
        prompt_template = all_prompts.get(candidate_id, {}).get('prompt_text', '[Template Missing]')
        return {
            "candidate_id": candidate_id,
            "prompt_template": prompt_template,
            "input_example_id": row['input_example_id'],
            "consensus_normalized_score": row['avg_normalized_score'],
            "consensus_raw_score": row['avg_raw_score'],
            "executed_prompt_text": row.get('executed_prompt_text', '[Missing]'),
            "output": row.get('output', '[Missing]'),
            "critiques": row['critiques']
        }

    if not df_details.empty:
        # Find the best and worst examples within this iteration.
        # Use fillna to handle potential NaN scores safely during comparison.
        idx_iter_best = df_details['avg_normalized_score'].fillna(-np.inf).idxmax()
        idx_iter_worst = df_details['avg_normalized_score'].fillna(np.inf).idxmin()
        iter_best_example = format_example(df_details.loc[idx_iter_best])
        iter_worst_example = format_example(df_details.loc[idx_iter_worst])

        # Update global best example if the iteration's best is superior.
        if global_best_example is None or iter_best_example['consensus_normalized_score'] > global_best_example['consensus_normalized_score']:
            logger.info(f"New global best example found. Score: {iter_best_example['consensus_normalized_score']:.4f}")
            global_best_example = iter_best_example

        # Update global worst example if the iteration's worst is inferior.
        if global_worst_example is None or iter_worst_example['consensus_normalized_score'] < global_worst_example['consensus_normalized_score']:
            logger.info(f"New global worst example found. Score: {iter_worst_example['consensus_normalized_score']:.4f}")
            global_worst_example = iter_worst_example

    # 5. Generate Execution Traces
    # Create detailed traces for analysis, linking results back to individual votes.
    execution_traces = []
    # Create a lookup dictionary for votes based on execution_result_id.
    # We drop the 'normalized_score' from the trace record as it's specific to this iteration's normalization context and can be confusing in historical analysis.
    votes_records = df_votes.drop(columns=['normalized_score'], errors='ignore').to_dict(orient='records')
    votes_lookup = {}
    for vote in votes_records:
        exec_id = vote.get('execution_result_id')
        if exec_id not in votes_lookup:
            votes_lookup[exec_id] = []
        votes_lookup[exec_id].append(vote)

    # Iterate through the detailed results to build the traces.
    for _, row in df_details.iterrows():
        candidate_id = row['candidate_id']
        execution_id = row['execution_result_id']
        prompt_template = all_prompts.get(candidate_id, {}).get('prompt_text', '[Template Missing]')
        
        # Retrieve the list of individual votes associated with this execution.
        individual_votes = votes_lookup.get(execution_id, [])
        
        trace = {
            "iteration": current_iteration,
            "candidate_id": candidate_id,
            "input_example_id": row['input_example_id'],
            "prompt_template": prompt_template,
            "executed_prompt_text": row.get('executed_prompt_text', '[Missing]'),
            "output": row.get('output', '[Missing]'),
            "avg_normalized_score": row['avg_normalized_score'],
            "avg_raw_score": row['avg_raw_score'],
            "votes": individual_votes # Attach the detailed votes.
        }
        execution_traces.append(trace)

    # 6. Format Iteration Results and Update History (Handling Elitism)
    iteration_results = []
    # Create a dictionary from the existing history for easy updates.
    # This is crucial for handling elite prompts carried over from previous iterations.
    current_history = {item['candidate_id']: item for item in state['history']}

    for _, row in aggregate_scores.iterrows():
        candidate_id = row['candidate_id']
        if candidate_id in all_prompts:
            prompt_text = all_prompts[candidate_id]['prompt_text']
            # Handle potential NaN scores safely.
            agg_score = 0.0 if pd.isna(row['aggregate_score']) else row['aggregate_score']
            raw_avg_score = 0.0 if pd.isna(row['raw_average_score']) else row['raw_average_score']
            result = {
                "candidate_id": candidate_id,
                "prompt_text": prompt_text,
                # Use the iteration the prompt was *originally generated* (tracking lineage).
                "iteration": all_prompts[candidate_id].get('iteration', current_iteration),
                "aggregate_score": agg_score,
                "raw_average_score": raw_avg_score
            }
            iteration_results.append(result)
            
            # Update or add the result in the comprehensive history.
            # This ensures the Elite prompt's score reflects its performance on the *current* batch, overwriting its previous score.
            current_history[candidate_id] = result

    # Convert the updated history dictionary back to a list.
    new_history = list(current_history.values())
    # CRITICAL: Ensure history is sorted descending by score. This provides the Optimizer LLM with ranked input for the next iteration.
    new_history.sort(key=lambda x: x['aggregate_score'], reverse=True)

    iteration_results.sort(key=lambda x: x['aggregate_score'], reverse=True)
    
    # 6b. Update Iteration Best Score History (for plotting performance over time).
    best_score_this_iteration = iteration_results[0]['aggregate_score'] if iteration_results else 0.0
    iteration_score_history.append(best_score_this_iteration)

    # 7. Update Global Best Result
    # The best result is simply the top entry in the newly sorted history.
    best_result = new_history[0] if new_history else state['best_result']

    # 8. Synthesize Critiques (Feedback Generation)
    if iteration_results:
        # Select the top 3 candidates for critique synthesis.
        top_candidates_ids = [r['candidate_id'] for r in iteration_results[:3]]
        # Filter for valid critiques (non-error, score > 0) related to these top candidates.
        valid_critiques_df = df_votes[
            (df_votes['score'] > 0) &
            (df_votes['candidate_id'].isin(top_candidates_ids)) &
            (~df_votes['critique'].str.startswith(tuple(SYSTEM_ERRORS)))
        ]
        # Get the unique list of critiques to avoid redundancy.
        valid_critiques = valid_critiques_df['critique'].dropna().unique().tolist()
    else:
        valid_critiques = []

    if valid_critiques:
        # Format the prompt for the synthesis task.
        synthesis_prompt = format_synthesis_prompt(valid_critiques)
        # Call the Optimizer LLM (or a dedicated synthesis model) to synthesize the critiques.
        synthesized_critiques = call_llm(OPTIMIZER_MODEL, synthesis_prompt, temperature=0.3) # Low temperature for summarization.
        if synthesized_critiques.startswith("LLM_CALL_ERROR"):
            synthesized_critiques = f"Synthesis failed: {synthesized_critiques}"
    else:
        synthesized_critiques = "No valid critiques generated for the top prompts."

    # 9. Return the comprehensively updated state.
    return {
        "history": new_history,
        "best_result": best_result,
        "synthesized_critiques": synthesized_critiques,
        "global_best_example": global_best_example,
        "global_worst_example": global_worst_example,
        "iteration_best_score_history": iteration_score_history,
        "execution_trace_history": execution_traces, 
    }
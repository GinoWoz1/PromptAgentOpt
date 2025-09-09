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

# Import models and templates (Updated imports for simplified workflow)
from .models import (
    PromptCandidate, Vote, IterationResult, EvaluationTask,
    OptimizationState, PerformanceExample, ExecutionTrace,
    PromptGenerationTrace
)
from .templates import (
    format_optimizer_prompt, format_synthesis_prompt,
    OPTIMIZER_SYSTEM_PROMPT, format_voting_prompt, VOTING_SYSTEM_PROMPT
)

# Import the LLM Manager
from intelligence_assets.llm.client import llm_manager

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------

try:
    import dirtyjson
except ImportError:
    dirtyjson = None
    logger.warning("dirtyjson not installed. JSON parsing will be less robust. Install with 'pip install dirtyjson'.")


def call_llm(model_name: str, prompt: str, system_prompt: Optional[str] = None, response_format="text", **kwargs) -> str:
    """Calls the specified LLM model via the LLMClientManager."""
    try:
        return llm_manager.call_model(model_name=model_name, prompt=prompt, system_prompt=system_prompt, response_format=response_format, **kwargs)
    except ConnectionError as e:
        # Specific handling for connection issues (e.g., Ollama not running)
        logger.error(f"Connection error calling LLM {model_name}. Is the service running? Error: {e}")
        return f"LLM_CALL_ERROR: Connection failed - {e}"
    except Exception as e:
        logger.error(f"Error calling LLM {model_name}: {e}", exc_info=True)
        return f"LLM_CALL_ERROR: {e}"

def parse_json_output(output: str) -> Any:
    """
    Parses the LLM output which is expected to be a JSON string.
    Uses robust extraction and lenient parsing (dirtyjson if available).
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
        if dirtyjson:
            try:
                # Use dirtyjson to handle internal errors (escaping, trailing commas, etc.)
                logger.info("Standard JSON parsing failed. Retrying with dirtyjson.")
                return dirtyjson.loads(json_str)
            except Exception as de:
                logger.error(f"JSON parsing failed (standard and dirtyjson). Errors: {e} | {de}. Attempted to parse: {json_str[:500]}...")
                raise ValueError(f"Failed to parse JSON output. Snippet: {json_str[:200]}")
        else:
            # Fail if standard parsing failed and dirtyjson is not available
            logger.error(f"JSON parsing error: {e}. Attempted to parse: {json_str[:500]}...")
            raise ValueError(f"Failed to parse JSON output. Snippet: {json_str[:200]}")

def write_prompt_trace(trace: PromptGenerationTrace, file_path: Optional[str]):
    """Appends a prompt generation trace record to the specified JSONL file."""
    if not file_path:
        return
    try:
        # Open in append mode ('a')
        with open(file_path, 'a', encoding='utf-8') as f:
            # ensure_ascii=False is crucial for saving non-English characters correctly
            json_record = json.dumps(trace, ensure_ascii=False)
            f.write(json_record + '\n')
    except Exception as e:
        # Log the error but do not stop the optimization process
        logger.error(f"CRITICAL: Failed to write prompt generation trace to {file_path}: {e}")

# -----------------------------------------------------------------------
# Node Definitions
# -----------------------------------------------------------------------

def generate_prompts(state: OptimizationState) -> Dict[str, Any]:
    """
    Step 1: Generate N prompts, ensure novelty, implement elitism, select the mini-batch, and trace the process.
    """
    iteration = state['current_iteration'] + 1
    N_CANDIDATES = state['N_CANDIDATES']
    OPTIMIZER_MODEL = state['optimizer_model']
    SIMILARITY_THRESHOLD = 0.90
    MAX_RETRIES = 3
    TRACE_FILE_PATH = state.get('prompt_trace_file_path')

    logger.info(f"\n--- Starting Iteration {iteration} --- Model: {OPTIMIZER_MODEL} ---")

    # 1. Prepare input and Elitism
    performance_history = state['history']
    synthesized_critiques = state['synthesized_critiques']
    # We work on a copy of all_tested_prompts, updating it as we accept new prompts
    all_tested_prompts = state['all_tested_prompts'].copy()

    new_candidates = {}
    elite_tracking = []

    # --- Elitism Implementation ---
    if state['best_result'] and N_CANDIDATES > 1:
        elite_id = state['best_result']['candidate_id']
        elite_candidate = all_tested_prompts.get(elite_id)

        if elite_candidate:
            logger.info(f"Elitism: Carrying forward Elite prompt (ID: {elite_id})")
            # Ensure the dictionary is copied so we don't modify the historical record inadvertently
            elite_copy = elite_candidate.copy()
            # Mark it as elite for this iteration
            elite_copy['is_elite'] = True
            new_candidates[elite_id] = elite_copy
            elite_tracking.append({
                "candidate_id": elite_id,
                "prompt_text": elite_copy['prompt_text'],
                "is_elite": True,
                "similarity_score": 1.0 # Elite prompts are definitionally similar to themselves
            })
        else:
            logger.warning(f"Elite candidate ID {elite_id} not found in all_tested_prompts.")

    # Prepare performance history summary for tracing (Top 10)
    MAX_HISTORY_FOR_TRACE = 10
    history_summary = [
        {"candidate_id": h['candidate_id'], "aggregate_score": h['aggregate_score'], "iteration": h['iteration']}
        for h in performance_history[:MAX_HISTORY_FOR_TRACE]
    ]

    # 2. Generate and Validate Prompts (The Novelty Loop)
    attempts = 0
    generated_texts = []

    while len(new_candidates) < N_CANDIDATES and attempts < MAX_RETRIES:
        attempts += 1

        # Initialize trace dictionary for this specific attempt
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

        # Calculate how many NEW prompts we need (accounts for elites)
        needed = N_CANDIDATES - len(new_candidates)
        trace['num_requested'] = needed

        if needed <= 0:
            break

        logger.info(f"Attempt {attempts}/{MAX_RETRIES}: Generating {needed} new prompts.")

        optimizer_prompt = format_optimizer_prompt(
            state['target_task_description'],
            synthesized_critiques,
            performance_history, # Passing structured history
            needed,
            iteration
        )

        raw_output = call_llm(
            model_name=OPTIMIZER_MODEL,
            prompt=optimizer_prompt,
            system_prompt=OPTIMIZER_SYSTEM_PROMPT,
            response_format="json",
            temperature=0.7 # Optimizer needs creativity
        )

        if raw_output.startswith("LLM_CALL_ERROR"):
            logger.error(f"LLM call failed (Attempt {attempts}): {raw_output}")
            # Record failure in trace and write immediately
            trace['error_message'] = raw_output
            write_prompt_trace(trace, TRACE_FILE_PATH)
            continue

        trace['llm_call_successful'] = True

        try:
            parsed_output = parse_json_output(raw_output)

            if isinstance(parsed_output, dict) and "prompts" in parsed_output and isinstance(parsed_output["prompts"], list):
                generated_texts = parsed_output["prompts"]
                logger.info(f"Successfully parsed {len(generated_texts)} prompts.")
                trace['num_generated'] = len(generated_texts)
            else:
                logger.warning(f"Unexpected output format (Attempt {attempts}). Output: {raw_output[:200]}")
                generated_texts = []
                # Record error and write trace
                trace['error_message'] = "Unexpected JSON format (missing 'prompts' key or not a list)."
                write_prompt_trace(trace, TRACE_FILE_PATH)
                continue
        except ValueError as e:
            logger.warning(f"JSON parsing failed (Attempt {attempts}): {str(e)}")
            generated_texts = []
            # Record error and write trace
            trace['error_message'] = f"JSON parsing failed: {str(e)}"
            write_prompt_trace(trace, TRACE_FILE_PATH)
            continue

        if not generated_texts:
            # Write trace if list was empty but parsing succeeded
            if trace['llm_call_successful'] and trace['num_generated'] == 0:
                trace['error_message'] = "LLM successfully generated an empty list of prompts."
                write_prompt_trace(trace, TRACE_FILE_PATH)
            continue

        try:
            # Using default provider (OpenAI) for embeddings for consistency
            generated_embeddings = llm_manager.get_embeddings(generated_texts)
        except Exception as e:
            logger.error(f"Embedding generation failed (Attempt {attempts}): {str(e)}")
            # Record error and write trace
            trace['error_message'] = f"Embedding generation failed: {str(e)}"
            write_prompt_trace(trace, TRACE_FILE_PATH)
            continue

        # Comparison set includes all historical prompts
        comparison_embeddings = [p['embedding'] for p in all_tested_prompts.values() if p.get('embedding')]

        rejected_count = 0
        accepted_count = 0

        # Initialize lists for detailed tracking within this attempt
        attempt_accepted = []
        attempt_rejected = []

        for text, embedding in zip(generated_texts, generated_embeddings):
            if len(new_candidates) >= N_CANDIDATES:
                break

            is_novel = True
            max_similarity = 0.0
            if comparison_embeddings:
                try:
                    embedding_np = np.array(embedding).reshape(1, -1)
                    comparison_embeddings_np = np.array(comparison_embeddings)
                    similarities = cosine_similarity(embedding_np, comparison_embeddings_np)[0]
                    max_similarity = np.max(similarities)
                    if max_similarity > SIMILARITY_THRESHOLD:
                        rejected_count += 1
                        is_novel = False
                except Exception as e:
                    logger.error(f"Similarity check failed (likely dimension mismatch): {str(e)}. Accepting prompt by default.")
                    is_novel = True # Proceed if similarity check fails

            if is_novel:
                candidate_id = str(uuid.uuid4())[:8]
                candidate = {
                    "candidate_id": candidate_id,
                    "prompt_text": text,
                    "iteration": iteration,
                    "embedding": embedding,
                    "is_elite": False
                }
                new_candidates[candidate_id] = candidate
                # Crucial: Add to all_tested_prompts immediately
                all_tested_prompts[candidate_id] = candidate
                comparison_embeddings.append(embedding)
                accepted_count += 1
                # Track acceptance details
                attempt_accepted.append({
                    "candidate_id": candidate_id,
                    "prompt_text": text,
                    "is_elite": False,
                    # Cast numpy float to native Python float for JSON serialization
                    "similarity_score": float(max_similarity)
                })
            # Track rejection details
            else:
                attempt_rejected.append({
                    "prompt_text": text,
                    "reason": f"Similarity threshold ({SIMILARITY_THRESHOLD}) exceeded",
                    "similarity_score": float(max_similarity)
                })

        logger.info(f"Attempt {attempts} results: {accepted_count} accepted, {rejected_count} rejected (similarity).")

        # Finalize and write the trace for this attempt
        trace['num_accepted'] = accepted_count
        trace['num_rejected_similarity'] = rejected_count
        trace['accepted_prompts'] = attempt_accepted
        trace['rejected_prompts'] = attempt_rejected

        # If this is the first attempt, include the elite prompts in the trace for completeness
        if attempts == 1 and elite_tracking:
            trace['accepted_prompts'] = elite_tracking + trace['accepted_prompts']
            trace['num_accepted'] += len(elite_tracking)


        # Write the completed trace for the attempt
        write_prompt_trace(trace, TRACE_FILE_PATH)

    # 3. Fallback Mechanism (if needed)
    if not new_candidates:
        logger.error("CRITICAL ERROR: No novel prompts were generated after all attempts. Initiating fallback.")

        # Initialize trace for the fallback mechanism
        fallback_trace = {
            "iteration": iteration,
            "attempt": "FALLBACK",
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
            # Generalized Fallback: Use the task description to generate a relevant prompt.
            raw_output = call_llm(
                model_name=OPTIMIZER_MODEL,
                # Instruct the LLM to generate a simple prompt based on the task description.
                prompt=f"Generate one simple, standard prompt template for the following task. Identify the necessary input variables and use Dollar-style placeholders (e.g., $input_text).\n\n<task_description>\n{state['target_task_description']}\n</task_description>",
                system_prompt="You are a helpful assistant. Generate a prompt template based on the task description. Use $ for placeholders.",
                response_format="text",
                temperature=0.9 # High temperature for fallback creativity
            )

            if not raw_output.startswith("LLM_CALL_ERROR"):
                fallback_trace['llm_call_successful'] = True
                fallback_trace['num_generated'] = 1
                fallback_trace['error_message'] = None # Clear the initiation message

                candidate_id = str(uuid.uuid4())[:8]
                try:
                    fallback_embedding = llm_manager.get_embeddings([raw_output.strip()])[0]
                except Exception as e:
                    logger.warning(f"Fallback embedding failed: {e}. Using dummy embedding.")
                    # Use a dummy embedding if the service fails. Size 1536 is common (e.g., OpenAI small).
                    fallback_embedding = [0.0] * 1536
                candidate = {
                    "candidate_id": candidate_id,
                    "prompt_text": raw_output.strip(),
                    "iteration": iteration,
                    "embedding": fallback_embedding,
                    "is_elite": False
                }
                new_candidates[candidate_id] = candidate
                all_tested_prompts[candidate_id] = candidate

                # Update trace with acceptance details
                fallback_trace['num_accepted'] = 1
                fallback_trace['accepted_prompts'].append({
                    "candidate_id": candidate_id,
                    "prompt_text": candidate['prompt_text'],
                    "is_elite": False,
                    "similarity_score": 0.0 # Fallback bypasses similarity check
                })
                write_prompt_trace(fallback_trace, TRACE_FILE_PATH)

            else:
                # Record failure and write trace before raising error
                fallback_trace['error_message'] = f"Fallback LLM call failed: {raw_output}"
                write_prompt_trace(fallback_trace, TRACE_FILE_PATH)
                raise RuntimeError(f"Optimizer LLM failed completely (including fallback): {raw_output}")
        except Exception as e:
            # Ensure trace is written if an unexpected error occurs during fallback handling
            if fallback_trace['error_message'] == "Failed to generate novel prompts after max retries. Initiating fallback.":
                fallback_trace['error_message'] = f"Unexpected error during fallback execution: {str(e)}"
                write_prompt_trace(fallback_trace, TRACE_FILE_PATH)
            raise RuntimeError(f"Optimizer LLM failed to generate novel prompts after multiple attempts and fallback failed: {str(e)}")

    # 4. Select Mini-Batch
    dataset = state['input_dataset']
    batch_size = min(state['MINI_BATCH_SIZE'], len(dataset))

    if batch_size > 0:
        # Random sampling without replacement for stochastic evaluation
        indices = np.random.choice(len(dataset), batch_size, replace=False)
        current_mini_batch = [dataset[i] for i in indices]
    else:
        current_mini_batch = []
        if len(dataset) == 0:
            logger.error("Input dataset is empty.")
            raise ValueError("Input dataset cannot be empty.")

    logger.info(f"Iteration {iteration} ready. {len(new_candidates)} candidates generated. Mini-batch size: {len(current_mini_batch)}.")

    return {
        "current_iteration": iteration,
        "current_candidates": new_candidates,
        # Pass the updated dictionary containing the new prompts
        "all_tested_prompts": all_tested_prompts,
        "current_mini_batch": current_mini_batch,
        # Reset accumulator for the new iteration
        "current_votes": [],
    }

# NEW: Combined execution and voting node.
def evaluate_prompt_node(task: EvaluationTask) -> Dict[str, List[Vote]]:
    """
    Step 2 (Parallel): Executes the prompt template on an input example and then sequentially calls the voters.
    This simplifies the graph structure by combining execution and voting into one parallel step.
    """
    candidate = task['candidate']
    input_example = task['input_example']
    ACTOR_MODEL = task['actor_model']
    VOTER_ENSEMBLE = task['voter_ensemble']
    TASK_DESCRIPTION = task['target_task_description']

    input_data = input_example.get('data', {})
    input_example_id = input_example.get('id', 'N/A')
    # Generate a unique ID for this specific execution event
    execution_id = str(uuid.uuid4())[:8]

    logger.debug(f"Evaluating prompt {candidate['candidate_id']} on input {input_example_id}")

    # -----------------------------------------------------------------------
    # 1. EXECUTION
    # -----------------------------------------------------------------------
    prompt_to_execute = "[Template Rendering Failed]"
    output = ""

    try:
        # Use string.Template ($ placeholders) for robust rendering.
        template = Template(candidate['prompt_text'])
        # safe_substitute prevents errors if a placeholder is missing and ignores {}.
        prompt_to_execute = template.safe_substitute(input_data)
    except Exception as e:
        # Catches potential errors like invalid $ syntax (e.g., ValueError for a bare $)
        error_msg = f"Template rendering failed (string.Template). Check for invalid syntax (e.g., bare $): {e}."
        logger.error(f"Prompt format error for {candidate['candidate_id']}: {error_msg}")
        output = f"LLM_CALL_ERROR: {error_msg}"
    else:
        # If formatting succeeds, execute the LLM call. Actor typically uses low temperature.
        output = call_llm(ACTOR_MODEL, prompt_to_execute, temperature=0.1)

    if output.startswith("LLM_CALL_ERROR"):
        logger.warning(f"Execution failed for candidate {candidate['candidate_id']}. Error: {output[:200]}...")

    # -----------------------------------------------------------------------
    # 2. VOTING (Sequential within this node)
    # -----------------------------------------------------------------------
    votes = []

    # Determine if voting should proceed or if an automatic failure vote is needed.
    execution_failed = output.startswith("LLM_CALL_ERROR") or output.strip() == ""

    # Iterate sequentially through the voters
    for voter_model_id in VOTER_ENSEMBLE:
        logger.debug(f"Voting with {voter_model_id} on execution {execution_id}")

        score = 0.0
        critique = ""

        if execution_failed:
            critique = f"Automatic failure due to execution error or empty output: {output[:100]}"
        else:
            # If execution succeeded, call the voter LLM
            voting_prompt = format_voting_prompt(
                task_description=TASK_DESCRIPTION,
                prompt_text=candidate['prompt_text'],
                input_context=input_data,
                output=output
            )

            raw_vote_output = call_llm(
                voter_model_id,
                voting_prompt,
                system_prompt=VOTING_SYSTEM_PROMPT,
                response_format="json",
                temperature=0.1 # Voters should be objective and consistent
            )

            # Robust parsing and validation
            try:
                vote_data = parse_json_output(raw_vote_output)

                if not isinstance(vote_data, dict):
                     raise ValueError(f"Voter response was valid JSON but not a dictionary. Received Type: {type(vote_data)}")

                if "score" not in vote_data or "critique" not in vote_data:
                     raise ValueError("Voter response missing required keys ('score', 'critique').")

                score = float(vote_data.get("score"))
                # Constrain score between 1 and 10 as requested in the prompt
                score = max(1.0, min(10.0, score))
                critique = str(vote_data.get("critique"))

            except (ValueError, TypeError, AttributeError) as e:
                logger.error(f"Failed to parse or validate vote from {voter_model_id}: {e}. Raw output: {raw_vote_output[:200]}")
                score = 0.0 # Ensure score is reset if parsing fails
                critique = f"Voter response parsing/validation failed: {e}"

        # Package the vote
        vote = {
            "vote_id": str(uuid.uuid4())[:8],
            "candidate_id": candidate['candidate_id'],
            "input_example_id": input_example_id,
            "voter_id": voter_model_id,
            "score": score,
            "critique": critique,
            # Attach execution details directly to the vote for traceability in aggregation
            "execution_id": execution_id,
            "executed_prompt_text": prompt_to_execute,
            "output": output,
            "input_example_data": input_data
        }
        votes.append(vote)

    # Return the list of votes generated in this node, which LangGraph accumulates into 'current_votes'
    return {"current_votes": votes}


def aggregate_and_score(state: OptimizationState) -> Dict[str, Any]:
    """
    Step 3 (Reduce): Aggregate votes, normalize scores, update global best/worst examples, track iteration performance, generate detailed traces, and synthesize critiques.
    """
    votes = state['current_votes']
    OPTIMIZER_MODEL = state['optimizer_model']
    all_prompts = state['all_tested_prompts']
    current_iteration = state['current_iteration']

    global_best_example = state.get('global_best_example')
    global_worst_example = state.get('global_worst_example')
    iteration_score_history = state.get('iteration_best_score_history', []).copy()

    logger.info(f"--- Aggregation and Scoring (Reduce) ---")
    logger.info(f"Accumulated {len(votes)} votes.")

    # 1. Filter and Prepare Data
    valid_candidate_ids = set(state.get('current_candidates', {}).keys())
    filtered_votes = []
    seen = set()
    # Filter out potential duplicate votes if any (e.g. from retries)
    for v in votes or []:
        cid = v.get('candidate_id')
        # Key includes execution_id and voter_id to distinguish unique evaluation events
        key = (v.get('execution_id'), v.get('voter_id'))
        if cid in valid_candidate_ids and key not in seen:
            filtered_votes.append(v)
            seen.add(key)

    if not filtered_votes:
        logger.warning("No votes recorded for current candidates in this iteration.")
        # Append 0.0 if no valid votes occurred.
        iteration_score_history.append(0.0)

        return {
            "synthesized_critiques": "No valid votes recorded for synthesis.",
            "history": state['history'],
            "best_result": state['best_result'],
            "global_best_example": global_best_example,
            "global_worst_example": global_worst_example,
            "iteration_best_score_history": iteration_score_history,
            "execution_trace_history": [],
        }

    df_votes = pd.DataFrame(filtered_votes)

    # 2. Z-Score Normalization (per voter)
    def normalize(x):
        """Calculates Z-scores for a series."""
        x = pd.to_numeric(x, errors='coerce')
        mean = x.mean()
        std = x.std()
        # Handle cases with no variation (std=0) or insufficient data (NaN)
        if std == 0 or np.isnan(std) or np.isnan(mean):
            return pd.Series(0.0, index=x.index)
        return (x - mean) / std

    # Identify votes resulting from system errors (parsing failures, execution errors)
    SYSTEM_ERRORS = ["Automatic failure", "Voter response parsing/validation failed"]
    # We only normalize valid votes (score > 0 and not a system error critique).
    valid_votes_mask = (df_votes['score'] > 0) & (~df_votes['critique'].str.startswith(tuple(SYSTEM_ERRORS)))
    df_votes['normalized_score'] = 0.0 # Initialize normalized score column

    if valid_votes_mask.any():
        # Apply normalization grouped by voter ID to the valid votes
        normalized_values = df_votes.loc[valid_votes_mask].groupby('voter_id')['score'].transform(normalize)
        df_votes.loc[valid_votes_mask, 'normalized_score'] = normalized_values
        # Fill any NaNs resulting from transform (e.g., if a voter only had one valid vote)
        df_votes['normalized_score'] = df_votes['normalized_score'].fillna(0.0)
    else:
        logger.warning("No valid votes (non-system errors and score > 0) in this iteration for normalization.")

    # 3. Aggregation
    # First, aggregate across voters for each execution (consensus score)
    # Group by the unique execution event (identified by execution_id)
    consensus_scores = df_votes.groupby(['candidate_id', 'input_example_id', 'execution_id']).agg(
        avg_normalized_score=('normalized_score', 'mean'),
        avg_raw_score=('score', 'mean'),
        critiques=('critique', list)
    ).reset_index()

    # Second, aggregate across the mini-batch for each candidate (final iteration score)
    aggregate_scores = consensus_scores.groupby('candidate_id').agg(
        aggregate_score=('avg_normalized_score', 'mean'),
        raw_average_score=('avg_raw_score', 'mean')
    ).reset_index()

    # 4. Update GLOBAL Best/Worst Examples AND Generate Traces

    # Prepare df_details by extracting execution details attached to the votes.
    execution_details_cols = ['execution_id', 'executed_prompt_text', 'output', 'input_example_data']

    # Extract unique execution details from the votes dataframe.
    if all(col in df_votes.columns for col in execution_details_cols):
        # We extract the details by taking the unique execution_ids from the votes dataframe.
        df_executions_unique = df_votes[execution_details_cols].drop_duplicates(subset=['execution_id'])
        # Merge consensus scores with the execution details
        df_details = pd.merge(consensus_scores, df_executions_unique, on='execution_id', how='left')
    else:
        # Fallback if details are missing (should not happen with the current implementation)
        logger.error("CRITICAL: Execution details missing in votes DataFrame. Traceability will be incomplete.")
        df_details = consensus_scores
        df_details['executed_prompt_text'] = '[Data missing in Vote]'
        df_details['output'] = '[Data missing in Vote]'
        df_details['input_example_data'] = {}


    def format_example(row) -> PerformanceExample:
        """Helper to format a row from df_details into a PerformanceExample."""
        candidate_id = row['candidate_id']
        prompt_template = all_prompts.get(candidate_id, {}).get('prompt_text', '[Template Missing]')
        return {
            "candidate_id": candidate_id,
            "prompt_template": prompt_template,
            "input_example_id": row['input_example_id'],
            "consensus_normalized_score": row['avg_normalized_score'],
            "consensus_raw_score": row['avg_raw_score'],
            "executed_prompt_text": row.get('executed_prompt_text', '[Missing]'),
            "output": row.get('output', '[Missing]'),
            # Critiques are already aggregated in consensus_scores
            "critiques": row.get('critiques', [])
        }

    if not df_details.empty:
        # Handle potential NaN scores when finding min/max
        idx_iter_best = df_details['avg_normalized_score'].fillna(-np.inf).idxmax()
        idx_iter_worst = df_details['avg_normalized_score'].fillna(np.inf).idxmin()
        iter_best_example = format_example(df_details.loc[idx_iter_best])
        iter_worst_example = format_example(df_details.loc[idx_iter_worst])

        # Update global tracking
        if global_best_example is None or iter_best_example['consensus_normalized_score'] > global_best_example['consensus_normalized_score']:
            logger.info(f"New global best example found. Score: {iter_best_example['consensus_normalized_score']:.4f}")
            global_best_example = iter_best_example

        if global_worst_example is None or iter_worst_example['consensus_normalized_score'] < global_worst_example['consensus_normalized_score']:
            logger.info(f"New global worst example found. Score: {iter_worst_example['consensus_normalized_score']:.4f}")
            global_worst_example = iter_worst_example

    # --- Generate Execution Traces ---
    execution_traces = []
    # Create a lookup dictionary for votes based on execution_id.
    # We need to prepare the votes for the trace by removing the redundant execution details we added earlier,
    # and also remove the 'normalized_score' as it's specific to this iteration's context.
    cols_to_drop = ['normalized_score', 'executed_prompt_text', 'output', 'input_example_data']
    # Create a list of cleaned vote dictionaries
    votes_records = df_votes.drop(columns=cols_to_drop, errors='ignore').to_dict(orient='records')

    votes_lookup = {}
    for vote in votes_records:
        exec_id = vote.get('execution_id')
        if exec_id not in votes_lookup:
            votes_lookup[exec_id] = []
        votes_lookup[exec_id].append(vote)

    for _, row in df_details.iterrows():
        candidate_id = row['candidate_id']
        execution_id = row['execution_id']
        prompt_template = all_prompts.get(candidate_id, {}).get('prompt_text', '[Template Missing]')

        # Retrieve the list of (cleaned) votes associated with this execution
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
            "votes": individual_votes
        }
        execution_traces.append(trace)

    # 5. Format Iteration Results and Update History (Handles Elitism)
    iteration_results = []
    # Create a dictionary from the existing history for easy updates
    current_history = {item['candidate_id']: item for item in state['history']}

    for _, row in aggregate_scores.iterrows():
        candidate_id = row['candidate_id']
        if candidate_id in all_prompts:
            prompt_text = all_prompts[candidate_id]['prompt_text']
            agg_score = 0.0 if pd.isna(row['aggregate_score']) else row['aggregate_score']
            raw_avg_score = 0.0 if pd.isna(row['raw_average_score']) else row['raw_average_score']
            result = {
                "candidate_id": candidate_id,
                "prompt_text": prompt_text,
                # Use the iteration the prompt was *originally generated*
                "iteration": all_prompts[candidate_id].get('iteration', current_iteration),
                "aggregate_score": agg_score,
                "raw_average_score": raw_avg_score
            }
            iteration_results.append(result)

            # Update or add the result in the comprehensive history
            # This ensures the Elite prompt's score reflects its performance on the current batch
            current_history[candidate_id] = result

    # Convert the updated history dictionary back to a list and sort it
    new_history = list(current_history.values())
    # CRITICAL: Ensure history is sorted descending by score for the next iteration's input
    new_history.sort(key=lambda x: x['aggregate_score'], reverse=True)

    iteration_results.sort(key=lambda x: x['aggregate_score'], reverse=True)

    # 5b. Update Iteration Best Score History
    # We track the best score achieved by any candidate *in this iteration* for early stopping.
    best_score_this_iteration = iteration_results[0]['aggregate_score'] if iteration_results else 0.0
    iteration_score_history.append(best_score_this_iteration)

    # 6. Update Global Best Result
    # The best result overall is the top entry in the sorted new_history
    best_result = new_history[0] if new_history else state['best_result']

    # 7. Synthesize Critiques
    # Focus synthesis on the top 3 performers of this iteration
    if iteration_results:
        top_candidates_ids = [r['candidate_id'] for r in iteration_results[:3]]
        # Select critiques that are valid (score > 0, not system error) and belong to the top candidates
        valid_critiques_df = df_votes[
            (df_votes['score'] > 0) &
            (df_votes['candidate_id'].isin(top_candidates_ids)) &
            (~df_votes['critique'].str.startswith(tuple(SYSTEM_ERRORS)))
        ]
        valid_critiques = valid_critiques_df['critique'].dropna().unique().tolist()
    else:
        valid_critiques = []

    if valid_critiques:
        logger.info(f"Synthesizing {len(valid_critiques)} critiques...")
        synthesis_prompt = format_synthesis_prompt(valid_critiques)
        # Use the optimizer model for synthesis with low temperature for coherence.
        synthesized_critiques = call_llm(OPTIMIZER_MODEL, synthesis_prompt, temperature=0.2)
        if synthesized_critiques.startswith("LLM_CALL_ERROR"):
            synthesized_critiques = f"Synthesis failed: {synthesized_critiques}"
    else:
        synthesized_critiques = "No valid critiques generated for the top prompts."

    logger.info(f"Iteration {current_iteration} scoring complete. Best score this iteration: {best_score_this_iteration:.4f}.")

    return {
        "history": new_history,
        "best_result": best_result,
        "synthesized_critiques": synthesized_critiques,
        "global_best_example": global_best_example,
        "global_worst_example": global_worst_example,
        "iteration_best_score_history": iteration_score_history,
        # Accumulate the detailed traces from this iteration
        "execution_trace_history": execution_traces,
    }
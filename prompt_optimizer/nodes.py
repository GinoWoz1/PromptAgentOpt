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

from sklearn.metrics.pairwise import cosine_similarity

# Import models and templates (Updated imports)
from .models import (
    PromptCandidate, ExecutionResult, Vote, IterationResult, ExecutionTask,
    VotingTask, OptimizationState, PerformanceExample, ExecutionTrace # Added ExecutionTrace
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
    logger.warning("dirtyjson not installed. JSON parsing will be less robust. Install with 'uv pip install dirtyjson'.")


def call_llm(model_name: str, prompt: str, system_prompt: Optional[str] = None, response_format="text", **kwargs) -> str:
    """Calls the specified LLM model via the LLMClientManager."""
    try:
        return llm_manager.call_model(model_name=model_name, prompt=prompt, system_prompt=system_prompt, response_format=response_format, **kwargs)
    except Exception as e:
        logger.error(f"Error calling LLM {model_name}: {e}")
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

# -----------------------------------------------------------------------
# Node Definitions
# -----------------------------------------------------------------------

# Synchronization Barrier (Reduce 1)
def synchronize_executions(state: OptimizationState) -> Dict[str, Any]:
    """
    Acts as a synchronization barrier (Reduce 1).
    """
    num_results = len(state['current_execution_results'])
    logger.info(f"--- Synchronization Point (Reduce 1) ---")
    logger.info(f"Accumulated {num_results} execution results. Proceeding to voting.")
    return {}

def generate_prompts(state: OptimizationState) -> Dict[str, Any]:
    """
    Step 1: Generate N prompts, ensure novelty (Q1), implement elitism (Q1), and select the mini-batch.
    """
    iteration = state['current_iteration'] + 1
    N_CANDIDATES = state['N_CANDIDATES']
    OPTIMIZER_MODEL = state['optimizer_model']
    # NEW (Q1): Relaxed threshold
    SIMILARITY_THRESHOLD = 0.90
    MAX_RETRIES = 3

    logger.info(f"--- Starting Iteration {iteration} --- Model: {OPTIMIZER_MODEL} ---")

    # 1. Prepare input and Elitism
    # NEW (Q1): Use performance history for guidance
    performance_history = state['history']
    synthesized_critiques = state['synthesized_critiques']
    # We work on a copy of all_tested_prompts, updating it as we accept new prompts
    all_tested_prompts = state['all_tested_prompts'].copy()

    new_candidates = {}
    
    # --- Elitism Implementation (Q1) ---
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
        else:
            logger.warning(f"Elite candidate ID {elite_id} not found in all_tested_prompts.")

    # 2. Generate and Validate Prompts (The Novelty Loop)
    attempts = 0
    generated_texts = []

    while len(new_candidates) < N_CANDIDATES and attempts < MAX_RETRIES:
        attempts += 1
        # Calculate how many NEW prompts we need (accounts for elites)
        needed = N_CANDIDATES - len(new_candidates)

        if needed <= 0:
            break

        logger.info(f"DEBUG: Attempt {attempts}/{MAX_RETRIES} to generate {needed} prompts")

        # NEW (Q1): Use the updated prompt format function
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
            temperature=0.7
        )

        if raw_output.startswith("LLM_CALL_ERROR"):
            logger.error(f"DEBUG: LLM call failed: {raw_output}")
            continue

        try:
            parsed_output = parse_json_output(raw_output)

            if isinstance(parsed_output, dict) and "prompts" in parsed_output and isinstance(parsed_output["prompts"], list):
                generated_texts = parsed_output["prompts"]
                logger.info(f"DEBUG: Successfully parsed {len(generated_texts)} prompts")
            else:
                logger.warning(f"DEBUG: Unexpected output format. Output: {raw_output[:200]}")
                generated_texts = []
                continue
        except ValueError as e:
            logger.warning(f"DEBUG: JSON parsing failed: {str(e)}")
            generated_texts = []
            continue

        if not generated_texts:
            continue

        try:
            generated_embeddings = llm_manager.get_embeddings(generated_texts)
        except Exception as e:
            logger.error(f"DEBUG: Embedding generation failed: {str(e)}")
            continue

        # Comparison set includes all historical prompts
        comparison_embeddings = [p['embedding'] for p in all_tested_prompts.values() if p.get('embedding')]

        rejected_count = 0
        accepted_count = 0

        for text, embedding in zip(generated_texts, generated_embeddings):
            if len(new_candidates) >= N_CANDIDATES:
                break

            is_novel = True
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
                    logger.error(f"DEBUG: Similarity check failed: {str(e)}")
                    is_novel = True

            if is_novel:
                candidate_id = str(uuid.uuid4())[:8]
                candidate = PromptCandidate(
                    candidate_id=candidate_id,
                    prompt_text=text,
                    iteration=iteration,
                    embedding=embedding,
                    is_elite=False # NEW (Q1)
                )
                new_candidates[candidate_id] = candidate
                # Crucial: Add to all_tested_prompts immediately
                all_tested_prompts[candidate_id] = candidate
                comparison_embeddings.append(embedding)
                accepted_count += 1

        logger.info(f"DEBUG: After attempt {attempts}: {accepted_count} accepted, {rejected_count} rejected")

    if not new_candidates:
        logger.error("DEBUG: CRITICAL ERROR - No novel prompts were generated after all attempts. Using fallback.")
        try:
            # CRITICAL UPDATE: Change placeholders from {} to $ for fallback consistency
            raw_output = call_llm(
                model_name=OPTIMIZER_MODEL,
                prompt="Generate one simple prompt for analyzing Chinese Buddhist text translation. Include placeholders for $chinese_context, $target_info, and $english_translation_block.",
                system_prompt="You are a helpful assistant. Generate a prompt template. Use $ for placeholders.",
                response_format="text",
                temperature=0.9
            )

            if not raw_output.startswith("LLM_CALL_ERROR"):
                candidate_id = str(uuid.uuid4())[:8]
                try:
                    fallback_embedding = llm_manager.get_embeddings([raw_output.strip()])[0]
                except:
                    fallback_embedding = [0.0] * 10 
                candidate = PromptCandidate(
                    candidate_id=candidate_id,
                    prompt_text=raw_output.strip(),
                    iteration=iteration,
                    embedding=fallback_embedding,
                    is_elite=False # NEW (Q1)
                )
                new_candidates[candidate_id] = candidate
                all_tested_prompts[candidate_id] = candidate
            else:
                raise RuntimeError(f"Optimizer LLM failed completely: {raw_output}")
        except Exception as e:
            raise RuntimeError(f"Optimizer LLM failed to generate novel prompts after multiple attempts: {str(e)}")

    dataset = state['input_dataset']
    batch_size = min(state['MINI_BATCH_SIZE'], len(dataset))

    if batch_size > 0:
        indices = np.random.choice(len(dataset), batch_size, replace=False)
        current_mini_batch = [dataset[i] for i in indices]
    else:
        current_mini_batch = []
        if len(dataset) == 0:
            logger.error("DEBUG: Input dataset is empty")
            raise ValueError("Input dataset cannot be empty")

    return {
        "current_iteration": iteration,
        "current_candidates": new_candidates,
        # Pass the updated dictionary containing the new prompts
        "all_tested_prompts": all_tested_prompts, 
        "current_mini_batch": current_mini_batch,
        "current_execution_results": [],
        "current_votes": [],
    }

def execute_prompt_node(task: ExecutionTask) -> Dict[str, List[ExecutionResult]]:
    """
    Executes the prompt template on a specific input example dynamically.
    Uses string.Template ($ placeholders) for robust rendering.
    """
    candidate = task['candidate']
    input_example = task['input_example']
    ACTOR_MODEL = task['actor_model']

    input_data = input_example.get('data', {})
    input_example_id = input_example.get('id', 'N/A')

    logger.debug(f"Executing prompt {candidate['candidate_id']} on input {input_example_id} with {ACTOR_MODEL}")

    prompt_to_execute = "[Template Rendering Failed]"
    output = ""

    # -----------------------------------------------------------------------
    # DYNAMIC INTERPOLATION AND EXECUTION (Updated)
    # -----------------------------------------------------------------------
    try:
        # Use string.Template instead of .format()
        template = Template(candidate['prompt_text'])
        # safe_substitute prevents errors if a placeholder is missing and ignores {}.
        prompt_to_execute = template.safe_substitute(input_data)
    except Exception as e:
        # Catches potential errors like invalid $ syntax (e.g., ValueError for a bare $)
        error_msg = f"Template rendering failed (string.Template). Check for invalid syntax (e.g., bare $): {e}."
        logger.error(f"Prompt format error for {candidate['candidate_id']}: {error_msg}")
        output = f"LLM_CALL_ERROR: {error_msg}"
    else:
        # If formatting succeeds, execute the LLM call.
        output = call_llm(ACTOR_MODEL, prompt_to_execute)

    # -----------------------------------------------------------------------
    # RESULT PACKAGING
    # -----------------------------------------------------------------------
    if output.startswith("LLM_CALL_ERROR"):
        logger.warning(f"Execution summary: Failed for candidate {candidate['candidate_id']}. Error: {output[:200]}...")

    result = ExecutionResult(
        execution_id=str(uuid.uuid4())[:8],
        candidate_id=candidate['candidate_id'],
        input_example_id=input_example_id,
        input_example_data=input_data,
        executed_prompt_text=prompt_to_execute,
        output=output
    )

    return {"current_execution_results": [result]}


def vote_on_result_node(task: VotingTask) -> Dict[str, List[Vote]]:
    """Evaluates the execution result using a Voter LLM. Includes robust parsing and validation."""
    voter_model_id = task['voter_id']
    exec_result = task['execution_result']
    output = exec_result['output']

    logger.debug(f"Voting with {voter_model_id} on result {exec_result['execution_id']}")

    if output.startswith("LLM_CALL_ERROR") or output.strip() == "":
        vote = Vote(
            vote_id=str(uuid.uuid4())[:8],
            execution_result_id=exec_result['execution_id'],
            candidate_id=exec_result['candidate_id'],
            input_example_id=exec_result['input_example_id'],
            voter_id=voter_model_id,
            score=0.0,
            critique=f"Automatic failure due to execution error or empty output: {output[:100]}"
        )
        return {"current_votes": [vote]}

    voting_prompt = format_voting_prompt(
        task_description=task['target_task_description'],
        prompt_text=task['prompt_text'],
        input_context=task['input_example_data'],
        output=output
    )

    raw_vote_output = call_llm(
        voter_model_id,
        voting_prompt,
        system_prompt=VOTING_SYSTEM_PROMPT,
        response_format="json"
    )

    try:
        vote_data = parse_json_output(raw_vote_output)

        if not isinstance(vote_data, dict):
            raise ValueError(f"Voter response was valid JSON but not a dictionary. Received Type: {type(vote_data)}")

        if "score" not in vote_data or "critique" not in vote_data:
            raise ValueError("Voter response missing required keys ('score', 'critique').")

        score = float(vote_data.get("score"))
        critique = str(vote_data.get("critique"))

    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Failed to parse or validate vote from {voter_model_id}: {e}")
        score = 0.0
        critique = f"Voter response parsing/validation failed: {e}"

    vote = Vote(
        vote_id=str(uuid.uuid4())[:8],
        execution_result_id=exec_result['execution_id'],
        candidate_id=exec_result['candidate_id'],
        input_example_id=exec_result['input_example_id'],
        voter_id=voter_model_id,
        score=score,
        critique=critique
    )

    return {"current_votes": [vote]}

def aggregate_and_score(state: OptimizationState) -> Dict[str, Any]:
    """
    Step 5: Aggregate votes, normalize scores, update global best/worst examples, track iteration performance (Q1), generate detailed traces (Q2), and synthesize critiques.
    """
    votes = state['current_votes']
    execution_results = state['current_execution_results']
    OPTIMIZER_MODEL = state['optimizer_model']
    all_prompts = state['all_tested_prompts']
    current_iteration = state['current_iteration'] # NEW (Q2)

    global_best_example = state.get('global_best_example')
    global_worst_example = state.get('global_worst_example')
    iteration_score_history = state.get('iteration_best_score_history', []).copy()

    # 1. Filter and Prepare Data
    valid_candidate_ids = set(state.get('current_candidates', {}).keys())
    filtered_votes = []
    seen = set()
    for v in votes or []:
        cid = v.get('candidate_id')
        key = (cid, v.get('input_example_id'), v.get('voter_id'))
        if cid in valid_candidate_ids and key not in seen:
            filtered_votes.append(v)
            seen.add(key)

    if not filtered_votes:
        logger.warning("No votes recorded for current candidates in this iteration.")
        iteration_score_history.append(-1.0)
        
        return {
            "synthesized_critiques": "No valid votes recorded for synthesis.",
            "history": state['history'],
            "best_result": state['best_result'],
            "global_best_example": global_best_example,
            "global_worst_example": global_worst_example,
            "iteration_best_score_history": iteration_score_history,
            "execution_trace_history": [], # NEW (Q2)
        }

    df_votes = pd.DataFrame(filtered_votes)

    # 2. Z-Score Normalization
    def normalize(x):
        x = pd.to_numeric(x, errors='coerce')
        mean = x.mean()
        std = x.std()
        if std == 0 or np.isnan(std) or np.isnan(mean):
            return pd.Series(0.0, index=x.index)
        return (x - mean) / std

    SYSTEM_ERRORS = ["Automatic failure", "Voter response parsing/validation failed"]
    valid_votes_mask = (df_votes['score'] > 0) & (~df_votes['critique'].str.startswith(tuple(SYSTEM_ERRORS)))
    df_votes['normalized_score'] = 0.0
    
    if valid_votes_mask.any():
        normalized_values = df_votes.loc[valid_votes_mask].groupby('voter_id')['score'].transform(normalize)
        df_votes.loc[valid_votes_mask, 'normalized_score'] = normalized_values
        df_votes['normalized_score'] = df_votes['normalized_score'].fillna(0.0)
    else:
        logger.warning("No valid votes (non-system errors and score > 0) in this iteration.")

    # 3. Aggregation
    consensus_scores = df_votes.groupby(['candidate_id', 'input_example_id', 'execution_result_id']).agg(
        avg_normalized_score=('normalized_score', 'mean'),
        avg_raw_score=('score', 'mean'),
        critiques=('critique', list)
    ).reset_index()

    aggregate_scores = consensus_scores.groupby('candidate_id').agg(
        aggregate_score=('avg_normalized_score', 'mean'),
        raw_average_score=('avg_raw_score', 'mean')
    ).reset_index()
    
    # 4. Update GLOBAL Best and Worst Examples AND Generate Traces (Q2)
    
    # (Prepare df_details by merging executions and consensus scores)
    filtered_executions = [e for e in execution_results if e.get('candidate_id') in valid_candidate_ids]
    
    if filtered_executions:
        df_executions = pd.DataFrame(filtered_executions)
        df_executions_subset = df_executions[['execution_id', 'executed_prompt_text', 'output']]
        df_details = pd.merge(consensus_scores, df_executions_subset, left_on='execution_result_id', right_on='execution_id', how='left')
    else:
        df_details = consensus_scores
        df_details['executed_prompt_text'] = '[Execution data missing]'
        df_details['output'] = '[Execution data missing]'

    def format_example(row) -> PerformanceExample:
        candidate_id = row['candidate_id']
        prompt_template = all_prompts.get(candidate_id, {}).get('prompt_text', '[Template Missing]')
        return PerformanceExample(
            candidate_id=candidate_id,
            prompt_template=prompt_template,
            input_example_id=row['input_example_id'],
            consensus_normalized_score=row['avg_normalized_score'],
            consensus_raw_score=row['avg_raw_score'],
            executed_prompt_text=row.get('executed_prompt_text', '[Missing]'),
            output=row.get('output', '[Missing]'),
            critiques=row['critiques']
        )

    if not df_details.empty:
        idx_iter_best = df_details['avg_normalized_score'].fillna(-np.inf).idxmax()
        idx_iter_worst = df_details['avg_normalized_score'].fillna(np.inf).idxmin()
        iter_best_example = format_example(df_details.loc[idx_iter_best])
        iter_worst_example = format_example(df_details.loc[idx_iter_worst])

        if global_best_example is None or iter_best_example['consensus_normalized_score'] > global_best_example['consensus_normalized_score']:
            logger.info(f"New global best example found. Score: {iter_best_example['consensus_normalized_score']:.4f}")
            global_best_example = iter_best_example

        if global_worst_example is None or iter_worst_example['consensus_normalized_score'] < global_worst_example['consensus_normalized_score']:
            logger.info(f"New global worst example found. Score: {iter_worst_example['consensus_normalized_score']:.4f}")
            global_worst_example = iter_worst_example

    # --- NEW (Q2): Generate Execution Traces ---
    execution_traces = []
    # Create a lookup dictionary for votes based on execution_result_id
    # We drop the 'normalized_score' as it's specific to this iteration's normalization context.
    votes_records = df_votes.drop(columns=['normalized_score'], errors='ignore').to_dict(orient='records')
    votes_lookup = {}
    for vote in votes_records:
        exec_id = vote.get('execution_result_id')
        if exec_id not in votes_lookup:
            votes_lookup[exec_id] = []
        votes_lookup[exec_id].append(vote)

    for _, row in df_details.iterrows():
        candidate_id = row['candidate_id']
        execution_id = row['execution_result_id']
        prompt_template = all_prompts.get(candidate_id, {}).get('prompt_text', '[Template Missing]')
        
        # Retrieve the list of votes associated with this execution
        individual_votes = votes_lookup.get(execution_id, [])
        
        trace = ExecutionTrace(
            iteration=current_iteration,
            candidate_id=candidate_id,
            input_example_id=row['input_example_id'],
            prompt_template=prompt_template,
            executed_prompt_text=row.get('executed_prompt_text', '[Missing]'),
            output=row.get('output', '[Missing]'),
            avg_normalized_score=row['avg_normalized_score'],
            avg_raw_score=row['avg_raw_score'],
            votes=individual_votes # Attach the detailed votes
        )
        execution_traces.append(trace)

    # 5. Format Iteration Results (Updated for Q1 Elitism handling)
    iteration_results = []
    # Create a dictionary from the existing history for easy updates (Q1)
    current_history = {item['candidate_id']: item for item in state['history']}

    for _, row in aggregate_scores.iterrows():
        candidate_id = row['candidate_id']
        if candidate_id in all_prompts:
            prompt_text = all_prompts[candidate_id]['prompt_text']
            agg_score = 0.0 if pd.isna(row['aggregate_score']) else row['aggregate_score']
            raw_avg_score = 0.0 if pd.isna(row['raw_average_score']) else row['raw_average_score']
            result = IterationResult(
                candidate_id=candidate_id,
                prompt_text=prompt_text,
                # Use the iteration the prompt was *originally generated*
                iteration=all_prompts[candidate_id].get('iteration', current_iteration),
                aggregate_score=agg_score,
                raw_average_score=raw_avg_score
            )
            iteration_results.append(result)
            
            # Update or add the result in the comprehensive history (Q1)
            # This ensures the Elite prompt's score reflects its performance on the current batch
            current_history[candidate_id] = result

    # Convert the updated history dictionary back to a list and sort it (Q1)
    new_history = list(current_history.values())
    # CRITICAL: Ensure history is sorted descending by score for the next iteration's input
    new_history.sort(key=lambda x: x['aggregate_score'], reverse=True)

    iteration_results.sort(key=lambda x: x['aggregate_score'], reverse=True)
    
    # 5b. Update Iteration Best Score History
    best_score_this_iteration = iteration_results[0]['aggregate_score'] if iteration_results else 0.0
    iteration_score_history.append(best_score_this_iteration)

    # 6. Update Best Result
    # Use the sorted new_history
    best_result = new_history[0] if new_history else state['best_result']

    # 7. Synthesize Critiques
    if iteration_results:
        top_candidates_ids = [r['candidate_id'] for r in iteration_results[:3]]
        valid_critiques_df = df_votes[
            (df_votes['score'] > 0) &
            (df_votes['candidate_id'].isin(top_candidates_ids)) &
            (~df_votes['critique'].str.startswith(tuple(SYSTEM_ERRORS)))
        ]
        valid_critiques = valid_critiques_df['critique'].dropna().unique().tolist()
    else:
        valid_critiques = []

    if valid_critiques:
        synthesis_prompt = format_synthesis_prompt(valid_critiques)
        synthesized_critiques = call_llm(OPTIMIZER_MODEL, synthesis_prompt, temperature=0.3)
        if synthesized_critiques.startswith("LLM_CALL_ERROR"):
            synthesized_critiques = f"Synthesis failed: {synthesized_critiques}"
    else:
        synthesized_critiques = "No valid critiques generated for the top prompts."

    return {
        "history": new_history,
        "best_result": best_result,
        "synthesized_critiques": synthesized_critiques,
        "global_best_example": global_best_example,
        "global_worst_example": global_worst_example,
        "iteration_best_score_history": iteration_score_history,
        "execution_trace_history": execution_traces, # NEW (Q2)
    }
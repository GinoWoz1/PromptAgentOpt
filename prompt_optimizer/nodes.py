# prompt_optimizer/nodes.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
import logging
import json
import re
import os # Ensure os is imported for the debug logs

from sklearn.metrics.pairwise import cosine_similarity

# Import models and templates
from .models import PromptCandidate, ExecutionResult, Vote, IterationResult, ExecutionTask, VotingTask, OptimizationState
from .templates import (format_optimizer_prompt, format_synthesis_prompt,
OPTIMIZER_SYSTEM_PROMPT, format_voting_prompt, VOTING_SYSTEM_PROMPT
)

# Import the LLM Manager (using the standardized name from client.py)
from intelligence_assets.llm.client import llm_manager

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------

def call_llm(model_name: str, prompt: str, system_prompt: Optional[str] = None, response_format="text", **kwargs) -> str:
    """Calls the specified LLM model via the LLMClientManager."""
    try:
        return llm_manager.call_model(model_name=model_name, prompt=prompt, system_prompt=system_prompt, response_format=response_format, **kwargs)
    except Exception as e:
        logger.error(f"Error calling LLM {model_name}: {e}")
        return f"LLM_CALL_ERROR: {e}"

def parse_json_output(output: str) -> Any:
    """Parses the LLM output which is expected to be a JSON string."""
    output = output.strip()
    if output.startswith("LLM_CALL_ERROR"):
        raise ValueError(output)

    # try extraction json from markdown fences if present
    match = re.search(r'```json\s*([\s\S]*?)\s*```', output)

    if match:
        json_str = match.group(1)
    else:
        json_str = output

    try:
        # CRITICAL FIX: Must load json_str, not the original output
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}. Output: {output[:500]}...")
        raise ValueError(f"Failed to parse JSON output.")

# -----------------------------------------------------------------------
# Node Definitions
# -----------------------------------------------------------------------

# Synchronization Barrier (Reduce 1)
def synchronize_executions(state: OptimizationState) -> Dict[str, Any]:
    """
    Acts as a synchronization barrier (Reduce 1).
    It waits for all parallel executions from Map 1 to complete and accumulate
    in the state before allowing the workflow to proceed to Map 2 (Voting).
    """
    num_results = len(state['current_execution_results'])
    logger.info(f"--- Synchronization Point (Reduce 1) ---")
    logger.info(f"Accumulated {num_results} execution results. Proceeding to voting.")
    # This node doesn't need to modify the state, just pass it through.
    return {}


def generate_prompts(state: OptimizationState) -> Dict[str, Any]:
    """
    Step 1: Generate N prompts, ensure novelty, and select the mini-batch.
    """
    iteration = state['current_iteration'] + 1
    N_CANDIDATES = state['N_CANDIDATES']
    OPTIMIZER_MODEL = state['optimizer_model']
    SIMILARITY_THRESHOLD = 0.95
    MAX_RETRIES = 3

    # DEBUG: Log full state details
    logger.info(f"--- Starting Iteration {iteration} --- Model: {OPTIMIZER_MODEL} ---")
    logger.info(f"DEBUG: Task description: {state['target_task_description'][:100]}...")
    logger.info(f"DEBUG: Historical prompts count: {len(state['all_tested_prompts'])}")
    logger.info(f"DEBUG: Synthesized critiques length: {len(state['synthesized_critiques'] or '')}")

    # 1. Prepare input
    historical_prompts = list(state['all_tested_prompts'].values())
    historical_texts = [p['prompt_text'] for p in historical_prompts]
    synthesized_critiques = state['synthesized_critiques']

    # 2. Generate and Validate Prompts (The Novelty Loop)
    new_candidates = {}
    attempts = 0
    generated_texts = []  # Initialize the variable

    while len(new_candidates) < N_CANDIDATES and attempts < MAX_RETRIES:
        attempts += 1
        needed = N_CANDIDATES - len(new_candidates)
        
        # DEBUG: Log attempts
        logger.info(f"DEBUG: Attempt {attempts}/{MAX_RETRIES} to generate {needed} prompts")

        optimizer_prompt = format_optimizer_prompt(
            state['target_task_description'],
            synthesized_critiques,
            historical_texts,
            needed,
            iteration
        )
        
        # DEBUG: Log the optimizer prompt
        logger.info(f"DEBUG: Optimizer prompt length: {len(optimizer_prompt)}")
        logger.info(f"DEBUG: Optimizer prompt snippet: {optimizer_prompt[:300]}...")

        # Call the Optimizer LLM
        logger.info(f"DEBUG: Calling LLM: {OPTIMIZER_MODEL}")
        raw_output = call_llm(
            model_name=OPTIMIZER_MODEL,
            prompt=optimizer_prompt,
            system_prompt=OPTIMIZER_SYSTEM_PROMPT,
            response_format="json",
            temperature=0.7
        )
        
        # DEBUG: Log the raw output
        logger.info(f"DEBUG: Raw LLM output length: {len(raw_output)}")
        logger.info(f"DEBUG: Raw LLM output snippet: {raw_output[:300]}...")
        
        # Check for API errors
        if raw_output.startswith("LLM_CALL_ERROR"):
            logger.error(f"DEBUG: LLM call failed: {raw_output}")
            continue

        try:
            logger.info("DEBUG: Attempting to parse JSON output")
            parsed_output = parse_json_output(raw_output)
            
            # DEBUG: Log the parsed output
            logger.info(f"DEBUG: Parsed output type: {type(parsed_output)}")
            logger.info(f"DEBUG: Parsed output keys: {parsed_output.keys() if isinstance(parsed_output, dict) else 'Not a dict'}")
            
            if isinstance(parsed_output, dict) and "prompts" in parsed_output and isinstance(parsed_output["prompts"], list):
                generated_texts = parsed_output["prompts"]
                logger.info(f"DEBUG: Successfully parsed {len(generated_texts)} prompts")
                # DEBUG: Log sample prompts
                for i, text in enumerate(generated_texts[:2]):
                    logger.info(f"DEBUG: Sample prompt {i+1}: {text[:100]}...")
            else:
                logger.warning(f"DEBUG: Unexpected output format. Output: {raw_output[:200]}")
                generated_texts = []
                continue
        except ValueError as e:
            logger.warning(f"DEBUG: JSON parsing failed: {str(e)}")
            generated_texts = []
            continue

        if not generated_texts:
            logger.warning("DEBUG: No prompts were generated by the LLM")
            continue
            
        logger.info(f"DEBUG: Attempting to get embeddings for {len(generated_texts)} prompts")

        # Get Embeddings
        try:
            generated_embeddings = llm_manager.get_embeddings(generated_texts)
            logger.info(f"DEBUG: Successfully got {len(generated_embeddings)} embeddings")
        except Exception as e:
            logger.error(f"DEBUG: Embedding generation failed: {str(e)}")
            # Check if OpenAI API key is set
            logger.info(f"DEBUG: OPENAI_API_KEY set: {'OPENAI_API_KEY' in os.environ}")
            continue

        # Prepare comparison embeddings
        comparison_embeddings = [p['embedding'] for p in historical_prompts if p.get('embedding')]
        comparison_embeddings.extend([c['embedding'] for c in new_candidates.values()])
        logger.info(f"DEBUG: Comparison embeddings count: {len(comparison_embeddings)}")

        # Track novelty rejections
        rejected_count = 0
        accepted_count = 0

        for text, embedding in zip(generated_texts, generated_embeddings):
            if len(new_candidates) >= N_CANDIDATES:
                break

            # 3. Semantic Similarity Check
            is_novel = True
            if comparison_embeddings:
                try:
                    similarities = cosine_similarity([embedding], comparison_embeddings)[0]
                    max_similarity = np.max(similarities)
                    logger.info(f"DEBUG: Max similarity for prompt: {max_similarity:.4f}")

                    if max_similarity > SIMILARITY_THRESHOLD:
                        logger.warning(f"DEBUG: Rejecting prompt due to high similarity ({max_similarity:.4f})")
                        rejected_count += 1
                        is_novel = False
                except Exception as e:
                    logger.error(f"DEBUG: Similarity check failed: {str(e)}")
                    is_novel = True  # Be permissive on errors

            if is_novel:
                candidate_id = str(uuid.uuid4())[:8]
                candidate = PromptCandidate(
                    candidate_id=candidate_id,
                    prompt_text=text,
                    iteration=iteration,
                    embedding=embedding
                )
                new_candidates[candidate_id] = candidate
                comparison_embeddings.append(embedding)
                accepted_count += 1
                logger.info(f"DEBUG: Accepted new prompt candidate: {text[:100]}...")

        logger.info(f"DEBUG: After attempt {attempts}: {accepted_count} accepted, {rejected_count} rejected")
        logger.info(f"DEBUG: Total candidates so far: {len(new_candidates)}/{N_CANDIDATES}")

    # End of attempts loop
    if not new_candidates:
        logger.error("DEBUG: CRITICAL ERROR - No novel prompts were generated after all attempts")
        logger.error(f"DEBUG: OPTIMIZER_MODEL: {OPTIMIZER_MODEL}")
        logger.error(f"DEBUG: Check API keys and model availability")
        
        # Instead of failing completely, try one last desperate attempt without similarity checking
        logger.info("DEBUG: Making one last attempt without similarity checking")
        try:
            raw_output = call_llm(
                model_name=OPTIMIZER_MODEL,
                prompt="Generate one simple prompt for analyzing Chinese Buddhist text translation. Include placeholders for {chinese_context}, {target_info}, and {english_translation_block}.",
                system_prompt="You are a helpful assistant. Generate a prompt template.",
                response_format="text",
                temperature=0.9
            )
            
            if not raw_output.startswith("LLM_CALL_ERROR"):
                # Just use the raw output as a prompt
                candidate_id = str(uuid.uuid4())[:8]
                candidate = PromptCandidate(
                    candidate_id=candidate_id,
                    prompt_text=raw_output.strip(),
                    iteration=iteration,
                    embedding=[0.0] * 10  # Dummy embedding
                )
                new_candidates[candidate_id] = candidate
                logger.info(f"DEBUG: Created emergency fallback prompt: {raw_output[:100]}...")
            else:
                raise RuntimeError(f"Optimizer LLM failed completely: {raw_output}")
        except Exception as e:
            raise RuntimeError(f"Optimizer LLM failed to generate novel prompts after multiple attempts: {str(e)}")

    # Update the state
    updated_all_tested = state['all_tested_prompts'].copy()
    updated_all_tested.update(new_candidates)

    # Select Mini-Batch for this iteration (Stochastic Sampling)
    dataset = state['input_dataset']
    batch_size = min(state['MINI_BATCH_SIZE'], len(dataset))
    
    logger.info(f"DEBUG: Input dataset size: {len(dataset)}")
    logger.info(f"DEBUG: Mini-batch size: {batch_size}")
    
    if batch_size > 0:
        # Randomly sample indices using numpy for efficiency
        indices = np.random.choice(len(dataset), batch_size, replace=False)
        current_mini_batch = [dataset[i] for i in indices]
        logger.info(f"DEBUG: Selected mini-batch with {len(current_mini_batch)} examples")
    else:
        current_mini_batch = []
        if len(dataset) == 0:
             logger.error("DEBUG: Input dataset is empty")
             raise ValueError("Input dataset cannot be empty")

    return {
        "current_iteration": iteration,
        "current_candidates": new_candidates,
        "all_tested_prompts": updated_all_tested,
        "current_mini_batch": current_mini_batch,
        # Clear accumulators
        "current_execution_results": [],
        "current_votes": [],
    }

def execute_prompt_node(task: ExecutionTask) -> Dict[str, List[ExecutionResult]]:
    """
    Executes the prompt template on a specific input example dynamically.
    Supports complex formatting (attribute/index access) for future adaptability.
    """
    candidate = task['candidate']
    input_example = task['input_example']
    ACTOR_MODEL = task['actor_model']

    # Safely access input data for interpolation and context passing
    # Use .get() for defensive coding in case keys are missing from the input_example structure
    input_data = input_example.get('data', {})
    input_example_id = input_example.get('id', 'N/A')

    logger.debug(f"Executing prompt {candidate['candidate_id']} on input {input_example_id} with {ACTOR_MODEL}")

    # -----------------------------------------------------------------------
    # DYNAMIC INTERPOLATION AND EXECUTION
    # -----------------------------------------------------------------------
    
    # We use a try...except...else block to clearly separate formatting errors from execution.
    try:
        # Use Python's built-in formatting with dictionary unpacking for maximum dynamism.
        # This handles simple placeholders, attribute access, and index access automatically.
        prompt_to_execute = candidate['prompt_text'].format(**input_data)

    except KeyError as e:
        # Handles missing dictionary keys (e.g., the prompt requires {A} but input_data lacks 'A')
        error_msg = f"Template rendering failed. Missing required key/placeholder: {e}."
        logger.error(f"Prompt format error for {candidate['candidate_id']}: {error_msg}")
        output = f"LLM_CALL_ERROR: {error_msg}"
        
    except (IndexError, AttributeError) as e:
        # Handles invalid access on existing objects (e.g., {my_list[99]} or {my_obj.missing_attr})
        error_msg = f"Template rendering failed. Invalid attribute or index access: {e}."
        logger.error(f"Prompt format error for {candidate['candidate_id']}: {error_msg}")
        output = f"LLM_CALL_ERROR: {error_msg}"

    except TypeError as e:
        # Handles type mismatches during formatting (e.g., trying to format a string as a float {value:.2f})
        error_msg = f"Template rendering failed. Type error during formatting: {e}."
        logger.error(f"Prompt format error for {candidate['candidate_id']}: {error_msg}")
        output = f"LLM_CALL_ERROR: {error_msg}"

    except Exception as e:
        # Catch-all for unexpected errors during formatting
        error_msg = f"Unexpected error during template rendering: {e}."
        logger.error(f"Unexpected error for {candidate['candidate_id']}: {error_msg}")
        output = f"LLM_CALL_ERROR: {error_msg}"
        
    else:
        # If formatting succeeds, execute the LLM call.
        # The call_llm helper function handles its own internal errors and returns a string.
        output = call_llm(ACTOR_MODEL, prompt_to_execute)

    # -----------------------------------------------------------------------
    # RESULT PACKAGING
    # -----------------------------------------------------------------------

    # Check if execution failed (either during rendering or LLM call)
    if output.startswith("LLM_CALL_ERROR"):
         logger.warning(f"Execution summary: Failed for candidate {candidate['candidate_id']}. Error: {output[:200]}...")

    # Construct the result
    result = ExecutionResult(
        execution_id=str(uuid.uuid4())[:8],
        candidate_id=candidate['candidate_id'],
        input_example_id=input_example_id,
        
        # The router_to_voting function relies on this field (input_example_data) 
        # to pass context to the voters.
        input_example_data=input_data,
        
        output=output
    )
    
    return {"current_execution_results": [result]}

def vote_on_result_node(task: VotingTask) -> Dict[str, List[Vote]]:
    """Evaluates the execution result using a Voter LLM."""
    voter_model_id = task['voter_id']
    exec_result = task['execution_result']
    output = exec_result['output']

    logger.debug(f"Voting with {voter_model_id} on result {exec_result['execution_id']}")

    # Handle cases where execution already failed
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

    # The voter needs the input context to judge the output effectively
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

    # Parse the vote
    try:
        vote_data = parse_json_output(raw_vote_output)
        score = float(vote_data.get("score"))
        critique = str(vote_data.get("critique"))
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Failed to parse vote from {voter_model_id}: {e}. Output: {raw_vote_output[:200]}")
        score = 0.0
        critique = f"Voter response parsing failed: {e}"

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
    Step 5: Aggregate votes using Z-Score Normalization across the mini-batch.
    """
    votes = state['current_votes']
    OPTIMIZER_MODEL = state['optimizer_model']

    # Filter votes to the current iteration's candidates and dedupe by (candidate_id, input_example_id, voter_id)
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
        return {"synthesized_critiques": "No votes recorded."}

    df = pd.DataFrame(filtered_votes)

    # 1. Z-Score Normalization Logic (Calibrating Voters)
    def normalize(x):
        x = pd.to_numeric(x, errors='coerce')
        mean = x.mean()
        std = x.std()
        if std == 0 or np.isnan(std) or np.isnan(mean):
            return 0.0
        return (x - mean) / std

    valid_votes_mask = (df['score'] > 0)

    if valid_votes_mask.any():
        # Apply normalization per voter
        df.loc[valid_votes_mask, 'normalized_score'] = df[valid_votes_mask].groupby('voter_id')['score'].transform(normalize)
        df['normalized_score'] = df['normalized_score'].fillna(0.0)
    else:
        logger.warning("No valid votes (score > 0) in this iteration.")
        df['normalized_score'] = 0.0

    # 2. Aggregate Scores (Averaging performance across the mini-batch)
    aggregate_scores = df.groupby('candidate_id').agg(
        aggregate_score=('normalized_score', 'mean'),
        raw_average_score=('score', 'mean')
    ).reset_index()

    # 3. Format results
    iteration_results = []
    for _, row in aggregate_scores.iterrows():
        candidate_id = row['candidate_id']
        if candidate_id in state['all_tested_prompts']:
            prompt_text = state['all_tested_prompts'][candidate_id]['prompt_text']

            iteration_results.append(IterationResult(
                candidate_id=candidate_id,
                prompt_text=prompt_text,
                iteration=state['current_iteration'],
                aggregate_score=row['aggregate_score'],
                raw_average_score=row['raw_average_score'],
            ))

    # Sort results
    iteration_results.sort(key=lambda x: x['aggregate_score'], reverse=True)

    # 4. Update History and Best Result
    new_history = state['history'] + iteration_results
    new_history.sort(key=lambda x: x['aggregate_score'], reverse=True)
    best_result = new_history[0] if new_history else state['best_result']

    # 5. Synthesize Critiques
    if iteration_results:
        top_candidates_ids = [r['candidate_id'] for r in iteration_results[:3]]
        valid_critiques_df = df[(df['score'] > 0) & (df['candidate_id'].isin(top_candidates_ids)) & (~df['critique'].str.startswith("Automatic failure"))]
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
        "synthesized_critiques": synthesized_critiques
    }
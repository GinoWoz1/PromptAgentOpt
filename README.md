
# PromptAgentOpt: A LangGraph Framework for Generalized Prompt Optimization

This project implements an advanced, iterative prompt optimization framework built on LangGraph. It is designed to evolve and refine LLM prompts to achieve high performance and, crucially, generalization across diverse inputs.

## Research Perspective on Generalization and User Inputs

To ensure a prompt generalizes across novel, unseen data, we must evaluate its performance across a diverse distribution of inputs during the optimization process itself. Simply providing a "goal" is insufficient as it provides no basis for evaluating performance on varied data.

### The Strategy: Stochastic Mini-Batch Optimization
Drawing inspiration from how machine learning models are trained to generalize, the most robust strategy is Stochastic Mini-Batch Optimization:

- **Input Dataset**: The user must provide a diverse dataset of input examples (`input_dataset`). This dataset is the most critical component, as it represents the distribution of inputs the prompt will encounter in a real-world scenario.

- **Stochastic Sampling (Mini-Batching)**: In each optimization iteration, instead of using a fixed set of test cases, we randomly sample a subset (a mini-batch of size K) from the full `input_dataset`.

- **Parallel Execution (N*K)**: We leverage LangGraph's parallelism to execute every candidate prompt (N) against every input in the mini-batch (K) simultaneously.

- **Parallel Evaluation (N*K*M)**: Every one of the N*K execution results is then evaluated by every voter (M) in the ensemble.

- **Robust Aggregation**: The final score for a prompt candidate in an iteration is its average normalized performance across the K samples in the mini-batch.

This stochastic approach ensures the optimization process rewards prompt strategies that perform consistently well across varied, randomly sampled inputs, directly promoting generalization and preventing the optimizer from overfitting to any single example.

## Required User Inputs
To initiate the process, the user must provide:

- **Task Description (Goal)**: A high-level, natural language description of the objective. This is used to guide both the optimizer and the voter LLMs.

- **Input Dataset (The Generalization Driver)**: A list of diverse input examples. Each example should be a dictionary. The keys in these dictionaries define the placeholders that must be used in the prompt templates (e.g., `[{'user_query': '...', 'document': '...'}, {'user_query': '...', 'document': '...'}]`).

- **Configuration**: Model selections (`optimizer_model`, `actor_model`, `voter_ensemble`) and optimization parameters (`max_iterations`, `N_CANDIDATES`, `MINI_BATCH_SIZE`).

## Code Implementation and Features
The implementation includes several sophisticated techniques:

- **Iterative Evolution (Optimizer LLM)**: Uses a powerful LLM to generate novel prompt candidates by analyzing performance history and synthesized critiques from previous iterations.

- **Semantic Novelty Filtering**: Ensures the optimizer doesn't waste time on redundant ideas by embedding new prompts and filtering out those that are too semantically similar to previously tested ones.

- **LLM-as-a-Judge Ensemble**: Employs a heterogeneous group of LLMs (e.g., GPT, Claude, Gemini) to evaluate prompt performance, reducing the evaluation bias inherent in any single model.

- **Z-Score Normalization**: Calibrates scores from different LLM judges by normalizing their scores within each iteration, providing a statistically robust evaluation metric.

- **Massive Parallelism**: Leverages LangGraph's parallel execution capabilities to simultaneously execute N prompts across K inputs and evaluate them with M voters in each cycle.

## Architecture Diagram

The workflow follows a sophisticated Map-Reduce pattern, iterated over several cycles to progressively refine the prompts:

```
graph TD
    A[Start] --> B(1. Generate N Prompts & <br> Sample Mini-Batch K);
    B -- Semantic Filtering --> B;
    B --> C{2. Router: Dispatch N*K Executions};

    subgraph Map 1: Parallel Execution (N Candidates * K Inputs)
        direction LR
        E1(Exec P1 on I1)
        E2(Exec P1 on I2)
        E3(Exec P2 on I1)
        E4(...)
    end

    C --> Map 1;
    Map 1 --> F{3. Router: Dispatch N*K*M Votes};

    subgraph Map 2: Parallel Voting (N*K Executions * M Voters)
        direction LR
        V1(Voter A on E1)
        V2(Voter B on E1)
        V3(Voter A on E2)
        V4(...)
    end

    F --> Map 2;
    Map 2 --> G(4. Aggregate & Score);

    G -- Z-Score Norm & Mini-Batch Avg --> H{5. Iterate?};
    H -- Synthesize Critiques --> B;
    H -- Max Iterations Reached --> I[End];
```

try:
    import google.generativeai as genai
    from google.generativeai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

# Assuming env.py is correctly implemented in the root
from env import getenv

logger = logging.getLogger(__name__)


class LLMClientManager:
    def __init__(self):
        self._clients = {}
        self._initialize_clients()

    # Fixed typo: _initialive_clients -> _initialize_clients
    def _initialize_clients(self):
        # OpenAI
        if OpenAI:
            try:
                # Fixed typo: OPEN_API_KEY -> OPENAI_API_KEY
                openai_key = getenv("OPENAI_API_KEY", None)
                if openai_key:
                    self._clients["openai"] = OpenAI(api_key=openai_key)
                    logger.info("OpenAI client initialized.")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")

        # Anthropic
        if anthropic:
            try:
                anthropic_key = getenv("ANTHROPIC_API_KEY", None)
                if anthropic_key:
                    self._clients["anthropic"] = anthropic.Anthropic(api_key=anthropic_key)
                    logger.info("Anthropic client initialized")
            except Exception as e:
                # Fixed f-string syntax
                logger.warning(f"Could not initialize Anthropic client: {e} ")

        # Google Gemini
        if genai:
            try:
                google_key = getenv("GOOGLE_API_KEY", None)
                if google_key:
                    genai.configure(api_key=google_key)
                    self._clients["google"] = genai
                    logger.info("Google Gemini API configured")
            except Exception as e:
                logger.warning(f"Could not initialize Google Gemini client: {e}")

    def get_client(self, provider: str):
        client = self._clients.get(provider)
        if not client:
            raise ValueError(f"Client for provider {provider} not initialized or supported.")
        return client

    def call_model(self, model_name: str, prompt: str, system_prompt: Optional[str] = None, response_format: str = "text", **kwargs) -> str:
        """
        A unified interface to call different LLM models.
        """
        if "gpt" in model_name:
            return self._call_openai(model_name, prompt, system_prompt, response_format, **kwargs)
        elif "claude" in model_name:
            return self._call_anthropic(model_name, prompt, system_prompt, response_format, **kwargs)
        elif "gemini" in model_name:
            # Fixed typo: __call_gemini -> _call_gemini
            return self._call_gemini(model_name, prompt, system_prompt, response_format, **kwargs)
        else:
            raise ValueError(f"Unknown or unsupported model name: {model_name}")

    def _call_openai(self, model_name: str, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs) -> str:
        client = self.get_client("openai")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Fixed variable name: response_format_ag -> response_format_arg
        response_format_arg = None
        if response_format == "json":
            response_format_arg = {"type": "json_object"}

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format=response_format_arg,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096)
        )
        return response.choices[0].message.content

    def _call_anthropic(self, model_name: str, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs) -> str:
        client = self.get_client("anthropic")

        if response_format == "json":
            if system_prompt and "JSON" not in system_prompt:
                system_prompt += "\n\nCRITICAL: Respond ONLY in the requested JSON format."
            elif not system_prompt:
                system_prompt = "CRITICAL: Respond ONLY in the requested JSON format."

        # Updated to use Messages API (client.messages.create)
        response = client.messages.create(
            model=model_name,
            system=system_prompt if system_prompt else None,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.0)
        )

        # Updated response parsing
        if response.content and hasattr(response.content[0], 'text'):
            return response.content[0].text.strip()

        return ""

    def _call_gemini(self, model_name: str, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs) -> str:
        if not genai_types:
             raise RuntimeError("Gemini types module missing for configuration.")
             
        client = self.get_client("google")

        # Fixed parameter name: system_instructions -> system_instruction
        model = client.GenerativeModel(model_name, system_instruction=system_prompt)

        config_kwargs = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_output_tokens": kwargs.get("max_tokens", 4096)
        }

        if response_format == "json":
             config_kwargs["response_mime_type"] = "application/json"

        generation_config = genai_types.GenerationConfig(**config_kwargs)

        response = model.generate_content(
            contents=prompt,
            generation_config=generation_config
        )

        try:
            return response.text
        except ValueError as e: # Changed Exception to ValueError for Gemini safety blocks
            logger.warning(f"Gemini response empty or blocked. Error: {e}. Feedback: {response.prompt_feedback}")
            return ""

    def get_embeddings(self, texts: List[str], model="text-embedding-3-small", provider: str = "openai") -> List[List[float]]:
        """
        Get embeddings for a list of texts from the specified provider.
        """
        # Default to openai if provider not specified or unavailable
        if provider != "openai" and provider not in self._clients:
             logger.warning(f"Provider {provider} not available or specified for embeddings. Defaulting to OpenAI.")
             provider = "openai"

        client = self.get_client(provider)
        texts = [text.replace("\n", " ") for text in texts]
        response = client.embeddings.create(input=texts, model=model)
        # ensure embeddings are returned in the same order as input texts
        return [data.embedding for data in sorted(response.data, key=lambda x: x.index)]

# Renamed global instance for clarity and consistency
llm_manager = LLMClientManager()

prompt_optimizer/models.py
(Updated to support the N*K*M mini-batch architecture.)

Python

# prompt_optimizer/models.py
import operator
from typing import Annotated, List, Dict, Any, TypedDict, Optional
import uuid

# -----------------------------------------------------------------------
# Core Data Structures for Generalization
# -----------------------------------------------------------------------

# NEW: Represents an input example from the user's dataset
class InputExample(TypedDict):
    id: str
    data: Dict[str, Any] # The actual data (e.g., {"comment": "..."})

class PromptCandidate(TypedDict):
    """ Represents a single prompt generated by the Optimizer LLM."""
    candidate_id: str
    prompt_text: str # This is now a template (e.g., "Analyze this: {comment}")
    iteration: int
    embedding: List[float]

class ExecutionResult(TypedDict):
    """ The result of running a candidate prompt against a SINGLE input example."""
    execution_id: str
    candidate_id: str
    input_example_id: str # NEW: Track which input this result corresponds to
    output: str

class Vote(TypedDict):
    """ A single evaluation by a voter LLM on a SINGLE execution result."""
    vote_id: str
    voter_id: str
    candidate_id: str
    input_example_id: str # NEW: Track which input this vote corresponds to
    score: float
    critique: str
    execution_result_id: str

class IterationResult(TypedDict):
    # Represents the aggregated performance of a prompt across the mini-batch
    candidate_id: str
    prompt_text: str
    iteration: int
    aggregate_score: float # The average normalized Z-score across the mini-batch
    raw_average_score: float

# -----------------------------------------------------------------------
# Task Definitions (Inputs for parallel nodes)
# -----------------------------------------------------------------------

class ExecutionTask(TypedDict):
    candidate: PromptCandidate
    input_example: InputExample # NEW: The specific input to test against
    actor_model: str

class VotingTask(TypedDict):
    execution_result: ExecutionResult
    voter_id: str
    target_task_description: str
    prompt_text: str
    input_example_data: Dict[str, Any] # NEW: Provide input context to the voter

# -----------------------------------------------------------------------
# Graph State
# -----------------------------------------------------------------------

class OptimizationState(TypedDict):
    """The main state for tracking the optimization workflow"""

    # Configuration
    max_iterations: int
    target_task_description: str
    N_CANDIDATES: int
    MINI_BATCH_SIZE: int # NEW: Configuration for generalization (K)

    # Input Data
    input_dataset: List[InputExample] # NEW: The dataset provided by the user

    # Models Configuration
    optimizer_model: str
    voter_ensemble: List[str]
    actor_model: str

    # Iteration State
    current_iteration: int
    current_candidates: Dict[str, PromptCandidate]
    current_mini_batch: List[InputExample] # NEW: The inputs sampled for this iteration

    # Accumulators (N * K * M parallelism)
    current_execution_results: Annotated[List[ExecutionResult], operator.add]
    current_votes: Annotated[List[Vote], operator.add]

    # History
    all_tested_prompts: Dict[str, PromptCandidate]
    history: List[IterationResult]

    # Synthesized critiques from the previous iteration
    synthesized_critiques: str

    # Best Result
    best_result: Optional[IterationResult]
prompt_optimizer/templates.py
(Updated to encourage placeholders and provide the necessary context during voting.)

Python

# prompt_optimizer/templates.py
from typing import List
import json

# -----------------------------------------------------------------------
# Optimizer LLM Template (The Generator)
# -----------------------------------------------------------------------

# Updated instructions for JSON object format and placeholders
OPTIMIZER_SYSTEM_PROMPT = """
You are an expert prompt optimization agent. Your goal is to iteratively refine prompts based on performance evaluations and critiques.

CRITICAL INSTRUCTIONS:
1. Exploration and Exploitation: Balance refining known good strategies with exploring novel approaches.
2. Inter-Iteration Novelty: You MUST NOT generate prompts semantically identical to those in the <history> section.
3. Intra-Batch Diversity: Prompts generated in this batch MUST be diverse from each other.
4. Input Placeholders: Prompts must be designed as templates to accept input data via f-string style placeholders (e.g., "Analyze this text: {input_text}"). Ensure you use appropriate placeholders based on the task description and input data structure.
5. Output Format: You MUST respond ONLY with a JSON object containing a key "prompts", which is a list of strings.
"""

def format_optimizer_prompt(task_desc: str, synthesized_critiques: str, history_prompts: List[str], num_candidates: int, iteration: int) -> str:
    """Formats the prompt for the Optimizer LLM."""

    # Context Management
    MAX_HISTORY_IN_PROMPT = 50
    truncated_history = history_prompts[-MAX_HISTORY_IN_PROMPT:]
    history_section = "\n".join([f"- {p}" for p in truncated_history])
    if len(history_prompts) > MAX_HISTORY_IN_PROMPT:
        history_section = f"... (and {len(history_prompts) - MAX_HISTORY_IN_PROMPT} earlier prompts truncated)\n" + history_section

    prompt = f"""
<target_task>
{task_desc}
</target_task>

<critiques_to_address>
Analyze these insights from the previous iteration. Generate prompts that specifically address these failure modes.
{synthesized_critiques if synthesized_critiques else "No critiques available (First iteration). Focus on diversity and ensure correct input placeholders (e.g., {placeholder_name}) are used."}
</critiques_to_address>

<history>
IMPORTANT: Avoid generating prompts similar to these previous attempts.
{history_section if history_section else "No history yet."}
</history>

<instructions>
Generate {num_candidates} new prompt candidates for Iteration {iteration}.
Ensure diversity and novelty as per the system instructions. Remember to include necessary input placeholders.

Respond ONLY with a JSON object in the following format:
{{"prompts": ["Prompt template 1...", "Prompt template 2..."]}}
</instructions>
"""
    return prompt

# -----------------------------------------------------------------------
# Critique Synthesizer Template
# -----------------------------------------------------------------------

def format_synthesis_prompt(critiques: List[str]) -> str:
    """Formats the prompt to synthesize critiques into actionable insights."""
    critiques_formatted = "\n\n".join([f"<critique>{c}</critique>" for c in critiques])
    return f"""
You are an expert analyst. The following critiques were provided by various judges on the performance of the top prompts in the last optimization iteration.

<critiques>
{critiques_formatted}
</critiques>

<instructions>
Synthesize these critiques into a concise, actionable summary. Identify the main failure modes, recurring themes, and provide concrete recommendations for how the prompts should be improved in the next iteration. Focus on overarching strategy.
</instructions>
"""

# -----------------------------------------------------------------------
# Voting Template (LLM-as-a-Judge)
# -----------------------------------------------------------------------

VOTING_SYSTEM_PROMPT = """
You are an impartial, expert judge evaluating the performance of an AI assistant on a specific task instance.
Your evaluation must be objective and based solely on the provided criteria.

<instructions>
1. Review the <task_description>.
2. Analyze the <prompt_used> (the strategy/template).
3. Examine the <input_context> (the specific data provided to the AI via the template).
4. Evaluate the <output_generated> (the AI's response).
5. Determine how well the output fulfills the task description given the specific input context.
6. Provide a score from 1 (Terrible) to 10 (Excellent).
7. Provide a concise critique explaining your rationale, focusing on why the output was good or bad for this specific input.

You MUST respond ONLY in a JSON object format:
{"score": float, "critique": "string"}
</instructions>
"""

# Updated signature to include input_context
def format_voting_prompt(task_description: str, prompt_text: str, input_context: Dict[str, Any], output: str) -> str:
    """Formats the user prompt for the Voter LLM."""
    
    # Format context dictionary for readability
    context_str = json.dumps(input_context, indent=2, ensure_ascii=False)

    return f"""
<task_description>
{task_description}
</task_description>

<prompt_used>
{prompt_text}
</prompt_used>

<input_context>
{context_str}
</input_context>

<output_generated>
{output}
</output_generated>

Please provide your evaluation in the required JSON format.
"""
prompt_optimizer/nodes.py
(Implements the mini-batch logic, template rendering, and fixes the critical JSON parsing bug.)

Python

# prompt_optimizer/nodes.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
import logging
import json
import re

from sklearn.metrics.pairwise import cosine_similarity

# Import models and templates
from .models import (
    PromptCandidate, ExecutionResult, Vote, IterationResult, ExecutionTask,
    VotingTask, OptimizationState
)
from .templates import (
    format_optimizer_prompt, format_synthesis_prompt, OPTIMIZER_SYSTEM_PROMPT,
    format_voting_prompt, VOTING_SYSTEM_PROMPT
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
        return llm_manager.call_model(
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=response_format,
            **kwargs
        )
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

def generate_prompts(state: OptimizationState) -> Dict[str, Any]:
    """
    Step 1: Generate N prompts, ensure novelty, and select the mini-batch.
    """
    iteration = state['current_iteration'] + 1
    N_CANDIDATES = state['N_CANDIDATES']
    OPTIMIZER_MODEL = state['optimizer_model']
    SIMILARITY_THRESHOLD = 0.95
    MAX_RETRIES = 3

    logger.info(f"--- Starting Iteration {iteration} --- Model: {OPTIMIZER_MODEL} ---")

    # 1. Prepare input
    historical_prompts = list(state['all_tested_prompts'].values())
    historical_texts = [p['prompt_text'] for p in historical_prompts]
    synthesized_critiques = state['synthesized_critiques']

    # 2. Generate and Validate Prompts (The Novelty Loop)
    new_candidates = {}
    attempts = 0

    while len(new_candidates) < N_CANDIDATES and attempts < MAX_RETRIES:
        attempts += 1
        needed = N_CANDIDATES - len(new_candidates)

        optimizer_prompt = format_optimizer_prompt(
            state['target_task_description'],
            synthesized_critiques,
            historical_texts,
            needed,
            iteration
        )

        # Call the Optimizer LLM
        raw_output = call_llm(
            model_name=OPTIMIZER_MODEL,
            prompt=optimizer_prompt,
            system_prompt=OPTIMIZER_SYSTEM_PROMPT,
            response_format="json",
            temperature=0.7
        )

        try:
            parsed_output = parse_json_output(raw_output)
            if isinstance(parsed_output, dict) and "prompts" in parsed_output and isinstance(parsed_output["prompts"], list):
                generated_texts = parsed_output["prompts"]
            else:
                logger.warning(f"Optimizer output format unexpected (Attempt {attempts}). Output: {raw_output[:200]}")
                continue
        except ValueError:
             logger.warning(f"Failed to parse optimizer output (Attempt {attempts}).")
             continue

        if not generated_texts: continue

        # Get Embeddings
        try:
            generated_embeddings = llm_manager.get_embeddings(generated_texts)
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            continue

        # Prepare comparison embeddings
        comparison_embeddings = [p['embedding'] for p in historical_prompts if p.get('embedding')]
        comparison_embeddings.extend([c['embedding'] for c in new_candidates.values()])

        for text, embedding in zip(generated_texts, generated_embeddings):
            if len(new_candidates) >= N_CANDIDATES:
                break

            # 3. Semantic Similarity Check
            is_novel = True
            if comparison_embeddings:
                try:
                    similarities = cosine_similarity([embedding], comparison_embeddings)[0]
                    max_similarity = np.max(similarities)

                    if max_similarity > SIMILARITY_THRESHOLD:
                        logger.warning(f"Rejecting prompt due to high similarity ({max_similarity:.2f}).")
                        is_novel = False
                except Exception as e:
                    logger.error(f"Error during similarity check: {e}. Skipping check.")

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

    if not new_candidates:
        raise RuntimeError("Optimizer LLM failed to generate novel prompts after multiple attempts.")

    # Update the state
    updated_all_tested = state['all_tested_prompts'].copy()
    updated_all_tested.update(new_candidates)

    # NEW: Select Mini-Batch for this iteration (Stochastic Sampling)
    dataset = state['input_dataset']
    batch_size = min(state['MINI_BATCH_SIZE'], len(dataset))
    if batch_size > 0:
        # Randomly sample indices using numpy for efficiency
        indices = np.random.choice(len(dataset), batch_size, replace=False)
        current_mini_batch = [dataset[i] for i in indices]
    else:
        current_mini_batch = []
        if len(dataset) == 0:
             logger.error("Input dataset is empty.")
             raise ValueError("Input dataset cannot be empty.")


    return {
        "current_iteration": iteration,
        "current_candidates": new_candidates,
        "all_tested_prompts": updated_all_tested,
        "current_mini_batch": current_mini_batch,
        # Clear accumulators
        "current_execution_results": [],
        "current_votes": [],
    }

def execute_prompt_node(task: ExecutionTask) -> ExecutionResult:
    """Executes the prompt template on a specific input example."""
    candidate = task['candidate']
    input_example = task['input_example']
    ACTOR_MODEL = task['actor_model']

    logger.debug(f"Executing prompt {candidate['candidate_id']} on input {input_example['id']} with {ACTOR_MODEL}")

    # NEW: Format the prompt template with the input data
    # This assumes the prompt uses f-string compatible placeholders (e.g., {comment})
    try:
        prompt_to_execute = candidate['prompt_text'].format(**input_example['data'])
    except KeyError as e:
        logger.error(f"Prompt format error: Missing key {e} in input data for prompt {candidate['candidate_id']}")
        # Handle error gracefully
        output = f"LLM_CALL_ERROR: Template rendering failed. Missing placeholder {e}."
    except Exception as e:
        logger.error(f"Unexpected template rendering error: {e}")
        output = f"LLM_CALL_ERROR: Template rendering failed. {e}."
    else:
         # Execute the task if rendering succeeded
        output = call_llm(ACTOR_MODEL, prompt_to_execute)

    # Check if execution failed
    if output.startswith("LLM_CALL_ERROR"):
         logger.warning(f"Execution failed for candidate {candidate['candidate_id']}: {output}")

    return ExecutionResult(
        execution_id=str(uuid.uuid4())[:8],
        candidate_id=candidate['candidate_id'],
        input_example_id=input_example['id'],
        output=output
    )

def vote_on_result_node(task: VotingTask) -> Vote:
    """Evaluates the execution result using a Voter LLM."""
    voter_model_id = task['voter_id']
    exec_result = task['execution_result']
    output = exec_result['output']

    logger.debug(f"Voting with {voter_model_id} on result {exec_result['execution_id']}")

    # Handle cases where execution already failed
    if output.startswith("LLM_CALL_ERROR") or output.strip() == "":
        return Vote(
            vote_id=str(uuid.uuid4())[:8],
            execution_result_id=exec_result['execution_id'],
            candidate_id=exec_result['candidate_id'],
            input_example_id=exec_result['input_example_id'],
            voter_id=voter_model_id,
            score=0.0,
            critique=f"Automatic failure due to execution error or empty output: {output[:100]}"
        )

    # NEW: The voter needs the input context to judge the output effectively
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

    return Vote(
        vote_id=str(uuid.uuid4())[:8],
        execution_result_id=exec_result['execution_id'],
        candidate_id=exec_result['candidate_id'],
        input_example_id=exec_result['input_example_id'],
        voter_id=voter_model_id,
        score=score,
        critique=critique
    )

def aggregate_and_score(state: OptimizationState) -> Dict[str, Any]:
    """
    Step 5: Aggregate votes using Z-Score Normalization across the mini-batch.
    """
    votes = state['current_votes']
    OPTIMIZER_MODEL = state['optimizer_model']

    if not votes:
        logger.warning("No votes recorded in this iteration.")
        return {"synthesized_critiques": "No votes recorded."}

    df = pd.DataFrame(votes)

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
    # We group by candidate_id to see how each prompt did across all inputs it faced.
    aggregate_scores = df.groupby('candidate_id').agg(
        aggregate_score=('normalized_score', 'mean'),
        raw_average_score=('score', 'mean')
    ).reset_index()

    # 3. Format results
    iteration_results = []
    for _, row in aggregate_scores.iterrows():
        candidate_id = row['candidate_id']
        # Ensure the candidate exists in the history
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
    # Focus critiques on the best performing prompts of this iteration to guide the next step
    if iteration_results:
        # Analyze top 3 prompts
        top_candidates_ids = [r['candidate_id'] for r in iteration_results[:3]]
        valid_critiques_df = df[(df['score'] > 0) & (df['candidate_id'].isin(top_candidates_ids)) & (~df['critique'].str.startswith("Automatic failure"))]
        # Get unique critiques related to the top performers
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
prompt_optimizer/workflow.py
(Updated the routers to implement the N*K*M parallelism.)

Python

# prompt_optimizer/workflow.py
import logging
from typing import List
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import Send

# Import the State definition
from .models import OptimizationState, ExecutionTask, VotingTask

# Import the nodes
from .nodes import (
    generate_prompts,
    execute_prompt_node,
    vote_on_result_node,
    aggregate_and_score
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Routers (Map/Reduce and Loop Control)
# -----------------------------------------------------------------------

def router_to_execution(state: OptimizationState) -> List[Send]:
    """
    Dispatcher for Map 1: Execute all candidates across the mini-batch.
    Parallelism: N_CANDIDATES * MINI_BATCH_SIZE
    """
    ACTOR_MODEL = state["actor_model"]
    # The mini-batch was selected during the generate_prompts node
    mini_batch = state["current_mini_batch"]
    sends = []

    if not mini_batch:
        # If the batch is empty, we skip execution.
        logger.warning("Mini-batch is empty. Skipping execution dispatch.")
        return []

    # N * K dispatch
    for candidate in state["current_candidates"].values():
        for input_example in mini_batch:
            sends.append(
                Send(
                    node_name="execute_prompt",
                    data=ExecutionTask(
                        candidate=candidate,
                        input_example=input_example,
                        actor_model=ACTOR_MODEL
                    )
                )
            )
    logger.info(f"Dispatching {len(sends)} execution tasks.")
    return sends

def router_to_voting(state: OptimizationState) -> List[Send]:
    """
    Dispatcher for Map 2: Evaluate all execution results.
    Parallelism: (N_CANDIDATES * MINI_BATCH_SIZE) * N_VOTERS
    """
    sends = []
    task_description = state["target_task_description"]
    all_prompts = state["all_tested_prompts"]
    
    # Create a lookup map for input example data to pass context to the voter
    input_lookup = {item['id']: item['data'] for item in state["current_mini_batch"]}

    # LangGraph waits for all executions (Reduce 1) before running this router.
    for result in state["current_execution_results"]:
        # Retrieve context needed for the voter
        prompt_text = all_prompts.get(result['candidate_id'], {}).get('prompt_text', '[Missing]')
        input_data = input_lookup.get(result['input_example_id'], {})

        for voter_id in state["voter_ensemble"]:
            task_data = VotingTask(
                execution_result=result,
                voter_id=voter_id,
                target_task_description=task_description,
                prompt_text=prompt_text,
                input_example_data=input_data
            )
            sends.append(Send(node_name="vote_on_result", data=task_data))
            
    logger.info(f"Dispatching {len(sends)} voting tasks.")
    return sends

def iteration_router(state: OptimizationState) -> str:
    """Step 6: Loop control."""
    if state["current_iteration"] >= state["max_iterations"]:
        logger.info(f"Reached max iterations ({state['max_iterations']}). Finishing.")
        return "finish"

    if state.get('best_result'):
        score = state['best_result']['aggregate_score']
        logger.info(f"Iteration {state['current_iteration']} complete. Best score so far: {score:.4f}")
    return "iterate"

# -----------------------------------------------------------------------
# Graph Construction
# -----------------------------------------------------------------------

def compile_optimizer_graph():
    """Compiles the optimization workflow into a LangGraph executable."""
    workflow = StateGraph(OptimizationState)

    # Add Nodes
    workflow.add_node("generate_prompts", generate_prompts)
    workflow.add_node("aggregate_and_score", aggregate_and_score)
    workflow.add_node("execute_prompt", execute_prompt_node)
    workflow.add_node("vote_on_result", vote_on_result_node)

    # Define Edges
    workflow.add_edge(START, "generate_prompts")

    # Map 1: Generation -> Execution (Across Mini-Batch)
    workflow.add_conditional_edges(
        "generate_prompts",
        router_to_execution,
        # This special edge ensures that if the router returns an empty list (no tasks),
        # the graph transitions directly to the aggregation node instead of getting stuck.
        {"__end__": "aggregate_and_score"}
    )


    # Map 2: Execution -> Voting
    workflow.add_conditional_edges("execute_prompt", router_to_voting)

    # Reduce 2: Voting -> Aggregation
    workflow.add_edge("vote_on_result", "aggregate_and_score")

    # The Loop
    workflow.add_conditional_edges(
        "aggregate_and_score",
        iteration_router,
        {"iterate": "generate_prompts", "finish": END}
    )

    # Compile the graph
    app = workflow.compile()
    return app
</details>

4. README.md
Here is a robust README file for your project.

PromptAgentOpt: A LangGraph Framework for Generalized Prompt Optimization
This project implements an advanced, iterative prompt optimization framework built on LangGraph. It is designed to evolve and refine LLM prompts to achieve high performance and, crucially, generalization across diverse inputs.

Overview
Traditional prompt optimization often overfits prompts to a small set of specific examples, making them brittle in production. PromptAgentOpt addresses this by employing several sophisticated techniques inspired by optimization theory and machine learning best practices:

Iterative Evolution (Optimizer LLM): Uses a powerful LLM (the "Optimizer") to generate novel prompt candidates by analyzing performance history and synthesized critiques from previous iterations.

Semantic Novelty Filtering: Ensures the optimizer doesn't waste time on redundant ideas by embedding new prompts and filtering out those that are too semantically similar to previously tested ones.

LLM-as-a-Judge Ensemble: Employs a heterogeneous group of LLMs (e.g., GPT, Claude, Gemini) to evaluate prompt performance. This "ensemble" approach reduces the evaluation bias inherent in any single model.

Z-Score Normalization: Calibrates scores from different LLM judges (who may have different scoring tendencies) by normalizing their scores within each iteration. This provides a statistically robust evaluation metric.

Stochastic Mini-Batch Optimization (The Key to Generalization): To ensure prompts generalize well, they are tested against a randomly sampled mini-batch (K) of input data in each iteration. This prevents overfitting by rewarding prompts that have a high average performance across diverse inputs.

Massive Parallelism (LangGraph): The architecture leverages LangGraph's powerful parallel execution capabilities (N*K*M parallelism) to simultaneously execute N prompts across K inputs and evaluate them with M voters in each cycle.

Architecture Diagram
The workflow follows a sophisticated Map-Reduce pattern, iterated over several cycles to progressively refine the prompts.

Code snippet

graph TD
    A[Start] --> B(1. Generate N Prompts & <br> Sample Mini-Batch K);
    B -- Semantic Filtering --> B;
    B --> C{2. Router: Dispatch N*K Executions};

    subgraph Map 1: Parallel Execution (N Candidates * K Inputs)
        direction LR
        E1(Exec P1 on I1)
        E2(Exec P1 on I2)
        E3(Exec P2 on I1)
        E4(...)
    end

    C --> Map 1;
    Map 1 --> F{3. Router: Dispatch N*K*M Votes};

    subgraph Map 2: Parallel Voting (N*K Executions * M Voters)
        direction LR
        V1(Voter A on E1)
        V2(Voter B on E1)
        V3(Voter A on E2)
        V4(...)
    end

    F --> Map 2;
    Map 2 --> G(4. Aggregate & Score);

    G -- Z-Score Norm & Mini-Batch Avg --> H{5. Iterate?};
    H -- Synthesize Critiques --> B;
    H -- Max Iterations Reached --> I[End];

## Key Files and Their Roles

The project is structured for modularity and clarity:

| File | Role |
|------|------|
| `prompt_optimizer/workflow.py` | The core LangGraph definition. Defines the graph structure, nodes, edges, and routers (dispatchers for parallel tasks). |
| `prompt_optimizer/nodes.py` | Contains the business logic for each node in the graph (Generation, Execution, Voting, Aggregation). |
| `prompt_optimizer/models.py` | Defines the data schemas (TypedDicts) for the graph's OptimizationState and all auxiliary data structures (Tasks, Votes, Results). |
| `prompt_optimizer/templates.py` | Stores the prompt templates used by the Optimizer LLM and the Voter LLMs. |
| `intelligence_assets/llm/client.py` | A unified client manager for interacting with different LLM providers (OpenAI, Anthropic, Google). |
| `env.py` | A simple utility for securely loading environment variables from a .env file. |
## How to Use

### Prerequisites

- Python 3.10+
- API keys for the LLM providers you intend to use
- Required Python packages

### 1. Setup

Install dependencies:

```bash
pip install "langchain-core>=0.2.2" "langgraph>=0.1.0" pandas numpy scikit-learn python-dotenv openai anthropic google-generativeai
```

Configure Environment Variables:
Create a `.env` file in the root directory and add your API keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

### 2. Running an Optimization
To run the process, create a script (`run_optimization.py`) to compile the graph and invoke it with your specific task and data:

```python
# run_optimization.py
import logging
import uuid
import json
from prompt_optimizer.workflow import compile_optimizer_graph
from prompt_optimizer.models import OptimizationState, InputExample

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. DEFINE THE GOAL
    # Describe the task and specify the placeholders for your input data.
    TASK_DESCRIPTION = """
    Analyze the sentiment of the provided movie review snippet and classify it as Positive, Negative, or Neutral.
    The prompt must accept the input via the placeholder {review}.
    """

    # 2. PROVIDE THE INPUT DATASET (Crucial for Generalization)
    # The keys in the dictionaries (e.g., "review") MUST match the placeholders in your task description.
    raw_dataset = [
        {"review": "Absolutely breathtaking! A masterpiece of cinema."},
        {"review": "I was bored to tears. The plot was predictable and slow."},
        {"review": "It was okay. Not great, not terrible. The acting was decent."},
        {"review": "A flawed gem. The visuals are stunning, but the ending felt rushed."},
        {"review": "I don't understand the hype. It felt very derivative."},
        {"review": "The cinematography was interesting and well-executed."},
        {"review": "A complete waste of time and money. Avoid at all costs."},
        {"review": "Highly recommended for fans of the genre!"},
        {"review": "The film just exists, it doesn't try to do anything special."},
        {"review": "An unforgettable experience that will stay with you for days."}
    ]
    
    # Format the dataset into the required InputExample structure
    input_dataset = [
        InputExample(id=str(uuid.uuid4())[:6], data=item) for item in raw_dataset
    ]

    # 3. CONFIGURE THE OPTIMIZATION RUN
    initial_state = OptimizationState(
        # Core Configuration
        max_iterations=3,          # Number of optimization cycles
        N_CANDIDATES=5,            # Prompts to generate per iteration (N)
        MINI_BATCH_SIZE=4,         # Inputs to test against per iteration (K)
        target_task_description=TASK_DESCRIPTION,
        input_dataset=input_dataset,

        # Model Configuration
        optimizer_model="claude-3-5-sonnet-20240620", 
        actor_model="gpt-3.5-turbo", 
        voter_ensemble=[
            "gpt-4o-mini",
            "claude-3-haiku-20240307",
            # "gemini-1.5-flash-latest" # Add more voters for robustness
        ],

        # --- Boilerplate initial state (do not change) ---
        current_iteration=0,
        current_candidates={},
        current_mini_batch=[],
        current_execution_results=[],
        current_votes=[],
        history=[],
        all_tested_prompts={},
        best_result=None,
        synthesized_critiques=""
    )

    # 4. COMPILE AND RUN THE GRAPH
    print("Compiling optimizer graph...")
    app = compile_optimizer_graph()

    print("🚀 Starting optimization workflow...")
    try:
        # Use invoke() to run the full process and get the final state.
        # Use stream() for real-time updates of each step.
        final_state = app.invoke(initial_state, {"recursion_limit": 150})

        print("\n--- ✅ Optimization Finished ---")
        best_result = final_state.get('best_result')
        if best_result:
            print(f"\n🏆 Best Aggregate Score (Normalized): {best_result['aggregate_score']:.4f}")
            print(f"   Best Raw Average Score: {best_result['raw_average_score']:.4f}")
            print("\n📋 Best Prompt Template Found:")
            print("---------------------------------")
            print(best_result['prompt_text'])
            print("---------------------------------")
            
            # Save the full history for analysis
            with open("optimization_history.json", "w") as f:
                json.dump(final_state['history'], f, indent=2)
            print("\nFull optimization history saved to optimization_history.json")

        else:
            print("Optimization finished without finding a best result. Check logs for errors.")

    except Exception as e:
         logging.error("An unexpected error occurred during execution:", exc_info=True)

if __name__ == "__main__":
    main()
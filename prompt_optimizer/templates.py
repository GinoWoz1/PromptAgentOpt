# prompt_optimizer/templates.py

from typing import List, Dict, Any 
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
    MAX_HISTORY_IN_PROMPT = 15
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
    context_str = json.dumps(input_context, indent=2)

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
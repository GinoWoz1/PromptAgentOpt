# idea_one_project/prompt_optimization/templates.py

from typing import List

# -----------------------------------------------------------------------
# Optimizer LLM Template (The Generator)
# -----------------------------------------------------------------------

OPTIMIZER_SYSTEM_PROMPT = """
You are an expert prompt optimization agent. Your goal is to iteratively refine prompts based on performance evaluations and critiques.

CRITICAL INSTRUCTIONS:
1. Exploration and Exploitation: Balance refining known good strategies (addressing critiques) with exploring novel approaches.
2. Inter-Iteration Novelty: You MUST NOT generate prompts semantically identical to those in the <history> section.
3. Intra-Batch Diversity: Prompts generated in this batch MUST be diverse from each other (e.g., explore CoT, Few-Shot, Personas, different structures).
"""

def format_optimizer_prompt(task_desc: str, synthesized_critiques: str, history_prompts: List[str], num_candidates: int, iteration: int) -> str:
    """Formats the prompt for the Optimizer LLM."""

    # Context Management: Truncate history if necessary to fit context limits.
    # The deterministic novelty check happens via embeddings in the workflow.
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
{synthesized_critiques if synthesized_critiques else "No critiques available (First iteration). Focus on diversity."}
</critiques_to_address>

<history>
IMPORTANT: Avoid generating prompts similar to these previous attempts.
{history_section if history_section else "No history yet."}
</history>

<instructions>
Generate {num_candidates} new prompt candidates for Iteration {iteration}.
Ensure diversity and novelty as per the system instructions.

Respond ONLY with a JSON list of strings, where each string is a new prompt candidate.
Example: ["Prompt candidate 1 text...", "Prompt candidate 2 text..."]
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
You are an expert analyst. The following critiques were provided by various judges on the performance of different prompts in an optimization process.

<critiques>
{critiques_formatted}
</critiques>

<instructions>
Synthesize these critiques into a concise, actionable summary. Identify the main failure modes, recurring themes, and provide concrete recommendations for how the prompts should be improved in the next iteration. Focus on overarching strategy rather than specific wording examples.
</instructions>
"""


# -----------------------------------------------------------------------
# Voting Template (LLM-as-a-Judge)
# -----------------------------------------------------------------------

VOTING_SYSTEM_PROMPT = """
You are an impartial, expert judge evaluating the performance of an AI assistant on a specific task.
Your evaluation must be objective and based solely on the provided criteria.

<instructions>
1. Review the <task_description>, the <prompt_used>, and the <output_generated>.
2. Evaluate how well the output fulfills the task description and the intent of the prompt.
3. Consider accuracy, completeness, clarity, and adherence to constraints.
4. Provide a score from 1 (Terrible) to 10 (Excellent).
5. Provide a concise critique explaining your rationale.

You MUST respond ONLY in a JSON object format:
{"score": float, "critique": "string"}
</instructions>
"""

def format_voting_prompt(task_description: str, prompt_text: str, output: str) -> str:
    """Formats the user prompt for the Voter LLM."""
    return f"""
<task_description>
{task_description}
</task_description>

<prompt_used>
{prompt_text}
</prompt_used>

<output_generated>
{output}
</output_generated>

Please provide your evaluation in the required JSON format.
"""
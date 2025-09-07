from typing import List, Dict, Any
import json

# -----------------------------------------------------------------------
# Optimizer LLM Template (The Generator)
# -----------------------------------------------------------------------

# Updated instructions (Q1)
OPTIMIZER_SYSTEM_PROMPT = """
You are an expert prompt optimization agent. Your goal is to iteratively refine prompts based on performance evaluations and critiques.

CRITICAL INSTRUCTIONS:
1. Exploration and Exploitation: Analyze the <performance_history> and <critiques_to_address>. Balance refining high-performing strategies (exploitation) with exploring novel approaches (exploration).
2. Iterative Improvement: Focus on generating prompts that are likely to score higher than the current best performer.
3. Input Placeholders (CRITICAL): Prompts must be designed as templates. You MUST use Dollar-style placeholders (e.g., "Analyze this text: $input_text"). DO NOT use f-string style (e.g., {input_text}).
4. Diversity: Prompts generated in this batch should be diverse from each other, unless you are focusing on minor variations of a highly successful strategy.
5. Output Format: You MUST respond ONLY with a JSON object containing a key "prompts", which is a list of strings.
"""

# Updated signature (Q1)
def format_optimizer_prompt(task_desc: str, synthesized_critiques: str, performance_history: List[Dict[str, Any]], num_candidates: int, iteration: int) -> str:
    """Formats the prompt for the Optimizer LLM."""
    
    # Context Management: Focus on the top performers (Q1)
    MAX_HISTORY_IN_PROMPT = 10
    # History is expected to be sorted by score descending
    top_performers = performance_history[:MAX_HISTORY_IN_PROMPT]

    history_section = ""
    if top_performers:
        for item in top_performers:
            # Using aggregate_score (normalized Z-score) for relative comparison
            score = item.get('aggregate_score', 0.0)
            prompt_text = item.get('prompt_text', '[Missing]')
            history_section += f"Score: {score:.4f}\nPrompt: {prompt_text}\n---\n"
    else:
        history_section = "No history yet."

    # Format the prompt with $ placeholders
    prompt = f"""
<target_task>
{task_desc}
</target_task>

<performance_history>
Review the scores and structures of the top-performing prompts found so far. Aim to beat these scores.
{history_section}
</performance_history>

<critiques_to_address>
Analyze these insights from the previous iteration. Use them to modify the strategies seen in the <performance_history> or create new ones that address these failure modes.
{synthesized_critiques if synthesized_critiques else "No critiques available (First iteration). Focus on generating diverse initial strategies and ensure correct input placeholders (e.g., $placeholder_name) are used."}
</critiques_to_address>

<instructions>
Generate {num_candidates} new prompt candidates for Iteration {iteration}.
Ensure diversity and novelty as per the system instructions. Remember to include necessary input placeholders using the required Dollar-style syntax ($).

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
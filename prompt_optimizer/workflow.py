# placeholder for workflow related information
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import uuid
import logging
import json

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import Send 

from sklearn.metrics.pairwise import cosine_similarity
from .models import (

    OptimizationState, PromptCandidtate, ExecutionResult, Vote, IterationResult, 
    ExecutionTask, VotingTask
)

from .templates import format_optimizer_prompt, format_synthesis_prompt





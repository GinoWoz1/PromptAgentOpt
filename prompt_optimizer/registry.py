# prompt_optimizer/registry.py

import logging
from langgraph.graph.state import CompiledStateGraph
# Import the function that defines the graph structure using relative import
from .workflow import build_optimizer_graph

logger = logging.getLogger(__name__)

class WorkflowRegistry:
    """Registry and compiler for optimization workflows."""

    def __init__(self):
        self._compiled_workflows: dict[str, CompiledStateGraph] = {}
        self._register_workflows()

    def _register_workflows(self):
        """Define and compile the available workflows."""
        logger.info("Initializing Workflow Registry...")
        try:
            # 1. Build the graph definition
            logger.info("Building optimization graph structure...")
            optimizer_graph = build_optimizer_graph()

            # 2. Compile it
            logger.info("Compiling optimization graph...")
            compiled_optimizer = optimizer_graph.compile()

            # 3. Register it with a specific name/version
            self.register("prompt_optimizer_v1", compiled_optimizer)
            logger.info("Workflow 'prompt_optimizer_v1' compiled and registered successfully.")

        except Exception as e:
            # Log compilation failures. The workflow will not be available if this fails.
            logger.error(f"Failed to compile optimization workflow during registry initialization: {e}", exc_info=True)

    def register(self, name: str, workflow: CompiledStateGraph):
        """Register a compiled workflow."""
        self._compiled_workflows[name] = workflow

    def get(self, name: str) -> CompiledStateGraph:
        """Get a compiled workflow by name."""
        workflow = self._compiled_workflows.get(name)
        if not workflow:
            # This error is raised if the requested workflow name doesn't exist or failed during initialization.
            raise ValueError(f"Workflow '{name}' not found or failed to compile during initialization.")
        return workflow

# Global instance
# This initializes the registry and compiles the graphs when the module is imported.
workflow_registry = WorkflowRegistry()
"""This module contains a python class `AgentCallbackHandler` which extends `BaseCallbackHandler`
and defines methods to handle events when an LLM (Language Model) starts and ends running."""
from typing import Any, Dict, List
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult



class AgentCallbackHandler(BaseCallbackHandler):
    """This Python class `AgentCallbackHandler` defines methods to handle events when a Language Model
    (LLM) starts and ends running."""
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running."""
        print(f"Prompt to LLM was:***\n{prompts[0]}")
        print("*******")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running."""
        print(f"***LLM Response:***\n{response.generations[0][0].text}")
        print("*******")

"""LiteLLM client wrapper for Graflow."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from graflow.utils.dotenv import load_env

if TYPE_CHECKING:
    from litellm import completion
    from litellm.types.utils import ModelResponse

try:
    from litellm import completion

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global flag to track if tracing has been set up
_litellm_tracing_enabled = False


def _patch_litellm_get_span_name() -> None:
    """Patch LiteLLM's _get_span_name to add span_id for unique span names.

    This prevents all litellm_request spans from being grouped together
    in the Langfuse UI by appending the current span_id to the span name.
    """
    try:
        from litellm.integrations.opentelemetry import (
            LITELLM_REQUEST_SPAN_NAME,
            OpenTelemetry,
        )
        from openinference.instrumentation.helpers import get_span_id
        from opentelemetry import trace

        # Check if already patched
        if hasattr(OpenTelemetry, "_get_span_name_original"):
            logger.debug("LiteLLM's _get_span_name already patched - skipping")
            return

        def _get_span_name_patched(self: OpenTelemetry, kwargs: dict) -> str:
            """
            Patched version of _get_span_name to generate more specific span names.
            It uses 'generation_name' from metadata if available, and appends span_id
            to make span names unique.
            """
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {}) or {}
            generation_name = metadata.get("generation_name")

            # Use generation_name for the span if provided, otherwise fallback
            span_name = str(generation_name if generation_name else LITELLM_REQUEST_SPAN_NAME)

            # Append current span_id to make span name unique
            current_span = get_span_id(trace.get_current_span())
            # Only add span_id if it's valid (not all zeros = invalid span)
            if current_span and current_span != "0000000000000000":
                return f"{span_name} [{current_span}]"
            return span_name

        # Store the original method and apply the patch
        OpenTelemetry._get_span_name_original = OpenTelemetry._get_span_name  # type: ignore
        OpenTelemetry._get_span_name = _get_span_name_patched  # type: ignore

        logger.info("Successfully patched LiteLLM's _get_span_name for unique span names")

    except ImportError as e:
        logger.warning(f"Could not patch LiteLLM's _get_span_name: {e}. This is expected if litellm is not installed.")
    except Exception as e:
        logger.error(f"Failed to patch LiteLLM's _get_span_name: {e}", exc_info=True)


def _patch_litellm_maybe_log_raw_request() -> None:
    """Patch LiteLLM's _maybe_log_raw_request to add span_id for unique raw span names.

    This prevents all raw_gen_ai_request spans from being grouped together
    in the Langfuse UI by appending the current span_id to the span name.
    """
    try:
        import litellm  # type: ignore[import-not-found]
        from litellm.integrations.opentelemetry import (
            RAW_REQUEST_SPAN_NAME,
            OpenTelemetry,
        )
        from openinference.instrumentation.helpers import get_span_id
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import Status, StatusCode

        # Check if already patched
        if hasattr(OpenTelemetry, "_maybe_log_raw_request_original"):
            logger.debug("LiteLLM's _maybe_log_raw_request already patched - skipping")
            return

        def _maybe_log_raw_request_patched(
            self: OpenTelemetry, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any, parent_span: Any
        ) -> None:
            """
            Patched version of _maybe_log_raw_request that adds span_id to raw span names.
            """
            # Only log raw LLM request/response if message_logging is on and not globally turned off
            if litellm.turn_off_message_logging or not self.message_logging:
                return

            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata") or {}
            generation_name = metadata.get("generation_name")

            # Base span name (same logic as original)
            raw_span_name = generation_name if generation_name else RAW_REQUEST_SPAN_NAME

            # Append current span_id to make span name unique
            current_span_id = get_span_id(otel_trace.get_current_span())
            # Only add span_id if it's valid (not all zeros = invalid span)
            if current_span_id and current_span_id != "0000000000000000":
                raw_span_name = f"{raw_span_name} [{current_span_id}]"

            # Create raw request span with unique name
            otel_tracer = self.get_tracer_to_use_for_request(kwargs)
            raw_span = otel_tracer.start_span(
                name=raw_span_name,
                start_time=self._to_ns(start_time),
                context=otel_trace.set_span_in_context(parent_span),
            )
            raw_span.set_status(Status(StatusCode.OK))
            self.set_raw_request_attributes(raw_span, kwargs, response_obj)
            raw_span.end(end_time=self._to_ns(end_time))

        # Store the original method and apply the patch
        OpenTelemetry._maybe_log_raw_request_original = OpenTelemetry._maybe_log_raw_request  # type: ignore
        OpenTelemetry._maybe_log_raw_request = _maybe_log_raw_request_patched  # type: ignore

        logger.info("Successfully patched LiteLLM's _maybe_log_raw_request for unique raw span names")

    except ImportError as e:
        logger.warning(
            f"Could not patch LiteLLM's _maybe_log_raw_request: {e}. This is expected if litellm is not installed."
        )
    except Exception as e:
        logger.error(f"Failed to patch LiteLLM's _maybe_log_raw_request: {e}", exc_info=True)


def setup_langfuse_for_litellm() -> None:
    """Setup Langfuse integration for LiteLLM.

    This configures LiteLLM to send traces to Langfuse using the OpenTelemetry-based
    integration. When used with LangFuseTracer, LiteLLM will automatically detect
    OpenTelemetry context (trace_id, span_id) set by the tracer and create properly
    nested spans in Langfuse.

    Uses the "langfuse_otel" callback which is compatible with Langfuse SDK v3+.

    Loads credentials from environment variables (via .env file):
    - LANGFUSE_PUBLIC_KEY: Langfuse public API key
    - LANGFUSE_SECRET_KEY: Langfuse secret API key
    - LANGFUSE_HOST: Langfuse host URL (optional, defaults to cloud.langfuse.com)

    Note:
        If credentials are not found, logs a warning and returns without enabling tracing.

    Example:
        ```python
        # .env file:
        # LANGFUSE_PUBLIC_KEY=pk-lf-...
        # LANGFUSE_SECRET_KEY=sk-lf-...
        # LANGFUSE_HOST=https://cloud.langfuse.com  # Optional

        from graflow.llm import setup_langfuse_for_litellm

        setup_langfuse_for_litellm()
        ```

    Note:
        - This function is idempotent (safe to call multiple times)
        - Uses "langfuse_otel" callback (compatible with Langfuse SDK v3+)
        - When LangFuseTracer is active, the OpenTelemetry context it sets will
          be automatically picked up by LiteLLM's Langfuse callback, creating
          properly nested traces in Langfuse UI.
        - LLMClient calls this automatically when enable_tracing=True (default)
        - Applies monkeypatches to LiteLLM's span naming methods for unique span names
    """
    global _litellm_tracing_enabled  # noqa: PLW0603

    # Check if already enabled (idempotent)
    if _litellm_tracing_enabled:
        logger.debug("LiteLLM Langfuse tracing already enabled")
        return

    # Load environment variables from .env file
    load_env()

    # Get credentials from environment
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        logger.warning(
            "Langfuse credentials not found. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables to enable tracing. "
            "Skipping Langfuse integration."
        )
        return

    # Enable Langfuse callback in LiteLLM
    # LiteLLM will automatically detect OpenTelemetry context set by LangFuseTracer
    # and create properly nested spans in Langfuse
    try:
        import litellm  # type: ignore[import-not-found]

        # Apply monkeypatches for unique span names
        # These patches add span_id to span names to prevent grouping in Langfuse UI
        _patch_litellm_get_span_name()  # Patch litellm_request spans
        _patch_litellm_maybe_log_raw_request()  # Patch raw_gen_ai_request spans

        # Use langfuse_otel callback for Langfuse SDK v3+ compatibility
        # This uses OpenTelemetry integration and avoids the sdk_integration parameter issue
        if "langfuse_otel" not in litellm.callbacks:
            litellm.callbacks.append("langfuse_otel")

        _litellm_tracing_enabled = True
        logger.info("Langfuse tracing enabled for LiteLLM (langfuse_otel callback)")
    except Exception as e:
        logger.warning(f"Failed to enable Langfuse tracing: {e}")


def extract_text(response: ModelResponse) -> str:
    """Extract text content from LiteLLM completion response.

    Args:
        response: LiteLLM ModelResponse object

    Returns:
        Text content from the first choice, or empty string if not found
    """
    try:
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                return getattr(choice.message, "content", "") or ""  # type: ignore
        return ""
    except Exception as e:
        logger.warning(f"Failed to extract text from response: {e}")
        return ""


class LLMClient:
    """Thin wrapper around LiteLLM for Graflow integration.

    This client provides a simple completion API that integrates with Graflow's
    tracing system. When Langfuse tracing is enabled (via setup_langfuse_for_litellm()),
    LiteLLM automatically detects OpenTelemetry context set by LangFuseTracer and
    creates nested spans.

    A single LLMClient instance is shared across all tasks in a workflow. Tasks
    can specify different models per completion call, allowing flexible model
    selection within the same task.

    Example:
        ```python
        from graflow.llm import LLMClient, setup_langfuse_for_litellm

        # Enable Langfuse tracing (optional)
        setup_langfuse_for_litellm()  # Loads from .env

        # Create client with default model
        llm_client = LLMClient(model="gpt-4o-mini", temperature=0.7)
        ```

    Example with @task decorator:
        ```python
        from graflow.core.decorators import task
        from graflow.llm import LLMClient

        @task(inject_llm_client=True)
        def flexible_task(llm: LLMClient, data: str) -> dict:
            # Use different models in the same task
            summary = llm.completion(
                messages=[{"role": "user", "content": data}],
                model="gpt-4o-mini"  # Fast, cheap
            )
            analysis = llm.completion(
                messages=[{"role": "user", "content": data}],
                model="gpt-4o"  # More powerful
            )
            return {"summary": summary, "analysis": analysis}
        ```
    """

    def __init__(self, model: str, enable_tracing: bool = True, **default_params: Any):
        """Initialize LLMClient.

        Args:
            model: Model identifier (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022").
                   This model is used as default for all completion() calls unless
                   overridden per call.
            enable_tracing: If True, automatically setup Langfuse tracing (default: True).
                          Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars.
                          Set to False if you want to manually control tracing setup.
            **default_params: Default parameters for litellm.completion() (e.g.,
                             temperature, max_tokens)

        Raises:
            RuntimeError: If litellm is not installed

        Example:
            ```python
            # Create client with automatic tracing (default)
            client = LLMClient(model="gpt-4o-mini", temperature=0.7)

            # Disable automatic tracing
            client = LLMClient(model="gpt-4o-mini", enable_tracing=False)

            # Use default model
            client.completion([...])  # Uses gpt-4o-mini

            # Override model for specific call
            client.completion([...], model="gpt-4o")  # Uses gpt-4o
            ```

        Note:
            If enable_tracing=True (default), this will automatically call
            setup_langfuse_for_litellm() to enable LiteLLM Langfuse callbacks.
        """
        if not LITELLM_AVAILABLE:
            raise RuntimeError("liteLLM is not installed. Install with: pip install litellm")

        # Setup tracing if enabled (idempotent - safe to call multiple times)
        if enable_tracing:
            setup_langfuse_for_litellm()

        self.model = model
        self.default_params = default_params

    def completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        generation_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **params: Any,
    ) -> ModelResponse:
        """Call LLM completion API.

        This method automatically integrates with Graflow's tracing system:
        - When LangFuseTracer is active, it sets OpenTelemetry context
        - LiteLLM's Langfuse callback detects this context automatically
        - LLM calls appear as nested spans in Langfuse UI

        Args:
            messages: List of message dicts with "role" and "content" keys
            model: Model to use. If not specified, uses self.model from __init__.
            generation_name: Optional name for Langfuse generation
            tags: Optional tags for Langfuse generation
            **params: Additional parameters for litellm.completion() (e.g., temperature,
                     max_tokens). These override default_params for this call only.

        Returns:
            LiteLLM ModelResponse object with completion results

        Example:
            ```python
            # Use default model
            response = client.completion(
                messages=[{"role": "user", "content": "Hello!"}]
            )

            # Override model for this call
            response = client.completion(
                messages=[{"role": "user", "content": "Complex task"}],
                model="gpt-4o",  # Use more powerful model
                temperature=0.7
            )

            # With Langfuse metadata
            response = client.completion(
                messages=[{"role": "user", "content": "Hello!"}],
                model="gpt-4o-mini",
                generation_name="greeting",
                tags=["test", "greeting"]
            )
            ```
        """
        # Merge parameters (call-specific overrides default)
        kwargs = {**self.default_params, **params}

        # Determine model to use (call-specific overrides default)
        actual_model = model or self.model

        # Add Langfuse metadata if provided
        if generation_name or tags:
            metadata = kwargs.get("metadata", {})
            if generation_name:
                metadata["generation_name"] = generation_name
            if tags:
                metadata["tags"] = tags
            kwargs["metadata"] = metadata

        # Call LiteLLM completion
        # LiteLLM's Langfuse callback will automatically detect OpenTelemetry context
        # set by LangFuseTracer and create nested spans
        return completion(model=actual_model, messages=messages, stream=False, **kwargs)  # type: ignore

    def completion_text(self, messages: List[Dict[str, str]], **params: Any) -> str:
        """Convenience method that returns text content directly.

        Args:
            messages: List of message dicts
            **params: Parameters for completion()

        Returns:
            Text content from first choice

        Example:
            ```python
            text = client.completion_text(
                messages=[{"role": "user", "content": "Hello!"}]
            )
            print(text)  # "Hello! How can I help you today?"
            ```
        """
        response = self.completion(messages, **params)
        return extract_text(response)


def make_message(role: str, content: str) -> Dict[str, str]:
    """Convenience helper to create OpenAI-style chat messages.

    Args:
        role: Message role ("system", "user", "assistant")
        content: Message content

    Returns:
        Message dict with role and content

    Example:
        ```python
        messages = [
            make_message("system", "You are helpful."),
            make_message("user", "Hello!")
        ]
        ```
    """
    return {"role": role, "content": content}

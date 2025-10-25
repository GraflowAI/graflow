"""Workflow execution tracing module.

This module provides abstract and concrete implementations for
tracing workflow execution in Graflow.

Available Tracers:
- Tracer: Abstract base class for all tracers
- NoopTracer: Default tracer that tracks runtime graph but produces no output
- ConsoleTracer: Console output tracer with formatted, colorized output
- LangFuseTracer: LangFuse integration tracer for LLM workflow observability

Examples:
    # Using NoopTracer (default)
    from graflow.trace import NoopTracer
    from graflow.core.context import ExecutionContext

    tracer = NoopTracer(enable_runtime_graph=True)
    context = ExecutionContext.create(graph, start_node, tracer=tracer)

    # Using ConsoleTracer for visible output
    from graflow.trace import ConsoleTracer

    tracer = ConsoleTracer(enable_colors=True, show_metadata=True)
    context = ExecutionContext.create(graph, start_node, tracer=tracer)

    # Using LangFuseTracer for LLM observability
    from graflow.trace import LangFuseTracer

    # Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env file
    tracer = LangFuseTracer()
    context = ExecutionContext.create(graph, start_node, tracer=tracer)
"""

__all__ = []

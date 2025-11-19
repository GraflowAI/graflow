"""Unit tests for LangFuseTracer."""

from unittest.mock import MagicMock, patch


class TestLangFuseTracerBasics:
    """Test basic LangFuseTracer functionality."""

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_initialization_enabled(self, mock_langfuse_class):
        """Test LangFuseTracer initialization when enabled."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(
            public_key="pk-test",
            secret_key="sk-test",
            enabled=True
        )

        assert tracer.enabled is True
        assert tracer.client == mock_client
        mock_langfuse_class.assert_called_once()

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    def test_initialization_disabled(self):
        """Test LangFuseTracer initialization when disabled."""
        from graflow.trace.langfuse import LangFuseTracer

        tracer = LangFuseTracer(enabled=False)
        assert tracer.enabled is False
        assert tracer.client is None

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    @patch("graflow.trace.langfuse.load_env")
    def test_initialization_from_env(self, mock_load_env, mock_langfuse_class):
        """Test LangFuseTracer loads config from .env."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client

        with patch.dict("os.environ", {
            "LANGFUSE_PUBLIC_KEY": "pk-env",
            "LANGFUSE_SECRET_KEY": "sk-env",
            "LANGFUSE_HOST": "https://test.langfuse.com"
        }):
            tracer = LangFuseTracer()

        mock_load_env.assert_called_once()
        assert tracer.enabled is True


class TestLangFuseTracerTraceLifecycle:
    """Test LangFuseTracer trace lifecycle."""

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_trace_start_creates_root_span(self, mock_langfuse_class):
        """Test trace_start creates a root span."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_root_span = MagicMock()
        mock_client.start_span.return_value = mock_root_span
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        tracer.trace_start("test_workflow", trace_id="trace_123")

        # Should create root span with trace_id
        mock_client.start_span.assert_called_once()
        call_kwargs = mock_client.start_span.call_args.kwargs
        assert call_kwargs["trace_context"] == {"trace_id": "trace_123"}
        assert call_kwargs["name"] == "test_workflow"

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_trace_end_ends_root_span(self, mock_langfuse_class):
        """Test trace_end ends the root span."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_root_span = MagicMock()
        mock_client.start_span.return_value = mock_root_span
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        tracer.trace_start("test_workflow")
        tracer.trace_end("test_workflow", output={"result": "success"})

        # Should update and end root span
        mock_root_span.update.assert_called_once()
        mock_root_span.end.assert_called_once()


class TestLangFuseTracerSpanLifecycle:
    """Test LangFuseTracer span lifecycle."""

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_span_start_creates_child_span(self, mock_langfuse_class):
        """Test span_start creates a child span."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_root_span = MagicMock()
        mock_task_span = MagicMock()
        mock_root_span.start_span.return_value = mock_task_span
        mock_client.start_span.return_value = mock_root_span
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        tracer.trace_start("workflow")
        tracer.span_start("task_1", metadata={"task_type": "Task"})

        # Should create child span from root
        mock_root_span.start_span.assert_called_once()
        call_kwargs = mock_root_span.start_span.call_args.kwargs
        assert call_kwargs["name"] == "task_1"
        assert call_kwargs["metadata"] == {"task_type": "Task"}

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_span_end_updates_and_closes_span(self, mock_langfuse_class):
        """Test span_end updates and closes the span."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_root_span = MagicMock()
        mock_task_span = MagicMock()
        mock_root_span.start_span.return_value = mock_task_span
        mock_client.start_span.return_value = mock_root_span
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        tracer.trace_start("workflow")
        tracer.span_start("task_1")
        tracer.span_end("task_1", output={"result": 42})

        # Should update and end span
        mock_task_span.update.assert_called_once()
        update_kwargs = mock_task_span.update.call_args.kwargs
        assert update_kwargs["output"] == {"result": 42}
        mock_task_span.end.assert_called_once()

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_nested_spans(self, mock_langfuse_class):
        """Test nested span creation."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_root_span = MagicMock()
        mock_parent_span = MagicMock()
        mock_child_span = MagicMock()

        mock_root_span.start_span.return_value = mock_parent_span
        mock_parent_span.start_span.return_value = mock_child_span
        mock_client.start_span.return_value = mock_root_span
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        tracer.trace_start("workflow")
        tracer.span_start("parent")
        tracer.span_start("child", parent_name="parent")

        # Parent should be created from root
        mock_root_span.start_span.assert_called_once()
        # Child should be created from parent
        mock_parent_span.start_span.assert_called_once()


class TestLangFuseTracerEvents:
    """Test LangFuseTracer event recording."""

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_event_creates_event_on_current_span(self, mock_langfuse_class):
        """Test event creates an event on the current span."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_root_span = MagicMock()
        mock_task_span = MagicMock()
        mock_root_span.start_span.return_value = mock_task_span
        mock_client.start_span.return_value = mock_root_span
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        tracer.trace_start("workflow")
        tracer.span_start("task")
        tracer.event("task_queued", metadata={"task_id": "next_task"})

        # Should create event on current span
        mock_task_span.create_event.assert_called_once()
        call_kwargs = mock_task_span.create_event.call_args.kwargs
        assert call_kwargs["name"] == "task_queued"
        assert call_kwargs["metadata"] == {"task_id": "next_task"}


class TestLangFuseTracerCloning:
    """Test LangFuseTracer cloning for parallel execution."""

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_clone_creates_new_instance(self, mock_langfuse_class):
        """Test clone creates a new tracer instance."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        cloned = tracer.clone("trace_123")

        assert cloned is not tracer
        assert isinstance(cloned, LangFuseTracer)
        assert cloned.enabled is True

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_clone_shares_client(self, mock_langfuse_class):
        """Test clone shares the Langfuse client."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        cloned = tracer.clone("trace_123")

        # Should share the same client (thread-safe)
        assert cloned.client is tracer.client

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    @patch("graflow.trace.langfuse.copy")
    def test_clone_copies_parent_span(self, mock_copy_module, mock_langfuse_class):
        """Test clone shallow copies parent's current span."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_root_span = MagicMock()
        mock_parallel_span = MagicMock()
        mock_copied_span = MagicMock()

        mock_root_span.start_span.return_value = mock_parallel_span
        mock_client.start_span.return_value = mock_root_span
        mock_copy_module.copy.return_value = mock_copied_span
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        tracer.trace_start("workflow")
        tracer.span_start("parallel_group")

        cloned = tracer.clone("trace_123")

        # Should shallow copy the current span
        mock_copy_module.copy.assert_called_once_with(mock_parallel_span)
        assert cloned._root_span == mock_copied_span

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_clone_has_independent_span_stack(self, mock_langfuse_class):
        """Test cloned tracer has independent span stack."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_root_span = MagicMock()
        mock_parallel_span = MagicMock()
        mock_task_span = MagicMock()

        mock_root_span.start_span.return_value = mock_parallel_span
        # When cloned tracer starts a span, it should use the copied root span
        mock_parallel_span.start_span = MagicMock(return_value=mock_task_span)
        mock_client.start_span.return_value = mock_root_span
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        tracer.trace_start("workflow")
        tracer.span_start("parallel_group")

        # Original has parallel_group in stack
        assert len(tracer._span_stack) == 1

        cloned = tracer.clone("trace_123")

        # Cloned should have empty span stack
        assert len(cloned._span_stack) == 0

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    def test_clone_when_disabled(self):
        """Test cloning a disabled tracer."""
        from graflow.trace.langfuse import LangFuseTracer

        tracer = LangFuseTracer(enabled=False)
        cloned = tracer.clone("trace_123")

        assert isinstance(cloned, LangFuseTracer)
        assert cloned.enabled is False
        assert cloned.client is None


class TestLangFuseTracerAttachToTrace:
    """Test LangFuseTracer attach_to_trace for distributed tracing."""

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_attach_to_trace_connects_to_existing_trace(self, mock_langfuse_class):
        """Test attach_to_trace connects to existing trace."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_worker_span = MagicMock()
        mock_client.start_span.return_value = mock_worker_span
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        tracer.attach_to_trace("trace_123", parent_span_id="span_456")

        # Should create root span with trace_id and parent_span_id
        mock_client.start_span.assert_called_once()
        call_kwargs = mock_client.start_span.call_args.kwargs
        assert call_kwargs["trace_context"] == {
            "trace_id": "trace_123",
            "parent_span_id": "span_456"
        }


class TestLangFuseTracerFlush:
    """Test LangFuseTracer flush functionality."""

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_flush_calls_client_flush(self, mock_langfuse_class):
        """Test flush calls client.flush()."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        tracer.flush()

        mock_client.flush.assert_called_once()

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_shutdown_calls_client_flush(self, mock_langfuse_class):
        """Test shutdown calls client.flush()."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        tracer.shutdown()

        mock_client.flush.assert_called_once()


class TestLangFuseTracerRuntimeGraph:
    """Test LangFuseTracer runtime graph tracking."""

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_runtime_graph_tracks_execution(self, mock_langfuse_class):
        """Test runtime graph tracks execution alongside LangFuse."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_root_span = MagicMock()
        mock_task_span = MagicMock()
        mock_root_span.start_span.return_value = mock_task_span
        mock_client.start_span.return_value = mock_root_span
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(
            public_key="pk",
            secret_key="sk",
            enabled=True,
            enable_runtime_graph=True
        )
        tracer.trace_start("workflow")
        tracer.span_start("task_1")
        tracer.span_end("task_1")
        tracer.trace_end("workflow")

        # Runtime graph should track execution
        graph = tracer.get_runtime_graph()
        assert graph is not None
        assert "task_1" in graph.nodes
        assert graph.nodes["task_1"]["status"] == "completed"

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse")
    def test_runtime_graph_disabled(self, mock_langfuse_class):
        """Test runtime graph can be disabled."""
        from graflow.trace.langfuse import LangFuseTracer

        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client

        tracer = LangFuseTracer(
            public_key="pk",
            secret_key="sk",
            enabled=True,
            enable_runtime_graph=False
        )

        assert tracer.get_runtime_graph() is None

"""
Unit tests for LLM Integration from Tasks and Workflows Guide.

Tests LLM client injection and agent injection features documented in the guide.
Uses mocking to avoid requiring actual API keys.
"""

from unittest.mock import Mock, PropertyMock, patch

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.llm.agents.base import LLMAgent
from graflow.llm.client import LLMClient


class TestLLMClientInjection:
    """Tests for inject_llm_client=True"""

    def test_llm_client_injection_basic(self):
        """Test basic LLM client injection"""

        with patch("graflow.llm.client.LLMClient") as patching_llm_client:
            mock_client = Mock(spec=LLMClient)
            mock_client.completion_text.return_value = "Hello, I'm an AI assistant!"
            patching_llm_client.return_value = mock_client

            with workflow("llm_test") as wf:

                @task(inject_llm_client=True)
                def generate_text(llm_client: LLMClient):
                    response = llm_client.completion_text(
                        messages=[{"role": "user", "content": "Say hello"}],
                        max_tokens=50
                    )
                    return response

                _, ctx = wf.execute(ret_context=True)

            result = ctx.get_result("generate_text")
            assert result == "Hello, I'm an AI assistant!"

    def test_llm_client_multiple_tasks(self):
        """Test LLM client injection across multiple tasks"""

        with patch("graflow.llm.client.LLMClient") as patching_llm_client:
            mock_client = Mock(spec=LLMClient)
            mock_client.completion_text.side_effect = [
                "Greeting message",
                "Answer to question",
                "Summary text"
            ]
            patching_llm_client.return_value = mock_client

            with workflow("multi_llm") as wf:

                @task(inject_llm_client=True)
                def task1(llm_client: LLMClient):
                    return llm_client.completion_text(
                        messages=[{"role": "user", "content": "Greet"}],
                        max_tokens=50
                    )

                @task(inject_llm_client=True)
                def task2(llm_client: LLMClient):
                    return llm_client.completion_text(
                        messages=[{"role": "user", "content": "Answer"}],
                        max_tokens=50
                    )

                @task(inject_llm_client=True)
                def task3(llm_client: LLMClient):
                    return llm_client.completion_text(
                        messages=[{"role": "user", "content": "Summarize"}],
                        max_tokens=50
                    )

                task1 >> task2 >> task3  # type: ignore

                _, ctx = wf.execute(ret_context=True)

            assert ctx.get_result("task1") == "Greeting message"
            assert ctx.get_result("task2") == "Answer to question"
            assert ctx.get_result("task3") == "Summary text"

    def test_llm_client_with_context_injection(self):
        """Test combining LLM client injection with context injection"""

        with patch("graflow.llm.client.LLMClient") as patching_llm_client:
            mock_client = Mock(spec=LLMClient)
            mock_client.completion_text.return_value = "AI response"
            patching_llm_client.return_value = mock_client

            with workflow("combined_injection") as wf:

                @task(inject_context=True, inject_llm_client=True)
                def process_with_llm(ctx: TaskExecutionContext, llm_client: LLMClient, prompt: str):
                    # Access both context and LLM client
                    channel = ctx.get_channel()
                    channel.set("prompt_used", prompt)

                    response = llm_client.completion_text(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100
                    )

                    channel.set("llm_response", response)
                    return response

                _, ctx = wf.execute(
                    ret_context=True,
                    initial_channel={"prompt": "Explain Python"}
                )

            result = ctx.get_result("process_with_llm")
            assert result == "AI response"
            assert ctx.get_channel().get("prompt_used") == "Explain Python"
            assert ctx.get_channel().get("llm_response") == "AI response"

    def test_llm_client_via_context_access(self):
        """Test accessing LLM client via context instead of injection"""

        mock_client = Mock(spec=LLMClient)
        mock_client.completion_text.return_value = "Context-accessed response"

        with patch.object(
            TaskExecutionContext,
            "llm_client",
            new_callable=PropertyMock,
            return_value=mock_client
        ):
            with workflow("context_llm") as wf:

                @task(inject_context=True)
                def use_llm_via_context(ctx: TaskExecutionContext):
                    # Access LLM client through context
                    response = ctx.llm_client.completion_text(
                        messages=[{"role": "user", "content": "Test"}],
                        max_tokens=50
                    )
                    return response

                result = wf.execute()
                assert result == "Context-accessed response"

            # Note: This test verifies the API pattern even if llm_client
            # property might not be fully implemented in context


class TestLLMAgentInjection:
    """Tests for inject_llm_agent='agent_name'"""

    def test_llm_agent_registration_and_injection(self):
        """Test registering and injecting LLM agent"""

        # Create mock agent
        mock_agent = Mock()
        mock_agent.run.return_value = {"output": "Agent result", "metadata": {}}

        with workflow("agent_test") as wf:
            # Register agent via workflow context (use factory pattern)
            wf.register_llm_agent("my_agent", lambda ctx: mock_agent)

            @task(inject_llm_agent="my_agent")
            def use_agent(llm_agent: LLMAgent):
                result = llm_agent.run("Process this query")
                return result["output"]

            result = wf.execute()

        assert result == "Agent result"
        mock_agent.run.assert_called_once_with("Process this query")

    def test_llm_agent_with_tools(self):
        """Test LLM agent with tool calls"""

        mock_agent = Mock()
        mock_agent.run.return_value = {
            "output": "Final answer after tool use",
            "metadata": {
                "tool_calls": [
                    {"tool": "search", "args": {"query": "test"}},
                    {"tool": "calculate", "args": {"expr": "2+2"}}
                ]
            }
        }

        with workflow("agent_tools") as wf:
            wf.register_llm_agent("tool_agent", lambda ctx: mock_agent)

            @task(inject_llm_agent="tool_agent")
            def complex_task(llm_agent: LLMAgent, query: str):
                result = llm_agent.run(query)
                return result

            result = wf.execute(initial_channel={"query": "What is 2+2 in the context of search?"})

        assert isinstance(result, dict)
        assert result["output"] == "Final answer after tool use"
        assert len(result["metadata"]["tool_calls"]) == 2

    def test_multiple_agents_in_workflow(self):
        """Test workflow with multiple different agents"""

        mock_agent1 = Mock()
        mock_agent1.run.return_value = {"output": "Agent 1 result"}

        mock_agent2 = Mock()
        mock_agent2.run.return_value = {"output": "Agent 2 result"}

        with workflow("multi_agent") as wf:
            wf.register_llm_agent("analyzer", lambda ctx: mock_agent1)
            wf.register_llm_agent("summarizer", lambda ctx: mock_agent2)

            @task(inject_llm_agent="analyzer")
            def analyze(llm_agent: LLMAgent):
                return llm_agent.run("Analyze this")["output"]

            @task(inject_llm_agent="summarizer")
            def summarize(llm_agent: LLMAgent):
                return llm_agent.run("Summarize this")["output"]

            analyze_task = analyze(task_id="analyze")
            summarize_task = summarize(task_id="summarize")

            analyze_task >> summarize_task  # type: ignore

            _, ctx = wf.execute(ret_context=True)

        assert ctx.get_result("analyze") == "Agent 1 result"
        assert ctx.get_result("summarize") == "Agent 2 result"

    def test_agent_with_context_injection(self):
        """Test combining agent injection with context injection"""

        mock_agent = Mock()
        mock_agent.run.return_value = {"output": "Agent response", "confidence": 0.95}

        with workflow("agent_context") as wf:
            wf.register_llm_agent("helper", lambda ctx: mock_agent)

            @task(inject_context=True, inject_llm_agent="helper")
            def process_with_agent(ctx: TaskExecutionContext, llm_agent: LLMAgent, data: dict):
                # Use both context and agent
                channel = ctx.get_channel()

                result = llm_agent.run(f"Process: {data}")
                channel.set("agent_confidence", result.get("confidence", 0))

                return result["output"]

            result = wf.execute(initial_channel={"data": {"value": 42}})

        assert result == "Agent response"


class TestLLMIntegrationPatterns:
    """Test common LLM integration patterns from the guide"""

    def test_multi_model_scenario(self):
        """Test using different models for different tasks"""

        with patch("graflow.llm.client.LLMClient") as patching_llm_client:
            # Different mock responses for different tasks
            mock_client = Mock(spec=LLMClient)

            def completion_side_effect(messages, model=None, **kwargs):
                if model == "gpt-4o-mini":
                    return "Quick answer"
                elif model == "gpt-4":
                    return "Detailed analysis"
                return "Default response"

            mock_client.completion_text.side_effect = completion_side_effect
            patching_llm_client.return_value = mock_client

            with workflow("multi_model") as wf:

                @task(inject_llm_client=True)
                def quick_task(llm_client: LLMClient):
                    # Use cheap model for simple task
                    return llm_client.completion_text(
                        messages=[{"role": "user", "content": "Quick question"}],
                        model="gpt-4o-mini"
                    )

                @task(inject_llm_client=True)
                def complex_task(llm_client: LLMClient):
                    # Use expensive model for complex task
                    return llm_client.completion_text(
                        messages=[{"role": "user", "content": "Complex analysis"}],
                        model="gpt-4"
                    )

                quick_task >> complex_task  # type: ignore

                _, ctx = wf.execute(ret_context=True)

            assert ctx.get_result("quick_task") == "Quick answer"
            assert ctx.get_result("complex_task") == "Detailed analysis"

    def test_llm_with_retry_pattern(self):
        """Test LLM task with retry pattern"""

        with patch("graflow.llm.client.LLMClient") as patching_llm_client:
            mock_client = Mock(spec=LLMClient)
            call_count = {"count": 0}

            def completion_with_retry(messages, **kwargs):
                call_count["count"] += 1
                if call_count["count"] < 2:
                    raise Exception("API error")
                return "Success after retry"

            mock_client.completion_text.side_effect = completion_with_retry
            patching_llm_client.return_value = mock_client

            with workflow("llm_retry") as wf:

                @task(inject_context=True, inject_llm_client=True)
                def retry_llm_call(ctx: TaskExecutionContext, llm_client: LLMClient):
                    channel = ctx.get_channel()
                    attempts = channel.get("attempts", default=0)

                    try:
                        response = llm_client.completion_text(
                            messages=[{"role": "user", "content": "Test"}]
                        )
                        return response
                    except Exception as e:
                        if attempts < 2:
                            channel.set("attempts", attempts + 1)
                            ctx.next_iteration()
                            return None
                        raise e

                result = wf.execute()

            assert result == "Success after retry"
            assert call_count["count"] == 2

    def test_llm_response_to_channel(self):
        """Test storing LLM responses in channel for downstream tasks"""

        with patch("graflow.llm.client.LLMClient") as patching_llm_client:
            mock_client = Mock(spec=LLMClient)
            mock_client.completion_text.return_value = "Analyzed data: positive sentiment"
            patching_llm_client.return_value = mock_client

            with workflow("llm_pipeline") as wf:

                @task(inject_context=True, inject_llm_client=True)
                def analyze(ctx: TaskExecutionContext, llm_client: LLMClient, text: str):
                    response = llm_client.completion_text(
                        messages=[{"role": "user", "content": f"Analyze: {text}"}]
                    )
                    # Store in channel for downstream
                    ctx.get_channel().set("analysis", response)
                    return response

                @task(inject_context=True)
                def process_analysis(ctx: TaskExecutionContext):
                    analysis = ctx.get_channel().get("analysis")
                    # Process the LLM analysis
                    return {"processed": True, "analysis": analysis}

                analyze >> process_analysis  # type: ignore

                _, ctx = wf.execute(
                    ret_context=True,
                    initial_channel={"text": "Great product!"}
                )

            result = ctx.get_result("process_analysis")
            assert result["processed"] is True
            assert "positive sentiment" in result["analysis"]

# GPT Newspaper Workflow Comparison

This directory contains **three different implementations** of the newspaper generation workflow, each demonstrating different Graflow patterns and use cases.

## Quick Comparison

| Feature | Simple Workflow | Dynamic Workflow | **Agent Workflow** |
|---------|----------------|------------------|-------------------|
| **File** | `newspaper_workflow.py` | `newspaper_dynamic_workflow.py` | `newspaper_agent_workflow.py` |
| **Complexity** | Low | High | Medium |
| **LLM Agents** | 0 (pure LLM tasks) | 0 (pure LLM tasks) | **2 (researcher, editor)** |
| **Tool Calling** | ❌ No | ❌ No | **✅ Yes (8 tools)** |
| **Research** | Single search | Multi-angle fan-out | **Autonomous multi-search** |
| **Writing** | Single LLM call | 3 parallel personas | Single LLM with revision |
| **Quality Control** | Simple critique | 3 parallel gates | **Agent-driven editorial** |
| **Revision Loop** | Fixed logic | Fixed logic | **Agent-controlled** |
| **Dependencies** | OpenAI/LiteLLM | OpenAI/LiteLLM | **Google ADK, Tavily, textstat** |

---

## 1. Simple Workflow (`newspaper_workflow.py`)

### Architecture
```
search → curate → write → critique → design
                      ↑         ↓
                      └─ loop ─┘
```

### Key Characteristics
- **All LLM tasks**: Simple `inject_llm_client` pattern
- **Linear flow**: Clean task chain with loop-back
- **Fixed critique logic**: Predefined revision criteria
- **Best for**: Simple pipelines, cost optimization, learning Graflow basics

### Example Task
```python
@task(inject_llm_client=True)
def write_task(llm: LLMClient, article: Dict) -> Dict:
    """Simple LLM completion - no tools, no agents"""
    messages = [{"role": "user", "content": f"Write article: {article}"}]
    return llm.completion_text(messages, model="gpt-4o-mini")
```

### Use When
- Starting with Graflow
- Cost is primary concern
- Simple content generation needs
- No need for complex decision-making

---

## 2. Dynamic Workflow (`newspaper_dynamic_workflow.py`)

### Architecture
```
topic_intake → search_router → curate → writer_personas → select_draft → write → critique → quality_gate → design
                    ↓                           ↓                ↑    ↓             ↓
            [dynamic searches]          [3 parallel writers]    │    │    [3 parallel checks]
                                                                 └────┘
```

### Key Characteristics
- **Runtime task generation**: `context.next_task()` for dynamic fan-out
- **Parallel execution**: 3 writer personas, 3 quality gates
- **Group policies**: `BestEffortGroupPolicy`, `AtLeastNGroupPolicy`
- **Complex coordination**: Channel-based state management
- **Best for**: Complex workflows, parallel processing, production scale

### Example Pattern
```python
# Dynamic task fan-out
for angle in angles:
    def run_angle(angle_label=angle):
        result = search_agent.run({"query": f"{query} - {angle_label}"})
        return result

    context.next_task(TaskWrapper(f"search_{angle_id}", run_angle))

# Parallel writer personas
writer_personas = (
    write_feature_task | write_brief_task | write_data_digest_task
).with_execution(
    backend=CoordinationBackend.THREADING,
    policy=BestEffortGroupPolicy()
)

# Parallel quality gates
quality_gate = (
    fact_check_task | compliance_check_task | risk_check_task
).with_execution(
    backend=CoordinationBackend.THREADING,
    policy=AtLeastNGroupPolicy(min_success=2)
)
```

### Use When
- Need parallel execution
- Multiple angles/approaches required
- Quality gates and validation stages
- Production-scale content generation

---

## 3. Agent Workflow (`newspaper_agent_workflow.py`) ⭐ **NEW**

### Architecture
```
topic_intake → research_agent → curate → write → editorial_agent → design
                    ↓                        ↑         ↓
            [autonomous tools]               └─ loop ─┘
            - web_search                  [autonomous tools]
            - extract_facts               - check_facts
            - refine_query                - assess_readability
                                         - verify_sources
                                         - suggest_improvements
```

### Key Characteristics
- **LLM Agents with tools**: Real function calling (Tavily, textstat)
- **Autonomous decision-making**: Agents decide tool usage and flow
- **ReAct pattern**: Agents reason → act → observe → iterate
- **Agent-controlled loops**: Editorial agent decides approve/revise
- **Best for**: Quality-critical tasks, autonomous workflows, agentic systems

### Example Tasks

**Research Agent (Autonomous)**
```python
@task(inject_llm_agent="researcher")
def research_task(llm_agent: LLMAgent, query: str) -> Dict:
    """
    Agent autonomously:
    1. Searches web (web_search tool)
    2. Extracts key facts (extract_key_facts tool)
    3. Refines queries as needed (refine_search_query tool)
    4. Decides when to stop
    """
    result = llm_agent.run(
        f"Research '{query}'. Use web_search, extract_key_facts, "
        f"and refine_search_query tools as needed. Provide comprehensive report."
    )
    return result["output"]
```

**Editorial Agent (Quality Control)**
```python
@task(inject_llm_agent="editor")
def editorial_task(context: TaskExecutionContext, llm_agent: LLMAgent) -> Dict:
    """
    Agent autonomously:
    1. Checks factual claims (check_factual_claims tool)
    2. Assesses readability (assess_readability tool)
    3. Verifies sources (verify_sources tool)
    4. Decides: APPROVE or REVISE with feedback
    5. Controls revision loop
    """
    result = llm_agent.run(
        f"Review article. Use check_factual_claims, assess_readability, "
        f"verify_sources tools. Decide: APPROVE or REVISE with specific feedback."
    )

    if result["decision"] == "revise":
        context.next_task(write_task, goto=True)  # Agent controls loop

    return result
```

**Agent Registration**
```python
def register_agents(exec_context: TaskExecutionContext):
    # Research Agent with search tools
    research_agent = LlmAgent(
        name="researcher",
        model="gemini-2.0-flash-exp",
        tools=[web_search, extract_key_facts, refine_search_query]
    )
    exec_context.register_llm_agent(
        "researcher",
        AdkLLMAgent(research_agent, app_name=exec_context.session_id)
    )

    # Editorial Agent with verification tools
    editorial_agent = LlmAgent(
        name="editor",
        model="gemini-2.0-flash-exp",
        tools=[check_factual_claims, assess_readability, verify_sources, suggest_improvements]
    )
    exec_context.register_llm_agent(
        "editor",
        AdkLLMAgent(editorial_agent, app_name=exec_context.session_id)
    )
```

### Use When
- Need autonomous decision-making
- Quality is critical (fact-checking, verification)
- Want transparent tool usage (visible in traces)
- Building agentic systems
- Exploring LLM agent patterns

---

## Tool Implementations

The agent workflow includes **8 real tools**:

### Research Agent Tools
1. **`web_search(query)`** - Tavily API search
2. **`extract_key_facts(sources_json, focus)`** - Extract relevant information
3. **`refine_search_query(original_query, findings)`** - Generate follow-up queries

### Editorial Agent Tools
4. **`check_factual_claims(article_json, sources_json)`** - Cross-reference claims
5. **`assess_readability(text)`** - Textstat metrics (Flesch score, grade level)
6. **`verify_sources(sources_json)`** - Check source credibility
7. **`suggest_improvements(article_json, issues)`** - Generate specific suggestions

---

## Installation Requirements

### Simple Workflow
```bash
# Minimal requirements
export TAVILY_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
uv pip install graflow tavily-python
```

### Dynamic Workflow
```bash
# Same as simple workflow
export TAVILY_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
uv pip install graflow tavily-python
```

### Agent Workflow
```bash
# Additional dependencies for agents
export TAVILY_API_KEY="your-key"
export GPT_NEWSPAPER_MODEL="gpt-4o-mini"  # LiteLLM-compatible agent model
uv pip install graflow tavily-python
uv add google-adk textstat
```

---

## Running the Workflows

### Simple Workflow
```bash
cd examples/gpt_newspaper/backend
PYTHONPATH=../../.. python newspaper_workflow.py
```

### Dynamic Workflow
```bash
cd examples/gpt_newspaper/backend
PYTHONPATH=../../.. python newspaper_dynamic_workflow.py
```

### Agent Workflow
```bash
cd examples/gpt_newspaper/backend
PYTHONPATH=../../.. python newspaper_agent_workflow.py
```

---

## Performance Comparison

| Metric | Simple | Dynamic | Agent |
|--------|--------|---------|-------|
| **Latency** | ~30s | ~45s | ~60s |
| **LLM Calls** | 4-6 | 10-15 | 8-12 + tools |
| **Cost (relative)** | 1x | 2-3x | 1.5-2x |
| **Quality** | Good | Very Good | **Excellent** |
| **Autonomy** | Low | Medium | **High** |
| **Transparency** | Medium | Medium | **High (tools visible)** |

---

## Decision Guide

Choose **Simple Workflow** if:
- ✅ Learning Graflow basics
- ✅ Cost is primary concern
- ✅ Simple content generation
- ✅ No complex requirements

Choose **Dynamic Workflow** if:
- ✅ Need parallel execution
- ✅ Multiple processing approaches
- ✅ Quality gates and validation
- ✅ Production scale

Choose **Agent Workflow** if:
- ✅ Quality is critical
- ✅ Need fact-checking/verification
- ✅ Want autonomous decisions
- ✅ Building agentic systems
- ✅ Exploring LLM agent patterns

---

## Key Insights

### Simple → Dynamic
**Progression**: Learn parallel execution, runtime task generation, group policies

### Simple → Agent
**Progression**: Learn LLM agents, tool calling, autonomous decision-making

### Dynamic → Agent
**Comparison**: Dynamic has more parallel tasks, Agent has autonomous reasoning

### All Three
**Study**: See evolution from fixed logic → parallel complexity → autonomous agents

---

## Further Reading

- **Graflow LLM Integration**: `docs/llm_integration_design.md`
- **Multi-Agent Example**: `examples/12_llm_integration/multi_agent_workflow.py`
- **Google ADK**: https://developers.google.com/adk
- **Tavily API**: https://tavily.com/

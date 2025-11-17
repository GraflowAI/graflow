# Agent Workflow Visual Architecture

## Workflow Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGENT WORKFLOW ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────────┘

                        ┌──────────────────┐
                        │  topic_intake    │
                        │ • Setup query    │
                        │ • Register agents│
                        │ • Init channels  │
                        └────────┬─────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────────┐
        │          RESEARCH AGENT (autonomous)               │
        │  ┌──────────────────────────────────────────────┐  │
        │  │ 🤖 LLMAgent with ReAct Pattern               │  │
        │  │                                               │  │
        │  │ Tools:                                        │  │
        │  │  • web_search(query) → Tavily API           │  │
        │  │  • extract_key_facts(sources, focus)        │  │
        │  │  • refine_search_query(original, findings)  │  │
        │  │                                               │  │
        │  │ Flow:                                         │  │
        │  │  1. web_search("AI developments")            │  │
        │  │  2. extract_key_facts(results, "statistics") │  │
        │  │  3. refine_search_query("AI", findings)      │  │
        │  │  4. web_search(refined_query)                │  │
        │  │  5. Compile comprehensive report             │  │
        │  └──────────────────────────────────────────────┘  │
        └──────────────────────┬─────────────────────────────┘
                               │ {summary, sources, image}
                               ▼
                      ┌──────────────────┐
                      │     curate       │
                      │ Simple LLM task  │
                      │ • Structure      │
                      │   research       │
                      └────────┬─────────┘
                               │ {structure, sources}
                               ▼
                      ┌──────────────────┐
                      │      write       │◄────────┐
                      │ Simple LLM task  │         │
                      │ • Draft article  │         │
                      │ • OR revise with │         │
                      │   feedback       │         │
                      └────────┬─────────┘         │
                               │ {article}         │
                               ▼                   │
        ┌──────────────────────────────────────────┼─────┐
        │        EDITORIAL AGENT (autonomous)      │     │
        │  ┌────────────────────────────────────┐  │     │
        │  │ 🤖 LLMAgent with ReAct Pattern     │  │     │
        │  │                                     │  │     │
        │  │ Tools:                              │  │     │
        │  │  • check_factual_claims()          │  │     │
        │  │  • assess_readability() → textstat │  │     │
        │  │  • verify_sources()                │  │     │
        │  │  • suggest_improvements()          │  │     │
        │  │                                     │  │     │
        │  │ Flow:                               │  │     │
        │  │  1. check_factual_claims(article)  │  │     │
        │  │  2. assess_readability(content)    │  │     │
        │  │  3. verify_sources(sources)        │  │     │
        │  │  4. Decide: APPROVE or REVISE      │  │     │
        │  │  5. If REVISE:                     │  │     │
        │  │     suggest_improvements()         │  │     │
        │  │     → goto write_task ────────────────┘     │
        │  │  6. If APPROVE: continue           │  │     │
        │  └────────────────────────────────────┘  │     │
        └──────────────────────┬──────────────────────────┘
                               │ approved
                               ▼
                      ┌──────────────────┐
                      │     design       │
                      │ • Create HTML    │
                      │ • Save file      │
                      └──────────────────┘

Legend:
  ┌──────┐
  │ Task │  = Regular task
  └──────┘

  ┌─────────────────┐
  │ 🤖 LLMAgent    │  = Agent with tools (autonomous)
  └─────────────────┘

  →  = Data flow
  ↑↓ = Revision loop (agent-controlled)
```

## Agent Comparison

### Research Agent vs Search Task

**Traditional Search (Simple Workflow)**
```python
@task
def search_task(query: str) -> Dict:
    # Single search call
    sources = tavily.search(query)
    return {"sources": sources}
```

**Research Agent (Agent Workflow)**
```python
@task(inject_llm_agent="researcher")
def research_task(llm_agent: LLMAgent, query: str) -> Dict:
    # Agent autonomously:
    # - Searches multiple times
    # - Extracts relevant facts
    # - Refines queries based on findings
    # - Decides when to stop
    result = llm_agent.run("Research thoroughly: " + query)
    return result["output"]
```

**Key Difference**: Agent makes autonomous decisions about tool usage

---

### Editorial Agent vs Critique Task

**Traditional Critique (Simple Workflow)**
```python
@task(inject_llm_client=True)
def critique_task(llm: LLMClient, article: Dict) -> Dict:
    # Fixed prompt, single LLM call
    messages = [{"role": "user", "content": f"Critique: {article}"}]
    critique = llm.completion_text(messages)

    # Fixed logic for revision decision
    if "issues" in critique.lower():
        context.next_task(write_task, goto=True)

    return {"critique": critique}
```

**Editorial Agent (Agent Workflow)**
```python
@task(inject_llm_agent="editor")
def editorial_task(llm_agent: LLMAgent, article: Dict) -> Dict:
    # Agent autonomously:
    # - Checks facts with tool
    # - Assesses readability with tool
    # - Verifies sources with tool
    # - Decides approve/revise based on evidence
    # - Generates specific improvement suggestions
    result = llm_agent.run(
        f"Review article. Use tools to verify quality. "
        f"Decide: APPROVE or REVISE with specific feedback."
    )

    # Agent controls the loop decision
    if result["decision"] == "revise":
        context.next_task(write_task, goto=True)

    return result
```

**Key Difference**: Agent uses tools for verification, makes evidence-based decisions

---

## Tool Call Flow Example

### Research Agent in Action

```
User Query: "Latest developments in artificial intelligence"

┌─────────────────────────────────────────────────────────┐
│ Agent Reasoning (internal to agent)                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ Thought: I need to search for recent AI developments    │
│ Action: web_search("AI developments 2024")              │
│ Observation: Found 5 articles about LLMs and robotics   │
│                                                          │
│ Thought: Good start, but need more specific data        │
│ Action: extract_key_facts(results, "statistics")        │
│ Observation: Extracted market size, funding data        │
│                                                          │
│ Thought: Missing regulatory aspect                      │
│ Action: refine_search_query("AI", findings)             │
│ Observation: Suggested "AI regulation 2024"             │
│                                                          │
│ Thought: Let me search for regulations                  │
│ Action: web_search("AI regulation 2024")                │
│ Observation: Found policy updates, EU AI Act            │
│                                                          │
│ Thought: Now I have comprehensive coverage               │
│ Final Output: {                                          │
│   summary: "...",                                        │
│   sources: [...],                                        │
│   key_findings: [...]                                    │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
```

### Editorial Agent in Action

```
Article: "AI Market to Reach $500B by 2024"

┌─────────────────────────────────────────────────────────┐
│ Agent Reasoning (internal to agent)                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ Thought: Need to verify claims systematically            │
│ Action: check_factual_claims(article, sources)          │
│ Observation: "$500B" claim not found in sources         │
│                                                          │
│ Thought: Check if writing is clear                      │
│ Action: assess_readability(article_text)                │
│ Observation: Flesch score 45 (difficult), grade 12      │
│                                                          │
│ Thought: Verify source credibility                      │
│ Action: verify_sources(sources)                         │
│ Observation: 2/5 sources from credible domains          │
│                                                          │
│ Thought: Multiple issues found, need revision            │
│ Decision: REVISE                                         │
│ Action: suggest_improvements(article, issues)           │
│ Observation: Generated 4 specific suggestions            │
│                                                          │
│ Final Output: {                                          │
│   decision: "revise",                                    │
│   issues: ["unverified claim", "poor readability"],     │
│   suggestions: "1. Verify $500B claim...\n2. ..."       │
│ }                                                        │
│                                                          │
│ → Workflow: next_task(write_task, goto=True)            │
└─────────────────────────────────────────────────────────┘
```

---

## Channel State Flow

```
Channel State Progression:

topic_intake:
  channel.set("query", "AI developments")
  channel.set("iteration", 0)

research_agent:
  channel.set("research_summary", "...")
  channel.set("research_sources", [...])
  channel.set("image", "https://...")

curate:
  channel.set("curated", {structure, sources})

write (iteration 0):
  channel.set("article", {title, content, ...})
  channel.set("iteration", 0)

editorial_agent (decision: revise):
  article["editorial_feedback"] = {suggestions, issues}
  channel.set("article", article)
  channel.set("iteration", 1)
  → goto write_task

write (iteration 1):
  # Read article from channel (has editorial_feedback)
  # Incorporate suggestions
  channel.set("article", revised_article)
  channel.set("iteration", 1)

editorial_agent (decision: approve):
  article["editorial_feedback"] = None
  channel.set("article", article)
  → continue to design

design:
  article = channel.get("article")
  # Create HTML
```

---

## Langfuse Trace Structure

The agent workflow produces rich traces in Langfuse:

```
Trace: article_agent_article_0
├─ Span: topic_intake
│  └─ duration: 50ms
│
├─ Span: research (LLMAgent)
│  ├─ Span: LlmAgent.run
│  │  ├─ Generation: planning
│  │  ├─ Tool Call: web_search("AI developments")
│  │  │  └─ result: {...}
│  │  ├─ Generation: reasoning
│  │  ├─ Tool Call: extract_key_facts(sources, "statistics")
│  │  │  └─ result: {...}
│  │  ├─ Generation: refinement
│  │  ├─ Tool Call: web_search("AI regulation 2024")
│  │  │  └─ result: {...}
│  │  └─ Generation: final_output
│  └─ duration: 8.5s
│
├─ Span: curate
│  ├─ Generation: gpt-4o-mini completion
│  └─ duration: 2.1s
│
├─ Span: write (iteration 0)
│  ├─ Generation: gpt-4o-mini completion
│  └─ duration: 3.2s
│
├─ Span: editorial (LLMAgent)
│  ├─ Span: LlmAgent.run
│  │  ├─ Generation: planning
│  │  ├─ Tool Call: check_factual_claims(article, sources)
│  │  │  └─ result: {verified: 2, unverified: 1}
│  │  ├─ Tool Call: assess_readability(text)
│  │  │  └─ result: {flesch: 45, grade: 12}
│  │  ├─ Tool Call: verify_sources(sources)
│  │  │  └─ result: {credibility_rate: 0.4}
│  │  ├─ Generation: decision
│  │  ├─ Tool Call: suggest_improvements(article, issues)
│  │  │  └─ result: "1. Verify claim...\n2. ..."
│  │  └─ Generation: final_decision
│  └─ duration: 6.8s
│
├─ Span: write (iteration 1)  ← Revision
│  ├─ Generation: gpt-4o-mini completion (with feedback)
│  └─ duration: 3.5s
│
├─ Span: editorial (LLMAgent)  ← Second review
│  ├─ Span: LlmAgent.run
│  │  ├─ Tool Call: check_factual_claims(revised_article, sources)
│  │  │  └─ result: {verified: 3, unverified: 0}
│  │  ├─ Tool Call: assess_readability(revised_text)
│  │  │  └─ result: {flesch: 65, grade: 9}
│  │  └─ Generation: approve_decision
│  └─ duration: 4.2s
│
└─ Span: design
   └─ duration: 100ms

Total Duration: ~28s
Tool Calls: 8 (visible in trace)
Revisions: 1 (agent-driven)
```

**Benefits of this trace structure:**
- See exactly which tools were called and why
- Observe agent reasoning process
- Track revision iterations
- Measure time per agent decision
- Debug tool call failures
- Optimize expensive operations

---

## Cost Analysis

**Per Article (estimated)**

| Workflow | Input Tokens | Output Tokens | Tool Calls | Cost |
|----------|-------------|---------------|------------|------|
| Simple | 5,000 | 2,000 | 0 | $0.03 |
| Dynamic | 15,000 | 5,000 | 0 | $0.10 |
| **Agent** | 8,000 | 3,000 | **8** | **$0.06** |

Agent workflow is more cost-effective than Dynamic while providing better quality through autonomous verification.

---

## Key Takeaways

1. **Autonomy**: Agents decide tool usage, not hardcoded logic
2. **Transparency**: All tool calls visible in traces
3. **Quality**: Evidence-based decisions via tools
4. **Flexibility**: Easy to add new tools without changing workflow
5. **ReAct Pattern**: Plan → Act → Observe → Iterate (internal to agent)
6. **Agent-Controlled Loops**: Editorial agent decides revision, not fixed logic

---

## Next Steps

### Extend Research Agent
```python
# Add more tools
tools=[
    web_search,
    extract_key_facts,
    refine_search_query,
    check_source_date,  # NEW: Check if sources are recent
    compare_sources,    # NEW: Compare conflicting info
    summarize_topic     # NEW: Multi-source synthesis
]
```

### Extend Editorial Agent
```python
# Add more verification tools
tools=[
    check_factual_claims,
    assess_readability,
    verify_sources,
    suggest_improvements,
    check_bias,           # NEW: Detect biased language
    verify_statistics,    # NEW: Validate numerical claims
    check_citations       # NEW: Ensure proper attribution
]
```

### Add Coordinator Agent
```python
# Meta-agent that coordinates researcher and editor
coordinator_agent = LlmAgent(
    name="coordinator",
    model="gemini-2.0-flash-exp",
    tools=[
        assign_research_task,
        review_article_status,
        request_revision,
        approve_publication
    ]
)
```

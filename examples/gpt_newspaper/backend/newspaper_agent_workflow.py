"""
GPT Newspaper Agent Workflow with Graflow
==========================================

This example demonstrates a multi-agent newspaper workflow using graflow's
LLMAgent integration with Google ADK. It showcases:

1. LLMAgent with tool calling (researcher, editor)
2. Agent-driven autonomous decision making (ReAct pattern)
3. Mixed agent + simple LLM tasks in same workflow
4. Agent-controlled write-editorial revision loop
5. Real tools: Tavily search, textstat readability, fact verification
6. Configurable models via environment variables

Prerequisites:
--------------
Dependencies:
  - uv add google-adk textstat tavily-python

Environment Variables:
  - TAVILY_API_KEY: Tavily API key for web search (required)
  - GPT_NEWSPAPER_MODEL: Agent model identifier (optional, default: "gpt-4o-mini")
                        * Supports any LiteLLM-compatible chat model string
                        * Must match the provider credentials you have configured
  - GRAFLOW_LLM_MODEL: Model for simple LLM tasks (optional, default: "gpt-4o-mini")
                      * Supports any LiteLLM-compatible model
                      * Used for curate and write tasks (non-agent tasks)
                      * Examples: "gpt-4o", "claude-3-5-sonnet-20241022", "gpt-4o-mini"

Model Configuration:
--------------------
The workflow uses TWO separate model configurations:

1. **Agent Model** (GPT_NEWSPAPER_MODEL):
   - Used by: Research Agent, Editorial Agent (via Google ADK + LiteLLM bridge)
   - Supports: Any LiteLLM-compatible chat model (OpenAI, Anthropic, Gemini via litellm, etc.)
   - Examples: "gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-2.0-flash-exp"
   - Default: "gpt-4o-mini"

2. **LLM Model** (GRAFLOW_LLM_MODEL):
   - Used by: Curate task, Write task (via LiteLLM)
   - Supports: Any LiteLLM-compatible model
   - Examples: "gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022"
   - Default: "gpt-4o-mini"

This separation allows:
- Cost optimization: Use cheaper models for simple tasks (writing)
- Quality optimization: Use powerful models for agent reasoning (research, editorial)

Example Usage:
--------------
```bash
# Set API keys / model configs
export TAVILY_API_KEY="tvly-..."
export GPT_NEWSPAPER_MODEL="gpt-4o-mini"  # or another LiteLLM-compatible model

# Option 1: Use defaults (gpt-4o-mini for agents and LLM)
python newspaper_agent_workflow.py

# Option 2: Customize models
export GPT_NEWSPAPER_MODEL="gpt-4o"  # Better agent reasoning
export GRAFLOW_LLM_MODEL="gpt-4o"            # Better writing quality
python newspaper_agent_workflow.py

# Option 3: Use Claude for simple tasks (cost optimization)
export GPT_NEWSPAPER_MODEL="claude-3-5-sonnet-20241022"  # Agents via LiteLLM
export GRAFLOW_LLM_MODEL="claude-3-5-sonnet-20241022"  # Writing (any LiteLLM model)
python newspaper_agent_workflow.py
```

Expected Output:
----------------
Generated newspaper HTML with agent-driven research and editorial review.
Agents will display tool calls and reasoning in console output.

Architecture Differences:
-------------------------
- newspaper_workflow.py: Simple LLM tasks, basic critique loop
- newspaper_dynamic_workflow.py: Complex parallel tasks, quality gates
- newspaper_agent_workflow.py: **LLM Agents with tools, autonomous decisions, configurable models**
"""

import json
import logging
import os
import re
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List

from agents import DesignerAgent, PublisherAgent
from config import Config

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.llm.agents.base import LLMAgent as BaseLLMAgent
from graflow.trace.langfuse import LangFuseTracer

# Configure logging FIRST to capture all logs including imports
# This must be done BEFORE any graflow imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Enable DEBUG for ADK agent logger to see trace propagation logs
logging.getLogger("graflow.llm.agents.adk_agent").setLevel(logging.DEBUG)

# Set workflow logger level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep workflow logs at INFO to avoid clutter

if TYPE_CHECKING:
    import textstat  # type: ignore
    from tavily import TavilyClient  # type: ignore

# Check for optional dependencies availability
try:
    import tavily  # type: ignore # noqa: F401
    TAVILY_AVAILABLE = True
except ImportError as e:
    logger.warning("tavily-python is not installed. Web search will not be available.", exc_info=e)
    TAVILY_AVAILABLE = False

try:
    import textstat  # type: ignore
    TEXTSTAT_AVAILABLE = True
except ImportError as e:
    logger.warning("textstat is not installed. Readability assessment will not be available.", exc_info=e)
    TEXTSTAT_AVAILABLE = False

try:
    import google.adk.agents
    import google.adk.models.lite_llm  # noqa: F401

    import graflow.llm.agents.adk_agent
    import graflow.llm.agents.base  # noqa: F401
    from graflow.llm.agents.adk_agent import setup_adk_tracing

    ADK_AVAILABLE = True

    # Setup ADK tracing with threading instrumentation
    # This enables OpenTelemetry context propagation across threads
    setup_adk_tracing()
    logger.info("ADK tracing setup complete")
except ImportError as e:
    logger.warning("Google ADK is not installed. AdkLLMAgent will not be available.", exc_info=e)
    ADK_AVAILABLE = False


MAX_REVISION_ITERATIONS = 3
DEFAULT_IMAGE_URL = "https://images.unsplash.com/photo-1542281286-9e0a16bb7366"

# ============================================================================
# Tool Functions for Research Agent
# ============================================================================

def web_search(query: str) -> str:
    """
    Search the web using Tavily API.

    Args:
        query: Search query string

    Returns:
        JSON string with search results: {results: [...], image: "..."}
    """
    if not TAVILY_AVAILABLE:
        return json.dumps({"error": "Tavily not installed", "results": [], "image": DEFAULT_IMAGE_URL})

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return json.dumps({"error": "TAVILY_API_KEY not set", "results": [], "image": DEFAULT_IMAGE_URL})

    try:
        client = TavilyClient(api_key=api_key) # type: ignore
        results = client.search(
            query=query,
            topic="news",
            max_results=5,
            include_images=True
        )

        sources = results.get("results", [])
        images = results.get("images", [])
        image = images[0] if images else DEFAULT_IMAGE_URL

        # Format for agent consumption
        formatted_sources = []
        for src in sources:
            formatted_sources.append({
                "title": src.get("title", ""),
                "url": src.get("url", ""),
                "content": src.get("content", "")[:500],  # Truncate for context
                "score": src.get("score", 0.0)
            })

        return json.dumps({
            "results": formatted_sources,
            "image": image,
            "query": query
        })
    except Exception as e:
        return json.dumps({"error": str(e), "results": [], "image": DEFAULT_IMAGE_URL})


def extract_key_facts(sources_json: str, focus: str) -> str:
    """
    Extract key facts from search results focusing on a specific aspect.

    Args:
        sources_json: JSON string of search results from web_search
        focus: What aspect to focus on (e.g., "statistics", "expert opinions", "timeline")

    Returns:
        Extracted key facts as formatted text
    """
    try:
        data = json.loads(sources_json)
        sources = data.get("results", [])

        if not sources:
            return f"No sources available to extract facts about: {focus}"

        # Extract content from all sources
        all_content = []
        for src in sources:
            content = src.get("content", "")
            if content:
                all_content.append(f"From '{src.get('title', 'Unknown')}': {content}")

        combined = "\n\n".join(all_content)
        return f"Key facts focusing on '{focus}':\n\n{combined[:2000]}"  # Truncate

    except json.JSONDecodeError:
        return f"Error: Invalid sources data for fact extraction about {focus}"


def refine_search_query(original_query: str, findings: str) -> str:
    """
    Generate a refined search query based on current findings.

    Args:
        original_query: The original search query
        findings: Current research findings

    Returns:
        Suggested refined query
    """
    # Simple heuristic refinement (agent will use this suggestion)
    refinements = [
        f"{original_query} latest developments",
        f"{original_query} expert analysis",
        f"{original_query} statistics and data",
        f"{original_query} impact and implications"
    ]

    # Return suggestions as formatted text
    return "Suggested refined queries:\n" + "\n".join(f"- {q}" for q in refinements)


# ============================================================================
# Tool Functions for Editorial Agent
# ============================================================================

def check_factual_claims(article_json: str, sources_json: str) -> str:
    """
    Cross-reference article claims against source material.

    Args:
        article_json: JSON with article content
        sources_json: JSON with source materials

    Returns:
        JSON string with verification results
    """
    try:
        article = json.loads(article_json)
        sources = json.loads(sources_json)

        content = article.get("content", "")
        source_list = sources.get("results", [])

        # Simple keyword matching for fact verification
        # In production, this would use more sophisticated NLP
        claims_verified = []
        claims_unverified = []

        # Extract sentences as potential claims
        sentences = re.split(r'[.!?]+', content)
        for sent in sentences[:5]:  # Check first 5 sentences
            sent_stripped = sent.strip()
            if len(sent_stripped) < 20:
                continue

            # Check if any source content mentions similar keywords
            words = set(sent_stripped.lower().split())
            matched = False
            for src in source_list:
                src_content = src.get("content", "").lower()
                src_words = set(src_content.split())
                # Simple overlap check
                overlap = len(words & src_words)
                if overlap > 3:  # Arbitrary threshold
                    matched = True
                    claims_verified.append({
                        "claim": sent_stripped[:100],
                        "source": src.get("title", ""),
                        "confidence": "medium"
                    })
                    break

            if not matched and len(sent_stripped) > 30:
                claims_unverified.append(sent_stripped[:100])

        return json.dumps({
            "verified_claims": claims_verified[:3],
            "unverified_claims": claims_unverified[:3],
            "verification_rate": len(claims_verified) / max(len(claims_verified) + len(claims_unverified), 1)
        })

    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON data", "verified_claims": [], "unverified_claims": []})


def assess_readability(text: str) -> str:
    """
    Assess text readability using textstat library.

    Args:
        text: Article text to analyze

    Returns:
        JSON string with readability metrics
    """
    if not TEXTSTAT_AVAILABLE or textstat is None:
        return json.dumps({
            "error": "textstat not installed",
            "flesch_score": None,
            "grade_level": None
        })

    try:
        flesch_score = textstat.flesch_reading_ease(text)
        grade_level = textstat.flesch_kincaid_grade(text)
        word_count = textstat.lexicon_count(text)
        sentence_count = textstat.sentence_count(text)

        # Interpret Flesch score
        if flesch_score >= 90:
            interpretation = "Very easy to read"
        elif flesch_score >= 80:
            interpretation = "Easy to read"
        elif flesch_score >= 70:
            interpretation = "Fairly easy to read"
        elif flesch_score >= 60:
            interpretation = "Standard difficulty"
        elif flesch_score >= 50:
            interpretation = "Fairly difficult to read"
        else:
            interpretation = "Difficult to read"

        return json.dumps({
            "flesch_reading_ease": round(flesch_score, 1),
            "interpretation": interpretation,
            "grade_level": round(grade_level, 1),
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": round(word_count / max(sentence_count, 1), 1)
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


def verify_sources(sources_json: str) -> str:
    """
    Verify source quality and credibility.

    Args:
        sources_json: JSON string with source data

    Returns:
        JSON string with source verification results
    """
    try:
        data = json.loads(sources_json)
        sources = data.get("results", [])

        verified_sources = []
        flagged_sources = []

        for src in sources:
            url = src.get("url", "")
            title = src.get("title", "")
            score = src.get("score", 0.0)

            # Simple heuristics for source credibility
            issues = []

            # Check for known credible domains (simplified)
            credible_domains = [
                "nytimes.com", "washingtonpost.com", "bbc.com", "reuters.com",
                "apnews.com", "theguardian.com", "wsj.com", "npr.org"
            ]

            domain_found = any(domain in url for domain in credible_domains)

            if score < 0.5:
                issues.append("Low relevance score")

            if not domain_found:
                issues.append("Domain not in known credible list")

            if len(title) < 10:
                issues.append("Title too short")

            if issues:
                # Source has credibility issues
                flagged_sources.append({
                    "url": url,
                    "title": title,
                    "issues": issues
                })
            else:
                # Source is credible
                verified_sources.append({
                    "url": url,
                    "title": title,
                    "score": score
                })

        return json.dumps({
            "verified_sources": len(verified_sources),
            "flagged_sources": len(flagged_sources),
            "credibility_rate": len(verified_sources) / max(len(sources), 1),
            "flagged_details": flagged_sources[:2]  # Show first 2 flagged
        })

    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid sources JSON"})


def suggest_improvements(article_json: str, issues: str) -> str:
    """
    Generate specific improvement suggestions based on identified issues.

    Args:
        article_json: JSON with article data
        issues: Description of issues found

    Returns:
        Structured improvement suggestions
    """
    try:
        article = json.loads(article_json)
        content = article.get("content", "")

        suggestions = []

        if "readability" in issues.lower():
            suggestions.append("Break long sentences into shorter ones for better readability")
            suggestions.append("Use simpler vocabulary where possible")

        if "fact" in issues.lower() or "claim" in issues.lower():
            suggestions.append("Add specific statistics or data points from sources")
            suggestions.append("Include direct quotes from credible sources")

        if "source" in issues.lower():
            suggestions.append("Add citations to reputable news organizations")
            suggestions.append("Include links or references to original sources")

        if len(content) < 500:
            suggestions.append("Expand the article with more details and context")

        if len(content) > 2000:
            suggestions.append("Consider condensing to focus on key points")

        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions))

    except json.JSONDecodeError:
        return "Unable to parse article data for suggestions"


# ============================================================================
# Workflow Definition
# ============================================================================

def create_article_workflow(query: str, article_id: str, output_dir: str, tracer=None):
    """
    Create an agent-driven workflow for article generation.

    This workflow demonstrates:
    - Research agent with autonomous multi-search capability
    - Editorial agent with fact-checking and quality assessment tools
    - Agent-controlled revision loop based on editorial decisions
    - Mix of agent tasks and simple LLM tasks

    Args:
        query: Article topic/query
        article_id: Unique identifier for this article
        output_dir: Output directory for article HTML
        tracer: Optional tracer for workflow execution tracking

    Returns:
        Workflow context
    """
    if not ADK_AVAILABLE:
        raise RuntimeError(
            "Google ADK not installed. Install with: uv add google-adk\n"
            "This workflow requires LLMAgent support."
        )

    # Initialize non-LLM agents
    designer_agent = DesignerAgent(output_dir)

    with workflow(f"article_agent_{article_id}", tracer=tracer) as wf:

        # Helper functions
        def _safe_get_result(context: TaskExecutionContext, task_id: str) -> Dict[str, Any] | None:
            try:
                return context.get_result(task_id)
            except (KeyError, ValueError):
                return None

        # ====================================================================
        # Agent Registration
        # ====================================================================

        def register_agents(exec_context: TaskExecutionContext):
            """Register LLM agents with specialized tools.

            Uses Config.AGENT_MODEL (GPT_NEWSPAPER_MODEL env var) for agent model.
            Defaults to "gpt-4o-mini" if not set.
            """
            if not ADK_AVAILABLE:
                raise RuntimeError("Google ADK not available")

            from google.adk.agents import LlmAgent
            from google.adk.models.lite_llm import LiteLlm

            from graflow.llm.agents.adk_agent import AdkLLMAgent

            configured_model = Config.AGENT_MODEL
            agent_model = LiteLlm(model=configured_model)

            print(f"[{article_id}] 🤖 Agent model: {configured_model}")

            # Research Agent - Autonomous researcher with search tools
            research_agent = LlmAgent(
                name="researcher",
                model=agent_model,
                tools=[web_search, extract_key_facts, refine_search_query]
            )
            wrapped_researcher = AdkLLMAgent(
                research_agent,
                app_name=exec_context.session_id
            )
            exec_context.register_llm_agent("researcher", wrapped_researcher)  # type: ignore[attr-defined]
            print(f"[{article_id}] ✅ Researcher agent registered (tools: web_search, extract_key_facts, refine_search_query)")

            # Editorial Agent - Quality control with verification tools
            editorial_agent = LlmAgent(
                name="editor",
                model=agent_model,
                tools=[check_factual_claims, assess_readability, verify_sources, suggest_improvements]
            )
            wrapped_editor = AdkLLMAgent(
                editorial_agent,
                app_name=exec_context.session_id
            )
            exec_context.register_llm_agent("editor", wrapped_editor)  # type: ignore[attr-defined]
            print(f"[{article_id}] ✅ Editorial agent registered (tools: check_factual_claims, assess_readability, verify_sources, suggest_improvements)")

        # ====================================================================
        # Task Definitions
        # ====================================================================

        @task(id=f"topic_intake_{article_id}", inject_context=True)
        def topic_intake_task(context: TaskExecutionContext) -> Dict:
            """Initialize workflow with query and setup channel state."""
            print(f"\n[{article_id}] 📋 Topic intake: {query}")

            # Register agents at workflow start
            register_agents(context)

            # Setup channel state
            channel = context.get_channel()
            channel.set("query", query)
            channel.set("iteration", 0)
            channel.set("research_sources", [])
            channel.set("image", None)

            return {"query": query, "article_id": article_id}

        @task(id=f"research_{article_id}", inject_context=True, inject_llm_agent="researcher")
        def research_task(context: TaskExecutionContext, llm_agent: BaseLLMAgent) -> Dict:
            """
            Research agent autonomously searches and gathers information.

            The agent uses ReAct pattern:
            1. Initial web_search for query
            2. Analyze results, identify gaps
            3. Use extract_key_facts to focus on specific aspects
            4. Optionally refine_search_query for follow-up searches
            5. Compile comprehensive research report
            """
            print(f"\n[{article_id}] 🔍 Research Agent starting (autonomous multi-search)...")

            # Note: trace_id is set during agent initialization (via app_name), not passed at runtime
            result = llm_agent.run(
                f"Research this topic thoroughly: '{query}'\n\n"
                f"Instructions:\n"
                f"1. Use web_search to find recent, credible sources\n"
                f"2. Use extract_key_facts to pull out important information\n"
                f"3. If you find gaps, use refine_search_query to create follow-up queries and search again\n"
                f"4. Aim for 3-5 high-quality sources\n"
                f"5. Provide a structured summary with key findings and sources\n\n"
                f"Focus on: factual information, recent developments, expert perspectives, data/statistics.",
                trace_id=context.trace_id,
                session_id=context.session_id
            )

            # Extract agent output
            output = result.get("output", {}) if isinstance(result, dict) else {}
            if isinstance(output, str):
                # Agent returned text summary
                research_summary = output
                sources = []
                image = DEFAULT_IMAGE_URL
            else:
                research_summary = output.get("summary", str(output))
                sources = output.get("sources", [])
                image = output.get("image", DEFAULT_IMAGE_URL)

            # Store in channel for downstream tasks
            channel = context.get_channel()
            channel.set("research_summary", research_summary)
            channel.set("research_sources", sources)
            channel.set("image", image)

            print(f"[{article_id}] ✅ Research complete: {len(research_summary)} chars summary, {len(sources)} sources")

            return {
                "summary": research_summary,
                "sources": sources,
                "image": image,
                "query": query
            }

        @task(id=f"curate_{article_id}", inject_context=True)
        def curate_task(context: TaskExecutionContext) -> Dict:
            """
            Organize research into article structure.

            Simple LLM task (no tools needed) - just structure the content.
            Uses Config.DEFAULT_MODEL (GRAFLOW_LLM_MODEL env var).
            """
            print(f"[{article_id}] 📝 Curating research into article structure...")
            llm = context.llm_client

            research = _safe_get_result(context, f"research_{article_id}")
            if not research:
                research = {
                    "summary": context.get_channel().get("research_summary", ""),
                    "sources": context.get_channel().get("research_sources", []),
                    "image": context.get_channel().get("image", DEFAULT_IMAGE_URL)
                }

            messages = [
                {
                    "role": "system",
                    "content": "You organize research into a clear article structure with sections."
                },
                {
                    "role": "user",
                    "content": f"Organize this research into article sections:\n\n{research['summary']}\n\n"
                               f"Create: 1) Headline, 2) Lead paragraph, 3) 3-4 body sections with headings"
                }
            ]

            structured = llm.completion_text(messages, model=Config.DEFAULT_MODEL)

            result = {
                "structure": structured,
                "sources": research.get("sources", []),
                "image": research.get("image", DEFAULT_IMAGE_URL),
                "query": query
            }

            context.get_channel().set("curated", result)
            print(f"[{article_id}] ✅ Article structure created")

            return result

        @task(id=f"write_{article_id}", inject_context=True)
        def write_task(context: TaskExecutionContext) -> Dict:
            """
            Write or revise article based on editorial feedback.

            Simple LLM task for content generation (no tools needed).
            Incorporates editorial suggestions if in revision mode.
            Uses Config.DEFAULT_MODEL (GRAFLOW_LLM_MODEL env var).
            """
            channel = context.get_channel()
            article = channel.get("article")
            iteration = channel.get("iteration", default=0)
            llm = context.llm_client

            # Initialize curated for type checker
            curated: Dict[str, Any] | None = None

            if article and article.get("editorial_feedback"):
                # Revision mode - incorporate editorial feedback
                feedback = article["editorial_feedback"]
                print(f"\n[{article_id}] 📝 Revising article (iteration {iteration})...")
                print(f"[{article_id}]   Addressing: {len(feedback.get('suggestions', ''))} chars of feedback")

                prompt = (
                    f"Revise this article addressing the editorial feedback:\n\n"
                    f"Original article:\n{article.get('content', '')}\n\n"
                    f"Editorial feedback:\n{feedback.get('suggestions', '')}\n\n"
                    f"Issues to fix:\n{feedback.get('issues', '')}\n\n"
                    f"Write improved version addressing all feedback."
                )
            else:
                # Initial draft mode
                print(f"\n[{article_id}] ✍️  Writing initial article draft...")
                curated = _safe_get_result(context, f"curate_{article_id}")
                if not curated:
                    curated = channel.get("curated", {})
                assert curated is not None, "Curated article structure missing for writing task"
                prompt = (
                    f"Write a newspaper article based on this structure:\n\n{curated.get('structure', '')}\n\n"
                    f"Requirements:\n"
                    f"- Professional journalism style\n"
                    f"- 3-5 paragraphs\n"
                    f"- Factual and objective tone\n"
                    f"- Include relevant details from research"
                )

            messages = [{"role": "user", "content": prompt}]
            content = llm.completion_text(messages, model=Config.DEFAULT_MODEL, max_tokens=1500)

            # Extract title (first line or generate)
            lines = content.strip().split("\n")
            title = lines[0] if lines else f"Article: {query}"
            if title.startswith("#"):
                title = title.lstrip("#").strip()

            # Format article
            result = {
                "title": title,
                "content": content,
                "date": time.strftime("%B %d, %Y"),
                "query": query,
                "sources": curated.get("sources", []) if curated else [],
                "image": channel.get("image", DEFAULT_IMAGE_URL),
                "paragraphs": [p.strip() for p in content.split("\n\n") if p.strip()][:5]
            }

            channel.set("article", result)
            channel.set("iteration", iteration)

            print(f"[{article_id}] ✅ Article written: '{title[:50]}...'")

            return result

        @task(id=f"editorial_{article_id}", inject_context=True, inject_llm_agent="editor")
        def editorial_task(context: TaskExecutionContext, llm_agent: "BaseLLMAgent") -> Dict:
            """
            Editorial agent reviews article and decides: approve or revise.

            The agent uses tools to:
            1. Check factual claims against sources (check_factual_claims)
            2. Assess readability metrics (assess_readability)
            3. Verify source credibility (verify_sources)
            4. Generate improvement suggestions (suggest_improvements)

            Agent makes autonomous decision: APPROVE or REVISE with specific feedback.
            """
            print(f"\n[{article_id}] 🔎 Editorial Agent reviewing article...")

            channel = context.get_channel()
            article: Dict[str, Any] | None = channel.get("article")
            iteration = channel.get("iteration", default=0)

            if not article:
                article = _safe_get_result(context, f"write_{article_id}")

            if not article:
                raise ValueError(f"Article data missing for editorial task: {article_id}")

            # Prepare article and sources as JSON for tools
            article_json = json.dumps({
                "title": article.get("title", ""),
                "content": article.get("content", ""),
                "query": query
            })

            sources_json = json.dumps({
                "results": article.get("sources", [])
            })

            review_prompt = (
                "You are the lead editorial agent for GPT Newspaper.\n"
                "Use your registered tools to verify accuracy, readability, and source quality.\n"
                "Return ONLY a JSON object with the following structure:\n"
                "{\n"
                '  "decision": "approve" | "revise",\n'
                '  "issues": "Bullet list of issues found",\n'
                '  "suggestions": "Concrete edits to fix the issues",\n'
                '  "confidence": "0-1 float score"\n'
                "}\n\n"
                "Decision rules:\n"
                "- APPROVE only if claims are verified AND readability is acceptable.\n"
                "- Otherwise REVISE with specific evidence-backed feedback."
            )

            agent_input = (
                f"{review_prompt}\n\n"
                f"Article JSON:\n{article_json}\n\n"
                f"Sources JSON:\n{sources_json}"
            )

            # Agent performs editorial review with tools (per docs/llm_integration_design.md)
            # Note: trace_id is set during agent initialization (via app_name), not passed at runtime
            result = llm_agent.run(agent_input)

            raw_output = ""
            if isinstance(result, dict):
                raw_output = str(result.get("output", ""))
            else:
                raw_output = str(result)

            decision = "revise"
            issues = ""
            suggestions = ""
            confidence = None

            try:
                parsed = json.loads(raw_output)
                decision = parsed.get("decision", "revise").lower()
                issues = parsed.get("issues", "")
                suggestions = parsed.get("suggestions", "")
                confidence = parsed.get("confidence")
            except json.JSONDecodeError:
                lowered = raw_output.lower()
                decision = "approve" if "approve" in lowered and "revise" not in lowered else "revise"
                issues = raw_output
                suggestions = raw_output

            editorial_result = {
                "decision": decision,
                "suggestions": suggestions,
                "issues": issues,
                "iteration": iteration,
                "confidence": confidence,
                "raw_output": raw_output
            }

            print(f"[{article_id}] 📊 Editorial decision: {decision.upper()}")

            # Decide whether to loop back for revision
            if decision == "revise" and iteration < MAX_REVISION_ITERATIONS:
                print(f"[{article_id}] 🔄 Revision needed, looping back to writer (iteration {iteration + 1}/{MAX_REVISION_ITERATIONS})")

                # Store feedback in article
                article["editorial_feedback"] = editorial_result
                channel.set("article", article)
                channel.set("iteration", iteration + 1)

                # Jump back to write task (goto=True)
                context.next_task(write_task, goto=True)
            elif decision == "revise" and iteration >= MAX_REVISION_ITERATIONS:
                print(f"[{article_id}] ⚠️  Max iterations reached, proceeding to publication")
                article["editorial_feedback"] = None
                channel.set("article", article)
            else:
                # Approved!
                print(f"[{article_id}] ✅ Article approved for publication")
                article["editorial_feedback"] = None
                channel.set("article", article)

            return editorial_result

        @task(id=f"design_{article_id}", inject_context=True)
        def design_task(context: TaskExecutionContext) -> Dict:
            """Design the article HTML layout."""
            print(f"[{article_id}] 🎨 Designing article layout...")

            channel = context.get_channel()
            article = channel.get("article")
            if not article:
                article = _safe_get_result(context, f"write_{article_id}")

            if not article:
                raise ValueError(f"Article data missing for design task: {article_id}")

            design_result = designer_agent.run(article)
            print(f"[{article_id}] ✅ Design complete: {design_result.get('path')}")

            return design_result

        # Build workflow graph
        # Linear flow: intake >> research >> curate >> write >> editorial >> design
        # Editorial agent controls loop-back to write task
        topic_intake_task >> research_task >> curate_task >> write_task >> editorial_task >> design_task # type: ignore

        return wf


def execute_article_workflow(query: str, article_id: str, output_dir: str) -> Dict:
    """
    Execute a single article workflow with agent-driven research and editorial.

    Args:
        query: Article query/topic
        article_id: Unique article identifier
        output_dir: Output directory for article HTML

    Returns:
        Completed article dict with design
    """
    print(f"\n{'=' * 80}")
    print(f"Processing Article (Agent Workflow): {query}")
    print(f"{'=' * 80}")

    # Create tracer for workflow observability
    try:
        tracer = LangFuseTracer(enable_runtime_graph=True)
        print(f"[{article_id}] LangFuse tracer initialized")
    except Exception as e:
        print(f"[{article_id}] Warning: LangFuse tracer initialization failed: {e}")
        print(f"[{article_id}] Continuing without tracing...")
        tracer = None

    # Create and execute workflow
    wf = create_article_workflow(query, article_id, output_dir, tracer=tracer)
    result = wf.execute(
        f"topic_intake_{article_id}",
        max_steps=20  # Allow for write-editorial cycles
    )

    # Shutdown tracer to flush remaining traces
    if tracer:
        tracer.shutdown()

    if isinstance(result, dict):
        return result

    raise ValueError(f"❌ Design task not found for {article_id}")


def run_newspaper_workflow(
    queries: List[str],
    layout: str = "layout_1.html",
    max_workers: int | None = None,
    output_dir: str | None = None,
):
    """
    Run the agent-driven newspaper workflow.

    Args:
        queries: List of article queries/topics
        layout: Newspaper layout template
        max_workers: Maximum number of parallel workers (currently ignored for agent workflow,
                    as agents process sequentially for better visibility)
        output_dir: Output directory (default: outputs/run_<timestamp>)

    Note:
        Unlike the dynamic workflow, the agent workflow processes articles sequentially
        to provide clearer visibility into agent reasoning and tool calls. The max_workers
        parameter is accepted for API compatibility but not used.
    """
    print("=" * 80)
    print("🗞️  GPT NEWSPAPER AGENT WORKFLOW")
    print("=" * 80)

    # Check dependencies
    if not ADK_AVAILABLE:
        error_msg = "Google ADK not installed. Install with: uv add google-adk"
        print(f"❌ Error: {error_msg}")
        raise RuntimeError(error_msg)

    if not TAVILY_AVAILABLE:
        print("⚠️  Warning: Tavily not installed (web search disabled)")

    if not TEXTSTAT_AVAILABLE:
        print("⚠️  Warning: textstat not installed (readability assessment disabled)")

    # Check API keys
    if not os.getenv("TAVILY_API_KEY"):
        error_msg = "TAVILY_API_KEY environment variable is required. Get your API key at: https://tavily.com/"
        print(f"❌ Error: {error_msg}")
        raise RuntimeError(error_msg)

    # Create output directory
    output_dir = output_dir or f"outputs/run_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}\n")

    # Process articles sequentially (for clearer agent interaction visibility)
    # In production, could parallelize with ThreadPoolExecutor
    completed_articles = []
    for i, query in enumerate(queries):
        article_id = f"article_{i}"
        try:
            result = execute_article_workflow(query, article_id, output_dir)
            completed_articles.append(result)
        except Exception as e:
            print(f"\n❌ Error processing article {article_id}: {e}")
            import traceback
            traceback.print_exc()

    if not completed_articles:
        error_msg = "No articles completed successfully"
        print(f"\n❌ {error_msg}")
        raise RuntimeError(error_msg)

    # Compile newspaper
    print(f"\n{'=' * 80}")
    print("📰 Compiling Newspaper")
    print(f"{'=' * 80}\n")

    from agents import EditorAgent
    editor_agent = EditorAgent(layout)
    newspaper_html = editor_agent.run(completed_articles)

    # Publish newspaper
    publisher_agent = PublisherAgent(output_dir)
    newspaper_path = publisher_agent.run(newspaper_html)

    print(f"\n{'=' * 80}")
    print("✨ NEWSPAPER GENERATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n📰 Newspaper: {newspaper_path}")
    print(f"📄 Articles: {len(completed_articles)}")
    print(f"\nOpen {newspaper_path} in your browser to view!\n")

    return newspaper_path


def main():
    """Run the agent-driven newspaper workflow with example queries.

    Environment Variables:
        GPT_NEWSPAPER_MODEL: Model for agent workflows (default: "gpt-4o-mini")
                            - Supports any LiteLLM-compatible chat model string
                            - Must align with provider API keys configured for LiteLLM

        GRAFLOW_LLM_MODEL: Model for simple LLM tasks (default: "gpt-4o-mini")
                          - Supports any LiteLLM-compatible model
                          - Used for curate and write tasks (non-agent tasks)

        TAVILY_API_KEY: Tavily API key for web search

    Example:
        ```bash
        # Use specific models
        export GPT_NEWSPAPER_MODEL="gpt-4o"
        export GRAFLOW_LLM_MODEL="gpt-4o"
        export TAVILY_API_KEY="tvly-..."

        python newspaper_agent_workflow.py
        ```
    """
    # Display configuration
    Config.display()

    # Validate configuration
    if not Config.validate():
        return

    # Check for required environment variables
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ Error: TAVILY_API_KEY environment variable is required")
        print("Get your API key at: https://tavily.com/")
        return

    # Example queries
    queries = [
        "Latest developments in artificial intelligence",
        "Climate change policy updates",
    ]

    # Run workflow
    run_newspaper_workflow(
        queries=queries,
        layout="layout_1.html"
    )


if __name__ == "__main__":
    main()


# ============================================================================
# Key Patterns Demonstrated:
# ============================================================================
#
# 1. **LLMAgent with Real Tools**
#    - Research agent: web_search (Tavily), extract_key_facts, refine_search_query
#    - Editorial agent: check_factual_claims, assess_readability, verify_sources
#    - Tools are real Python functions, not simulated
#
# 2. **Agent Registration in Workflow**
#    def register_agents(exec_context):
#        researcher = AdkLLMAgent(LlmAgent(name="researcher", tools=[...]))
#        exec_context.register_llm_agent("researcher", researcher)
#
#    @task(inject_llm_agent="researcher")
#    def research_task(llm_agent: LLMAgent):
#        result = llm_agent.run("Research...")
#
# 3. **Agent-Controlled Revision Loop**
#    - Editorial agent decides: approve or revise
#    - If revise: context.next_task(write_task, goto=True)
#    - Agent provides structured feedback for revision
#
# 4. **Mixed Agent + Simple LLM Tasks**
#    - Research: LLMAgent with tools (autonomous search)
#    - Curate: Simple LLM (structure content)
#    - Write: Simple LLM (generate text)
#    - Editorial: LLMAgent with tools (verify + decide)
#
# 5. **ReAct Pattern**
#    - Agent receives task with tool descriptions
#    - Agent reasons: what tools to use, in what order
#    - Agent acts: calls tools, observes results
#    - Agent iterates: refines approach based on observations
#
# 6. **Tool Design Principles**
#    - Each tool has single, clear purpose
#    - Tools return structured data (JSON strings)
#    - Tools handle errors gracefully
#    - Tools are testable independently
#
# ============================================================================
# Comparison with Other Workflows:
# ============================================================================
#
# newspaper_workflow.py:
# - Simple LLM tasks throughout
# - Basic critique loop (fixed logic)
# - Single search, single write
# - Good for: Simple pipelines, cost optimization
#
# newspaper_dynamic_workflow.py:
# - Complex parallel execution (3 writers, 3 quality gates)
# - Dynamic task fan-out (multiple search angles)
# - Rich coordination patterns
# - Good for: Complex workflows, parallel processing
#
# newspaper_agent_workflow.py (THIS FILE):
# - **LLM Agents with autonomous decision-making**
# - **Real tool calling (Tavily, textstat)**
# - **Agent-controlled revision loop**
# - Good for: Autonomous workflows, quality-critical tasks
#
# ============================================================================

"""Critique Agent - Provides feedback on articles"""

import json
from datetime import datetime
from typing import Dict, List, Optional

from graflow.llm.client import LLMClient, make_message


def _format_sources_for_review(sources: List[Dict]) -> str:
    """Format source articles for cross-reference review."""
    lines: list[str] = []
    for i, src in enumerate(sources, 1):
        title = src.get("title", "Untitled")
        url = src.get("url", "")
        content = (src.get("content") or "")[:600]
        published = src.get("published_date", "unknown date")
        lines.append(f"[Source {i}] {title}\n  URL: {url}\n  Published: {published}\n  Excerpt: {content}")
    return "\n\n".join(lines)


class CritiqueAgent:
    """Agent that provides critique feedback on written articles."""

    def __init__(self, llm_client: LLMClient, model: Optional[str] = None):
        """Initialize CritiqueAgent with injected LLMClient.

        Args:
            llm_client: Injected LLMClient instance
            model: Optional model override for this agent's completions
        """
        self.client = llm_client
        self.model = model  # Optional model override

    def critique_article(self, article: Dict) -> Optional[str]:
        """
        Provide critique feedback on an article.

        Args:
            article: Article dict

        Returns:
            Critique feedback string or None if article is acceptable
        """
        title = article.get("title", "Untitled")
        summary = article.get("summary", "")
        paragraphs = article.get("paragraphs", [])
        sources = article.get("sources", [])
        revision_message = article.get("message")

        sources_text = _format_sources_for_review(sources) if sources else "(no sources available)"

        revision_note = ""
        if revision_message:
            revision_note = f"""
--- WRITER'S REVISION NOTE ---
The writer has revised the article based on your previous critique.
Writer's message: {revision_message}
--- END REVISION NOTE ---
"""

        prompt = f"""Today's date is {datetime.now().strftime("%d/%m/%Y")}.

--- ARTICLE ---
Title: {title}

Summary: {summary}

Paragraphs:
{json.dumps(paragraphs, ensure_ascii=False, indent=2)}
--- END ARTICLE ---

--- SOURCE MATERIAL ---
{sources_text}
--- END SOURCE MATERIAL ---
{revision_note}
Your task is to review the article and provide feedback. Focus on the following:

1. **Factual accuracy**: Cross-reference dates, numbers, names, and events mentioned in the article against the source material above. Flag any dates or facts that are not supported by the sources or appear incorrect.
2. **Temporal consistency**: Verify that the timeline of events is accurate. Check that "yesterday", "last week", specific dates, etc. are consistent with today's date ({datetime.now().strftime("%d/%m/%Y")}) and the source publication dates.
3. **Attribution**: Ensure claims are properly attributed to sources. Flag any unsupported assertions.
4. **Writing quality**: Check for clarity, coherence, and readability.

If the article is factually accurate, well-written, and properly sourced, return exactly the word "None".
Otherwise, provide concise, actionable feedback listing specific issues to fix.

Please return a string of your critique or the word "None".
"""

        feedback = self.client.completion_text(
            messages=[
                make_message(
                    "system",
                    "You are a rigorous newspaper fact-checker and writing critic. You verify factual claims, dates, and temporal information against source material, and provide actionable feedback to improve article accuracy and quality.",
                ),
                make_message("user", prompt),
            ],
            model=self.model,
        ).strip()

        if feedback.lower() == "none" or not feedback:
            return None

        print(f"For article: {article.get('title', 'Unknown')}")
        print(f"Critique Feedback: {feedback}\n")

        return feedback

    def run(self, article: Dict) -> Dict:
        """
        Run the critique agent.

        Args:
            article: Article dict

        Returns:
            Updated article dict with 'critique' field
        """
        critique = self.critique_article(article)

        article["critique"] = critique
        # Clear message field if critique is provided
        if critique is not None:
            article["message"] = None

        return article

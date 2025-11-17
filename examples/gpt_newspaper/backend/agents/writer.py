"""Writer Agent - Writes news articles"""

import json
from datetime import datetime
from typing import Dict, Optional

from graflow.llm.client import LLMClient, make_message

ARTICLE_JSON_FORMAT = """
{
  "title": "title of the article",
  "date": "today's date",
  "paragraphs": [
    "paragraph 1",
    "paragraph 2",
    "paragraph 3",
    "paragraph 4",
    "paragraph 5"
  ],
  "summary": "2 sentences summary of the article"
}
"""

REVISE_JSON_FORMAT = """
{
  "paragraphs": [
    "paragraph 1",
    "paragraph 2",
    "paragraph 3",
    "paragraph 4",
    "paragraph 5"
  ],
  "message": "message to the critique explaining changes"
}
"""


class WriterAgent:
    """Agent that writes and revises news articles."""

    def __init__(self, llm_client: LLMClient, model: Optional[str] = None):
        """Initialize WriterAgent with injected LLMClient.

        Args:
            llm_client: Injected LLMClient instance
            model: Optional model override for this agent's completions
        """
        self.client = llm_client
        self.model = model  # Optional model override

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove Markdown code fences from a response when present."""
        stripped = text.strip()
        if stripped.startswith("```"):
            fence_end = stripped.find("\n")
            if fence_end != -1:
                stripped = stripped[fence_end + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        return stripped.strip()

    def _parse_json_response(self, raw_text: str | None, *, context: str) -> Dict:
        """Parse JSON content returned by the model with helpful errors."""
        if raw_text is None:
            raise RuntimeError(
                f"{context}: model '{self.model}' returned no content. "
                "Ensure the model response_format supports JSON output."
            )

        cleaned = self._strip_code_fence(raw_text)
        if not cleaned:
            raise RuntimeError(
                f"{context}: model '{self.model}' returned an empty response. "
                "Check LLM configuration or try a different model."
            )

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            preview = cleaned.strip().replace("\n", " ")[:200]
            raise RuntimeError(
                f"{context}: expected JSON response from model '{self.model}', "
                f"but received: '{preview}...'"
            ) from exc

    def write_article(self, query: str, sources: list) -> Dict:
        """
        Write a new article based on query and sources.

        Args:
            query: The article topic/query
            sources: List of source articles

        Returns:
            Article dict with title, date, paragraphs, summary
        """
        prompt = f"""Today's date is {datetime.now().strftime('%d/%m/%Y')}.

Query or Topic: {query}

Sources:
{sources}

Your task is to write a critically acclaimed article about the provided query or topic based on the sources.

Please return nothing but a JSON in the following format:
{ARTICLE_JSON_FORMAT}
"""

        response_text = self.client.completion_text(
            messages=[
                make_message(
                    "system",
                    "You are a newspaper writer. Your sole purpose is to write a well-written article about a topic using a list of articles.",
                ),
                make_message("user", prompt),
            ],
            model=self.model,
            response_format={"type": "json_object"},
        )

        return self._parse_json_response(response_text, context="WriterAgent.write_article")

    def revise_article(self, article: Dict) -> Dict:
        """
        Revise an article based on critique feedback.

        Args:
            article: Article dict with 'critique' field

        Returns:
            Dict with revised 'paragraphs' and 'message' to critique
        """
        prompt = f"""Article:
{article}

Your task is to edit the article based on the critique given.

Please return json format with the revised 'paragraphs' and a new 'message' field to the critique that explains your changes or why you didn't change anything.

Please return nothing but a JSON in the following format:
{REVISE_JSON_FORMAT}
"""

        response_text = self.client.completion_text(
            messages=[
                make_message(
                    "system",
                    "You are a newspaper editor. Your sole purpose is to edit a well-written article based on given critique.",
                ),
                make_message("user", prompt),
            ],
            model=self.model,
            response_format={"type": "json_object"},
        )

        revision = self._parse_json_response(response_text, context="WriterAgent.revise_article")

        print(f"For article: {article.get('title', 'Unknown')}")
        print(f"Writer Revision Message: {revision.get('message', '')}\n")

        return revision

    def run(self, article: Dict) -> Dict:
        """
        Run the writer agent - either write new article or revise based on critique.

        Args:
            article: Article dict

        Returns:
            Updated article dict
        """
        critique = article.get("critique")

        if critique is not None:
            # Revise existing article
            revision = self.revise_article(article)
            article.update(revision)
        else:
            # Write new article
            new_content = self.write_article(article["query"], article["sources"])
            article.update(new_content)

        return article

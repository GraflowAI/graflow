"""Curator Agent - Selects most relevant articles"""

import json
from datetime import datetime
from typing import Any, Dict, List

from config import Config
from utils.litellm import LiteLLMClient, make_message


class CuratorAgent:
    """Agent that curates the most relevant sources for an article."""

    def __init__(self, model: str | None = None, *, client_params: Dict[str, Any] | None = None):
        self.model = model or Config.DEFAULT_MODEL
        params = dict(Config.DEFAULT_MODEL_PARAMS)
        if client_params:
            params.update(client_params)
        self.client = LiteLLMClient(self.model, **params)

    def curate_sources(self, query: str, sources: List[Dict]) -> List[Dict]:
        """
        Curate the 5 most relevant sources for the query.

        Args:
            query: The search query
            sources: List of source dictionaries

        Returns:
            Filtered list of 5 most relevant sources
        """
        prompt = f"""Today's date is {datetime.now().strftime('%d/%m/%Y')}.

Topic or Query: {query}

Your task is to return the 5 most relevant articles for the provided topic or query.

Here is a list of articles:
{sources}

Please return nothing but a list of the URLs as a JSON array: ["url1", "url2", "url3", "url4", "url5"]
"""

        response_text = self.client.chat_text(
            [
                make_message(
                    "system",
                    "You are a personal newspaper editor. Your sole purpose is to choose 5 most relevant articles for me to read from a list of articles.",
                ),
                make_message("user", prompt),
            ]
        )

        try:
            chosen_urls = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: extract URLs from response
            chosen_urls = [s["url"] for s in sources[:5]]

        # Filter sources to only include chosen ones
        curated = [s for s in sources if s.get("url") in chosen_urls]

        # Ensure we have at least some sources
        if not curated:
            curated = sources[:5]

        return curated[:5]

    def run(self, article: Dict) -> Dict:
        """
        Run the curator agent.

        Args:
            article: Article dict with 'query' and 'sources' fields

        Returns:
            Updated article dict with curated 'sources'
        """
        article["sources"] = self.curate_sources(article["query"], article["sources"])
        return article

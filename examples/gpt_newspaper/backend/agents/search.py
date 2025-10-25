"""Search Agent - Searches the web for news articles"""

import os
from typing import Dict, List, Tuple

from tavily import TavilyClient


class SearchAgent:
    """Agent that searches the web for news articles using Tavily API."""

    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str) -> Tuple[List[Dict], str]:
        """
        Search for news articles on the given query.

        Args:
            query: The search query

        Returns:
            Tuple of (sources list, image url)
        """
        results = self.client.search(
            query=query, topic="news", max_results=10, include_images=True
        )
        sources = results.get("results", [])

        # Get first image or use default
        images = results.get("images", [])
        image = (
            images[0]
            if images
            else "https://images.unsplash.com/photo-1542281286-9e0a16bb7366"
        )

        return sources, image

    def run(self, article: Dict) -> Dict:
        """
        Run the search agent.

        Args:
            article: Article dict with 'query' field

        Returns:
            Updated article dict with 'sources' and 'image' fields
        """
        sources, image = self.search(article["query"])
        article["sources"] = sources
        article["image"] = image
        return article

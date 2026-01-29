"""Search Agent - Web search for company and industry information."""

from dataclasses import dataclass

from config import Config
from tavily import TavilyClient


@dataclass
class SearchResult:
    """Single search result."""

    title: str
    url: str
    content: str
    score: float
    published_date: str | None = None


@dataclass
class SearchResponse:
    """Search response containing multiple results."""

    query: str
    results: list[SearchResult]
    image_url: str | None = None


class SearchAgent:
    """Agent for searching company and industry information using Tavily."""

    def __init__(self, api_key: str | None = None):
        """Initialize with Tavily API key."""
        self.api_key = api_key or Config.TAVILY_API_KEY
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is required")
        self.client = TavilyClient(api_key=self.api_key)

    def search_company_news(self, company_name: str, max_results: int | None = None) -> SearchResponse:
        """Search for recent news about a company."""
        max_results = max_results or Config.MAX_SEARCH_RESULTS
        # Use quotes around company name for exact matching, add general topic for relevance
        query = f'"{company_name}" announcement news'
        return self._execute_search(query, max_results, topic="general")

    def search_industry_trends(
        self, company_name: str, industry: str | None = None, max_results: int | None = None
    ) -> SearchResponse:
        """Search for industry trends related to the company."""
        max_results = max_results or Config.MAX_SEARCH_RESULTS
        if industry:
            query = f'"{industry}" industry trends market 2025 2026'
        else:
            query = f'"{company_name}" industry market trends'
        return self._execute_search(query, max_results, topic="news")

    def search_company_profile(self, company_name: str, max_results: int | None = None) -> SearchResponse:
        """Search for company profile and basic information."""
        max_results = max_results or 5
        query = f'"{company_name}" company profile about overview'
        return self._execute_search(query, max_results, topic="general")

    def search_competitors(self, company_name: str, max_results: int | None = None) -> SearchResponse:
        """Search for competitor information."""
        max_results = max_results or 5
        query = f'"{company_name}" competitors alternatives comparison'
        return self._execute_search(query, max_results, topic="general")

    def _execute_search(self, query: str, max_results: int, topic: str = "news") -> SearchResponse:
        """Execute a Tavily search."""
        try:
            response = self.client.search(
                query=query,
                topic=topic,
                max_results=max_results,
                include_images=True,
            )

            results = []
            for item in response.get("results", []):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        content=item.get("content", ""),
                        score=item.get("score", 0.0),
                        published_date=item.get("published_date"),
                    )
                )

            images = response.get("images", [])
            image_url = images[0] if images else None

            return SearchResponse(query=query, results=results, image_url=image_url)

        except Exception as e:
            print(f"Search error for query '{query}': {e}")
            return SearchResponse(query=query, results=[])

    def search_all(self, company_name: str) -> dict[str, SearchResponse]:
        """Execute all search types for a company."""
        return {
            "company_news": self.search_company_news(company_name),
            "industry_trends": self.search_industry_trends(company_name),
            "company_profile": self.search_company_profile(company_name),
            "competitors": self.search_competitors(company_name),
        }

    def to_context_string(self, responses: dict[str, SearchResponse]) -> str:
        """Convert search responses to a context string for LLM."""
        parts = []

        for category, response in responses.items():
            if not response.results:
                continue

            category_name = {
                "company_news": "企業ニュース",
                "industry_trends": "業界動向",
                "company_profile": "企業プロフィール",
                "competitors": "競合情報",
            }.get(category, category)

            parts.append(f"\n## {category_name}\n")

            for i, result in enumerate(response.results, 1):
                parts.append(f"### [{i}] {result.title}")
                parts.append(f"URL: {result.url}")
                if result.published_date:
                    parts.append(f"公開日: {result.published_date}")
                parts.append(f"\n{result.content}\n")

        return "\n".join(parts)

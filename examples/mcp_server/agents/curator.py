"""Curator Agent - Filter and organize search results."""

import json
from dataclasses import dataclass

from agents.search import SearchResponse
from config import Config

from graflow.llm.client import LLMClient


@dataclass
class CuratedSource:
    """A curated source with relevance assessment."""

    title: str
    url: str
    content: str
    relevance_score: float
    category: str
    key_points: list[str]
    published_date: str | None = None


@dataclass
class CurationResult:
    """Result of curation process."""

    company_name: str
    curated_sources: list[CuratedSource]
    summary: str
    categories: dict[str, int]


class CuratorAgent:
    """Agent for curating and filtering search results."""

    SYSTEM_PROMPT = """あなたは企業情報のキュレーターです。
検索結果を分析し、以下の基準で重要度を評価してください：

1. 関連性: 対象企業に直接関係があるか
2. 鮮度: 最新の情報か
3. 信頼性: 信頼できるソースか
4. 重要度: ビジネス上重要な情報か

回答は必ず以下のJSON形式で返してください：
```json
{
  "curated_sources": [
    {
      "title": "記事タイトル",
      "url": "URL",
      "relevance_score": 0.95,
      "category": "ニュース|財務|製品|人事|業界動向|競合",
      "key_points": ["要点1", "要点2"],
      "include": true
    }
  ],
  "summary": "検索結果の要約（2-3文）"
}
```"""

    def __init__(self, llm_client: LLMClient | None = None):
        """Initialize curator with LLM client."""
        self.llm = llm_client or LLMClient(model=Config.DEFAULT_MODEL)

    def curate(
        self,
        company_name: str,
        search_responses: dict[str, SearchResponse],
        min_relevance: float = 0.5,
    ) -> CurationResult:
        """Curate search results."""
        all_results = []
        for category, response in search_responses.items():
            for result in response.results:
                all_results.append(
                    {
                        "title": result.title,
                        "url": result.url,
                        "content": result.content[:500],
                        "published_date": result.published_date,
                        "original_category": category,
                    }
                )

        if not all_results:
            return CurationResult(
                company_name=company_name,
                curated_sources=[],
                summary="検索結果が見つかりませんでした。",
                categories={},
            )

        user_prompt = f"""以下は「{company_name}」に関する検索結果です。
これらをキュレートして、重要度順に整理してください。

検索結果:
{json.dumps(all_results, ensure_ascii=False, indent=2)}

JSONで回答してください。"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.llm.completion_text(
                messages=messages,
                model=Config.DEFAULT_MODEL,
                generation_name="curator",
            )

            json_str = self._extract_json(response)
            result_data = json.loads(json_str)

            curated_sources = []
            categories: dict[str, int] = {}

            for source in result_data.get("curated_sources", []):
                if not source.get("include", True):
                    continue

                score = source.get("relevance_score", 0.0)
                if score < min_relevance:
                    continue

                category = source.get("category", "その他")
                categories[category] = categories.get(category, 0) + 1

                original = next(
                    (r for r in all_results if r["url"] == source.get("url")),
                    None,
                )

                curated_sources.append(
                    CuratedSource(
                        title=source.get("title", ""),
                        url=source.get("url", ""),
                        content=original["content"] if original else "",
                        relevance_score=score,
                        category=category,
                        key_points=source.get("key_points", []),
                        published_date=original.get("published_date") if original else None,
                    )
                )

            curated_sources.sort(key=lambda x: x.relevance_score, reverse=True)

            return CurationResult(
                company_name=company_name,
                curated_sources=curated_sources,
                summary=result_data.get("summary", ""),
                categories=categories,
            )

        except Exception as e:
            print(f"Curation error: {e}")
            curated_sources = []
            for category, response in search_responses.items():
                for result in response.results:
                    curated_sources.append(
                        CuratedSource(
                            title=result.title,
                            url=result.url,
                            content=result.content,
                            relevance_score=result.score,
                            category=category,
                            key_points=[],
                            published_date=result.published_date,
                        )
                    )

            return CurationResult(
                company_name=company_name,
                curated_sources=curated_sources,
                summary="自動キュレーションに失敗しました。",
                categories={},
            )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]

        return text

    def to_context_string(self, result: CurationResult) -> str:
        """Convert curation result to context string for writer."""
        parts = [
            f"# {result.company_name} - キュレート済み情報",
            f"\n## 概要\n{result.summary}",
            "\n## カテゴリ別ソース数",
        ]

        for category, count in result.categories.items():
            parts.append(f"- {category}: {count}件")

        parts.append("\n## ソース詳細\n")

        for i, source in enumerate(result.curated_sources, 1):
            parts.append(f"### [{i}] {source.title}")
            parts.append(f"- カテゴリ: {source.category}")
            parts.append(f"- 関連度: {source.relevance_score:.2f}")
            parts.append(f"- URL: {source.url}")
            if source.published_date:
                parts.append(f"- 公開日: {source.published_date}")
            if source.key_points:
                parts.append("- 要点:")
                for point in source.key_points:
                    parts.append(f"  - {point}")
            parts.append(f"\n{source.content}\n")

        return "\n".join(parts)

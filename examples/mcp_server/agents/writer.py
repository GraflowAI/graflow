"""Writer Agent - Generate company intelligence report."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config import Config

from graflow.llm.client import LLMClient


@dataclass
class ReportSection:
    """A section of the report."""

    title: str
    content: str
    sources: list[str] = field(default_factory=list)


@dataclass
class CompanyReport:
    """Company intelligence report."""

    company_name: str
    generated_at: str
    executive_summary: str
    sections: list[ReportSection]
    key_takeaways: list[str]
    sources: list[dict[str, str]]
    iteration: int = 0
    feedback: str | None = None


class WriterAgent:
    """Agent for writing company intelligence reports."""

    SYSTEM_PROMPT = """あなたは企業情報アナリストです。
提供された情報を元に、訪問先企業に関する包括的なインテリジェンスレポートを作成してください。

レポートには以下を含めてください：
1. エグゼクティブサマリー（3-5文）
2. 企業概要
3. 最新ニュース・動向
4. 業界動向・市場環境
5. 競合状況（該当する場合）
6. 重要なポイント（商談に役立つ情報）

回答は必ず以下のJSON形式で返してください：
```json
{
  "executive_summary": "エグゼクティブサマリー",
  "sections": [
    {
      "title": "セクションタイトル",
      "content": "セクション内容（Markdown形式可）",
      "sources": ["参照したソースのURL"]
    }
  ],
  "key_takeaways": [
    "商談で活用できるポイント1",
    "商談で活用できるポイント2"
  ]
}
```"""

    REVISION_PROMPT = """前回のレポートに対して以下のフィードバックがありました。
このフィードバックを反映してレポートを改善してください。

## フィードバック
{feedback}

## 前回のレポート
{previous_report}

## 元の情報源
{sources}

改善したレポートをJSON形式で返してください。"""

    def __init__(self, llm_client: LLMClient | None = None):
        """Initialize writer with LLM client."""
        self.llm = llm_client or LLMClient(model=Config.WRITER_MODEL)

    def write(
        self,
        company_name: str,
        curated_context: str,
        iteration: int = 0,
        feedback: str | None = None,
        previous_report: "CompanyReport | None" = None,
    ) -> CompanyReport:
        """Write or revise a company report."""
        if feedback and previous_report:
            user_prompt = self.REVISION_PROMPT.format(
                feedback=feedback,
                previous_report=self._report_to_string(previous_report),
                sources=curated_context,
            )
        else:
            user_prompt = f"""以下は「{company_name}」に関するキュレート済み情報です。
この情報を元に、商談準備用のインテリジェンスレポートを作成してください。

{curated_context}

JSONで回答してください。"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.llm.completion_text(
                messages=messages,
                model=Config.WRITER_MODEL,
                generation_name=f"writer_iteration_{iteration}",
            )

            json_str = self._extract_json(response)
            result_data = json.loads(json_str)

            sections = []
            for section_data in result_data.get("sections", []):
                sections.append(
                    ReportSection(
                        title=section_data.get("title", ""),
                        content=section_data.get("content", ""),
                        sources=section_data.get("sources", []),
                    )
                )

            all_sources = []
            seen_urls = set()
            for section in sections:
                for url in section.sources:
                    if url not in seen_urls:
                        all_sources.append({"url": url})
                        seen_urls.add(url)

            return CompanyReport(
                company_name=company_name,
                generated_at=datetime.now().isoformat(),
                executive_summary=result_data.get("executive_summary", ""),
                sections=sections,
                key_takeaways=result_data.get("key_takeaways", []),
                sources=all_sources,
                iteration=iteration,
                feedback=feedback,
            )

        except Exception as e:
            print(f"Writer error: {e}")
            return CompanyReport(
                company_name=company_name,
                generated_at=datetime.now().isoformat(),
                executive_summary=f"レポート生成中にエラーが発生しました: {e}",
                sections=[],
                key_takeaways=[],
                sources=[],
                iteration=iteration,
                feedback=feedback,
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

    def _report_to_string(self, report: CompanyReport) -> str:
        """Convert report to string for revision."""
        parts = [
            f"# {report.company_name} インテリジェンスレポート",
            f"\n## エグゼクティブサマリー\n{report.executive_summary}",
        ]

        for section in report.sections:
            parts.append(f"\n## {section.title}\n{section.content}")

        if report.key_takeaways:
            parts.append("\n## 重要ポイント")
            for point in report.key_takeaways:
                parts.append(f"- {point}")

        return "\n".join(parts)

    def to_markdown(self, report: CompanyReport) -> str:
        """Convert report to Markdown format."""
        parts = [
            f"# {report.company_name} インテリジェンスレポート",
            f"\n*生成日時: {report.generated_at}*",
            f"\n## エグゼクティブサマリー\n{report.executive_summary}",
        ]

        for section in report.sections:
            parts.append(f"\n## {section.title}\n{section.content}")

        if report.key_takeaways:
            parts.append("\n## 商談で活用できるポイント")
            for point in report.key_takeaways:
                parts.append(f"- {point}")

        if report.sources:
            parts.append("\n## 参照ソース")
            for source in report.sources:
                url = source.get("url", "")
                parts.append(f"- {url}")

        return "\n".join(parts)

    def to_dict(self, report: CompanyReport) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "company_name": report.company_name,
            "generated_at": report.generated_at,
            "executive_summary": report.executive_summary,
            "sections": [{"title": s.title, "content": s.content, "sources": s.sources} for s in report.sections],
            "key_takeaways": report.key_takeaways,
            "sources": report.sources,
            "iteration": report.iteration,
        }

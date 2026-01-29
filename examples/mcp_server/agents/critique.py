"""Critique Agent - Review and provide feedback on reports."""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agents.writer import CompanyReport
from config import Config

from graflow.llm.client import LLMClient


class CritiqueDecision(Enum):
    """Decision from critique agent."""

    APPROVE = "approve"
    REVISE = "revise"


@dataclass
class CritiqueResult:
    """Result from critique agent."""

    decision: CritiqueDecision
    overall_score: float
    feedback: str
    issues: list[str]
    suggestions: list[str]
    scores: dict[str, float]


class CritiqueAgent:
    """Agent for critiquing company intelligence reports."""

    SYSTEM_PROMPT = """あなたは企業インテリジェンスレポートのレビュアーです。
レポートを以下の観点で評価してください：

1. **正確性** (accuracy): 情報は事実に基づいているか
2. **完全性** (completeness): 必要な情報が網羅されているか
3. **関連性** (relevance): 商談準備に役立つ情報か
4. **構成** (structure): 論理的で読みやすい構成か
5. **ソース品質** (source_quality): 信頼できるソースを参照しているか

各項目を0.0-1.0で評価し、総合スコアが0.7未満の場合は「revise」を推奨してください。

回答は必ず以下のJSON形式で返してください：
```json
{
  "decision": "approve または revise",
  "overall_score": 0.85,
  "scores": {
    "accuracy": 0.9,
    "completeness": 0.8,
    "relevance": 0.85,
    "structure": 0.9,
    "source_quality": 0.8
  },
  "feedback": "全体的な評価コメント",
  "issues": ["問題点1", "問題点2"],
  "suggestions": ["改善提案1", "改善提案2"]
}
```"""

    def __init__(self, llm_client: LLMClient | None = None, approval_threshold: float = 0.7):
        """Initialize critique agent."""
        self.llm = llm_client or LLMClient(model=Config.CRITIQUE_MODEL)
        self.approval_threshold = approval_threshold

    def critique(
        self,
        report: CompanyReport,
        curated_context: str,
        iteration: int = 0,
    ) -> CritiqueResult:
        """Critique a company report."""
        report_text = self._report_to_string(report)

        user_prompt = f"""以下のレポートを評価してください。

## レポート
{report_text}

## 元の情報ソース（参考）
{curated_context[:3000]}

これはイテレーション {iteration + 1} のレポートです。
JSONで回答してください。"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.llm.completion_text(
                messages=messages,
                model=Config.CRITIQUE_MODEL,
                generation_name=f"critique_iteration_{iteration}",
            )

            json_str = self._extract_json(response)
            result_data = json.loads(json_str)

            overall_score = result_data.get("overall_score", 0.0)

            llm_decision = result_data.get("decision", "revise").lower()
            if overall_score >= self.approval_threshold and llm_decision == "approve":
                decision = CritiqueDecision.APPROVE
            else:
                decision = CritiqueDecision.REVISE

            return CritiqueResult(
                decision=decision,
                overall_score=overall_score,
                feedback=result_data.get("feedback", ""),
                issues=result_data.get("issues", []),
                suggestions=result_data.get("suggestions", []),
                scores=result_data.get("scores", {}),
            )

        except Exception as e:
            print(f"Critique error: {e}")
            return CritiqueResult(
                decision=CritiqueDecision.APPROVE,
                overall_score=0.5,
                feedback=f"評価中にエラーが発生しました: {e}",
                issues=[],
                suggestions=[],
                scores={},
            )

    def _report_to_string(self, report: CompanyReport) -> str:
        """Convert report to string for critique."""
        parts = [
            f"# {report.company_name} インテリジェンスレポート",
            f"\n## エグゼクティブサマリー\n{report.executive_summary}",
        ]

        for section in report.sections:
            parts.append(f"\n## {section.title}\n{section.content}")
            if section.sources:
                parts.append(f"参照: {', '.join(section.sources)}")

        if report.key_takeaways:
            parts.append("\n## 重要ポイント")
            for point in report.key_takeaways:
                parts.append(f"- {point}")

        return "\n".join(parts)

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

    def format_feedback(self, result: CritiqueResult) -> str:
        """Format critique result as feedback string for writer."""
        parts = [
            "## 評価結果",
            f"- 総合スコア: {result.overall_score:.2f}",
            f"- 判定: {result.decision.value}",
        ]

        if result.scores:
            parts.append("\n## 項目別スコア")
            for category, score in result.scores.items():
                category_ja = {
                    "accuracy": "正確性",
                    "completeness": "完全性",
                    "relevance": "関連性",
                    "structure": "構成",
                    "source_quality": "ソース品質",
                }.get(category, category)
                parts.append(f"- {category_ja}: {score:.2f}")

        if result.feedback:
            parts.append(f"\n## フィードバック\n{result.feedback}")

        if result.issues:
            parts.append("\n## 問題点")
            for issue in result.issues:
                parts.append(f"- {issue}")

        if result.suggestions:
            parts.append("\n## 改善提案")
            for suggestion in result.suggestions:
                parts.append(f"- {suggestion}")

        return "\n".join(parts)

    def to_dict(self, result: CritiqueResult) -> dict[str, Any]:
        """Convert critique result to dictionary."""
        return {
            "decision": result.decision.value,
            "overall_score": result.overall_score,
            "feedback": result.feedback,
            "issues": result.issues,
            "suggestions": result.suggestions,
            "scores": result.scores,
        }

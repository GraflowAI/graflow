"""Critique Agent - Provides feedback on articles"""

from datetime import datetime
from typing import Dict, Optional

from utils.litellm import LiteLLMClient, make_message


class CritiqueAgent:
    """Agent that provides critique feedback on written articles."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = LiteLLMClient(model)

    def critique_article(self, article: Dict) -> Optional[str]:
        """
        Provide critique feedback on an article.

        Args:
            article: Article dict

        Returns:
            Critique feedback string or None if article is acceptable
        """
        prompt = f"""Today's date is {datetime.now().strftime('%d/%m/%Y')}.

Article:
{article}

Your task is to provide a really short feedback on the article only if necessary.

If you think the article is good, please return exactly the word "None".

If you noticed the field 'message' in the article, it means the writer has revised the article based on your previous critique. You can provide feedback on the revised article or just return "None" if you think the article is good now.

Please return a string of your critique or the word "None".
"""

        feedback = self.client.chat_text(
            [
                make_message(
                    "system",
                    "You are a newspaper writing critique. Your sole purpose is to provide short feedback on a written article so the writer will know what to fix.",
                ),
                make_message("user", prompt),
            ]
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

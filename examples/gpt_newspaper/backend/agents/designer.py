"""Designer Agent - Designs article HTML"""

import os
import re
from typing import Dict


class DesignerAgent:
    """Agent that designs HTML layout for articles."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_html_template(self) -> str:
        """Load the article HTML template."""
        template_path = os.path.join(os.path.dirname(__file__), "..", "templates", "article", "index.html")
        with open(template_path) as f:
            return f.read()

    def design_article(self, article: Dict) -> Dict:
        """
        Design HTML for an article.

        Args:
            article: Article dict with title, date, image, paragraphs

        Returns:
            Updated article dict with 'html' and 'path' fields
        """
        html_template = self.load_html_template()

        # Replace placeholders
        html = html_template.replace("{{title}}", article["title"])
        html = html.replace("{{image}}", article["image"])
        html = html.replace("{{date}}", article["date"])

        # Replace paragraphs
        paragraphs = article["paragraphs"]
        for i in range(min(5, len(paragraphs))):
            html = html.replace(f"{{{{paragraph{i + 1}}}}}", paragraphs[i])

        article["html"] = html

        # Save article HTML
        filename = re.sub(r'[\/:*?"<>| ]', "_", article["query"])
        filename = f"{filename}.html"
        path = os.path.join(self.output_dir, filename)

        with open(path, "w") as f:
            f.write(html)

        article["path"] = filename

        return article

    def run(self, article: Dict) -> Dict:
        """
        Run the designer agent.

        Args:
            article: Article dict

        Returns:
            Updated article dict with HTML design
        """
        return self.design_article(article)

"""Editor Agent - Compiles articles into newspaper"""

import os
from typing import Dict, List

ARTICLE_TEMPLATES = {
    "layout_1.html": """
    <div class="article">
        <a href="{{path}}" target="_blank"><h2>{{title}}</h2></a>
        <img src="{{image}}" alt="Article Image">
        <p>{{summary}}</p>
    </div>
    """,
    "layout_2.html": """
    <div class="article">
        <img src="{{image}}" alt="Article Image">
        <div>
            <a href="{{path}}" target="_blank"><h2>{{title}}</h2></a>
            <p>{{summary}}</p>
        </div>
    </div>
    """,
    "layout_3.html": """
    <div class="article">
        <a href="{{path}}" target="_blank"><h2>{{title}}</h2></a>
        <img src="{{image}}" alt="Article Image">
        <p>{{summary}}</p>
    </div>
    """,
}


class EditorAgent:
    """Agent that compiles individual articles into a newspaper."""

    def __init__(self, layout: str = "layout_1.html"):
        self.layout = layout

    def load_newspaper_template(self) -> str:
        """Load the newspaper layout template."""
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "templates",
            "newspaper",
            "layouts",
            self.layout,
        )
        with open(template_path) as f:
            return f.read()

    def compile_newspaper(self, articles: List[Dict]) -> str:
        """
        Compile articles into a newspaper.

        Args:
            articles: List of article dicts

        Returns:
            Complete newspaper HTML
        """
        if not articles:
            return "<html><body><h1>No articles found</h1></body></html>"

        newspaper_template = self.load_newspaper_template()
        article_template = ARTICLE_TEMPLATES[self.layout]

        # Generate articles HTML
        articles_html = ""
        for article in articles:
            article_html = article_template.replace("{{title}}", article["title"])
            article_html = article_html.replace("{{image}}", article["image"])
            article_html = article_html.replace("{{summary}}", article["summary"])
            article_html = article_html.replace("{{path}}", article["path"])
            articles_html += article_html

        # Replace placeholders in newspaper template
        newspaper_html = newspaper_template.replace("{{date}}", articles[0]["date"])
        newspaper_html = newspaper_html.replace("{{articles}}", articles_html)

        return newspaper_html

    def run(self, articles: List[Dict]) -> str:
        """
        Run the editor agent.

        Args:
            articles: List of article dicts

        Returns:
            Compiled newspaper HTML
        """
        return self.compile_newspaper(articles)

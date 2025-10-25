"""Publisher Agent - Publishes the final newspaper"""

import os


class PublisherAgent:
    """Agent that publishes the final newspaper HTML."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def publish(self, newspaper_html: str) -> str:
        """
        Publish the newspaper HTML to a file.

        Args:
            newspaper_html: Complete newspaper HTML

        Returns:
            Path to published newspaper
        """
        newspaper_path = os.path.join(self.output_dir, "newspaper.html")

        with open(newspaper_path, "w") as f:
            f.write(newspaper_html)

        print(f"\n📰 Newspaper published to: {newspaper_path}\n")

        return newspaper_path

    def run(self, newspaper_html: str) -> str:
        """
        Run the publisher agent.

        Args:
            newspaper_html: Complete newspaper HTML

        Returns:
            Path to published newspaper
        """
        return self.publish(newspaper_html)

"""Feedback handler base class for HITL callback notifications."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from graflow.hitl.types import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)


class FeedbackHandler:
    """Base class for handling feedback lifecycle callbacks.

    Override methods to receive notifications about feedback requests,
    responses, and timeouts. Useful for integrating with external systems
    like Slack, email, logging, web UI, etc.

    All methods have empty default implementations (noop), so you only
    need to override the ones you care about.

    Example:
        ```python
        class SlackFeedbackHandler(FeedbackHandler):
            def on_request_created(self, request: FeedbackRequest) -> None:
                slack_client.send_message(
                    channel="#approvals",
                    text=f"New approval needed: {request.prompt}",
                    blocks=[{
                        "type": "actions",
                        "elements": [{
                            "type": "button",
                            "text": "Approve",
                            "url": f"https://app.example.com/feedback/{request.feedback_id}"
                        }]
                    }]
                )

            def on_response_received(self, request: FeedbackRequest, response: FeedbackResponse) -> None:
                slack_client.send_message(
                    channel="#approvals",
                    text=f"✅ Approved by {response.responded_by}"
                )

        # Use with request_feedback
        handler = SlackFeedbackHandler()
        context.request_feedback(
            feedback_type="approval",
            prompt="Deploy to production?",
            handler=handler
        )
        ```
    """

    def on_request_created(self, request: FeedbackRequest) -> None:
        """Called when a feedback request is created (before polling).

        This is the ideal place to send notifications to external systems
        (Slack, email, web UI, etc.) to alert users about pending feedback.

        Args:
            request: The feedback request that was created
        """
        pass

    def on_response_received(self, request: FeedbackRequest, response: FeedbackResponse) -> None:
        """Called when a feedback response is received (after polling succeeds).

        Args:
            request: The original feedback request
            response: The feedback response that was received
        """
        pass

    def on_request_timeout(self, request: FeedbackRequest) -> None:
        """Called when a feedback request times out (no response received).

        Args:
            request: The feedback request that timed out
        """
        pass


class SlackFeedbackHandler(FeedbackHandler):
    """Handler that sends Slack notifications for feedback events.

    Sends Slack messages via webhook when feedback is requested, received, or times out.

    Example:
        ```python
        from graflow.hitl.handler import SlackFeedbackHandler

        handler = SlackFeedbackHandler(
            webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            channel="#approvals",
            feedback_ui_url="https://your-app.com/feedback/{feedback_id}"
        )

        # Use handler with request_feedback
        context.request_feedback(
            feedback_type="approval",
            prompt="Deploy to production?",
            handler=handler
        )
        ```
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str = "#approvals",
        feedback_ui_url: Optional[str] = None,
    ):
        """Initialize Slack handler.

        Args:
            webhook_url: Slack webhook URL
            channel: Slack channel to post to (default: #approvals)
            feedback_ui_url: Optional URL template for feedback UI.
                             Use {feedback_id} placeholder (e.g., "https://app.com/feedback/{feedback_id}")
        """
        self.webhook_url = webhook_url
        self.channel = channel
        self.feedback_ui_url = feedback_ui_url

    def _send_slack_message(self, message: dict) -> bool:
        """Send message to Slack webhook.

        Args:
            message: Slack message payload

        Returns:
            True if successful, False otherwise
        """
        try:
            import requests

            response = requests.post(
                self.webhook_url,
                json=message,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error("Failed to send Slack message: %s", e, exc_info=True)
            return False

    def on_request_created(self, request: FeedbackRequest) -> None:
        """Send Slack notification when feedback is requested."""
        logger.info(
            "Sending Slack notification for feedback request: %s",
            request.feedback_id,
            extra={"feedback_id": request.feedback_id, "channel": self.channel},
        )

        # Build blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🔔 {request.feedback_type.value.title()} Request",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{request.prompt}*",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Task: `{request.task_id}` | Timeout: `{request.timeout}s` | ID: `{request.feedback_id}`",
                    }
                ],
            },
        ]

        # Add action button if feedback_ui_url is configured
        if self.feedback_ui_url:
            feedback_url = self.feedback_ui_url.format(feedback_id=request.feedback_id)
            blocks.append(
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Respond",
                            },
                            "url": feedback_url,
                            "style": "primary",
                        }
                    ],
                }
            )

        message = {
            "channel": self.channel,
            "text": f"🔔 Feedback needed: {request.prompt}",
            "blocks": blocks,
        }

        self._send_slack_message(message)

    def on_response_received(self, request: FeedbackRequest, response: FeedbackResponse) -> None:
        """Send Slack notification when feedback is received."""
        logger.info(
            "Sending Slack notification for feedback response: %s",
            request.feedback_id,
            extra={"feedback_id": request.feedback_id, "channel": self.channel},
        )

        # Determine response details
        if hasattr(response, "approved"):
            status_emoji = "✅" if response.approved else "❌"
            status_text = "Approved" if response.approved else "Rejected"
        elif hasattr(response, "text"):
            status_emoji = "💬"
            status_text = f"Text: {response.text[:100]}"  # type: ignore
        elif hasattr(response, "selected"):
            status_emoji = "✓"
            status_text = f"Selected: {response.selected}"
        else:
            status_emoji = "✓"
            status_text = "Received"

        responded_by = response.responded_by or "someone"

        message = {
            "channel": self.channel,
            "text": f"{status_emoji} Feedback received from {responded_by}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{status_emoji} *{status_text}*\nResponded by: {responded_by}",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"ID: `{request.feedback_id}`",
                        }
                    ],
                },
            ],
        }

        self._send_slack_message(message)

    def on_request_timeout(self, request: FeedbackRequest) -> None:
        """Send Slack notification when feedback times out."""
        logger.info(
            "Sending Slack notification for feedback timeout: %s",
            request.feedback_id,
            extra={"feedback_id": request.feedback_id, "channel": self.channel},
        )

        message = {
            "channel": self.channel,
            "text": f"⏱️ Feedback request timed out: {request.prompt}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"⏱️ *Feedback Request Timed Out*\n{request.prompt}",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"No response after `{request.timeout}s` | ID: `{request.feedback_id}`",
                        }
                    ],
                },
            ],
        }

        self._send_slack_message(message)

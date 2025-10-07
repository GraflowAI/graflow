"""Tests for DirectTaskHandler."""

from unittest.mock import Mock

import pytest

from graflow.core.handlers.direct import DirectTaskHandler


class TestDirectTaskHandler:
    """Test DirectTaskHandler implementation."""

    def test_direct_task_handler_success(self):
        """Test successful task execution."""
        # Create mock task
        mock_task = Mock()
        mock_task.task_id = "test_task"
        mock_task.run.return_value = "success_result"

        # Create mock context
        mock_context = Mock()

        # Create handler and execute
        handler = DirectTaskHandler()
        handler.execute_task(mock_task, mock_context)

        # Verify task.run() was called
        mock_task.run.assert_called_once()

        # Verify context.set_result() was called with correct arguments
        mock_context.set_result.assert_called_once_with("test_task", "success_result")

    def test_direct_task_handler_with_exception(self):
        """Test task execution with exception."""
        # Create mock task that raises exception
        mock_task = Mock()
        mock_task.task_id = "failing_task"
        test_exception = ValueError("test error")
        mock_task.run.side_effect = test_exception

        # Create mock context
        mock_context = Mock()

        # Create handler and execute
        handler = DirectTaskHandler()

        # Execute and expect exception
        with pytest.raises(ValueError, match="test error"):
            handler.execute_task(mock_task, mock_context)

        # Verify task.run() was called
        mock_task.run.assert_called_once()

        # Verify context.set_result() was called with exception
        mock_context.set_result.assert_called_once_with("failing_task", test_exception)

    def test_direct_task_handler_stores_result(self):
        """Test that handler correctly stores various result types."""
        # Test with None result
        mock_task = Mock()
        mock_task.task_id = "none_task"
        mock_task.run.return_value = None
        mock_context = Mock()

        handler = DirectTaskHandler()
        handler.execute_task(mock_task, mock_context)

        mock_context.set_result.assert_called_once_with("none_task", None)

        # Test with dict result
        mock_task = Mock()
        mock_task.task_id = "dict_task"
        mock_task.run.return_value = {"key": "value"}
        mock_context = Mock()

        handler.execute_task(mock_task, mock_context)

        mock_context.set_result.assert_called_once_with("dict_task", {"key": "value"})

        # Test with list result
        mock_task = Mock()
        mock_task.task_id = "list_task"
        mock_task.run.return_value = [1, 2, 3]
        mock_context = Mock()

        handler.execute_task(mock_task, mock_context)

        mock_context.set_result.assert_called_once_with("list_task", [1, 2, 3])

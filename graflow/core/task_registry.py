"""Function registry and resolution for distributed task execution.

This module provides function serialization, registration, and resolution
capabilities for executing tasks across distributed workers.
"""

from __future__ import annotations

import base64
import importlib
import inspect
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple

try:
    import cloudpickle
except ImportError:
    import pickle as cloudpickle

from graflow.exceptions import GraflowRuntimeError

if TYPE_CHECKING:
    from graflow.core.task import Executable

logger = logging.getLogger(__name__)


class TaskResolutionError(GraflowRuntimeError):
    """Raised when a task cannot be resolved."""
    pass


class TaskRegistry:
    """Global task registry for fallback resolution."""

    _registry: ClassVar[Dict[str, Executable]] = {}

    @classmethod
    def register(cls, name: str, task: Executable) -> None:
        """Register task in global registry.

        Args:
            name: Task name or key for registration
            task: Task to register
        """
        cls._registry[name] = task
        logger.debug(f"Registered task: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Executable]:
        """Get task from registry.

        Args:
            name: Task name to retrieve

        Returns:
            Task or None if not found
        """
        return cls._registry.get(name)

    @classmethod
    def clear(cls) -> None:
        """Clear registry (mainly for testing)."""
        cls._registry.clear()

    @classmethod
    def list_tasks(cls) -> list[str]:
        """List all registered task names."""
        return list(cls._registry.keys())


class TaskSerializer:
    """Handles task serialization and deserialization."""

    @staticmethod
    def serialize_task(task: Executable,
                      strategy: str = "reference") -> Dict[str, Any]:
        """Serialize task using specified strategy.

        Args:
            task: Task to serialize
            strategy: Serialization strategy - "reference", "pickle", or "source"

        Returns:
            Dictionary containing serialized task data

        Raises:
            ValueError: If strategy is unsupported
        """
        if strategy == "reference":
            return TaskSerializer._serialize_reference(task)
        elif strategy == "pickle":
            return TaskSerializer._serialize_pickle(task)
        elif strategy == "source":
            return TaskSerializer._serialize_source(task)
        else:
            raise ValueError(f"Unsupported serialization strategy: {strategy}")

    @staticmethod
    def _serialize_reference(task: Executable) -> Dict[str, Any]:
        """Serialize task as importable reference."""
        return {
            "strategy": "reference",
            "module": task.__module__,
            "name": task.__name__,
            "qualname": getattr(task, "__qualname__", task.__name__)
        }

    @staticmethod
    def _serialize_source(task: Executable) -> Dict[str, Any]:
        """Serialize task as source code."""
        try:
            source_code = inspect.getsource(task)
            return {
                "strategy": "source",
                "source": source_code,
                "name": task.__name__,
                "module": getattr(task, "__module__", "unknown"),
                "qualname": getattr(task, "__qualname__", task.__name__)
            }
        except Exception as e:
            raise TaskResolutionError(f"Failed to get source for task: {e}") from e

    @staticmethod
    def _serialize_pickle(task: Executable) -> Dict[str, Any]:
        """Serialize task using cloudpickle."""
        try:
            pickled_data = cloudpickle.dumps(task)
            encoded_data = base64.b64encode(pickled_data).decode('utf-8')
            return {
                "strategy": "pickle",
                "data": encoded_data,
                "name": getattr(task, "__name__", "unknown"),
                "module": getattr(task, "__module__", "unknown")
            }
        except Exception as e:
            raise TaskResolutionError(f"Failed to pickle task: {e}") from e

    @staticmethod
    def deserialize_task(task_data: Dict[str, Any]) -> Tuple[Executable, str]:
        """Deserialize task with fallback strategy.

        Strategy resolution order:
        1. Try the specified strategy
        2. Try registry lookup as fallback
        3. Raise TaskResolutionError if all fail

        Args:
            task_data: Serialized task data

        Returns:
            Tuple of (task_object, resolution_method)

        Raises:
            TaskResolutionError: When task cannot be resolved
        """
        strategy = task_data.get("strategy", "reference")

        # Try primary strategy
        try:
            if strategy == "reference":
                return TaskSerializer._deserialize_reference(task_data)
            elif strategy == "pickle":
                return TaskSerializer._deserialize_pickle(task_data)
            elif strategy == "source":
                return TaskSerializer._deserialize_source(task_data)
        except Exception as e:
            logger.debug(f"Primary strategy '{strategy}' failed: {e}")

        # Try registry fallback
        task_name = task_data.get("name", "")
        module_name = task_data.get("module", "")

        # Try different naming patterns
        for name_pattern in [
            task_name,
            f"{module_name}.{task_name}",
            task_data.get("qualname", "")
        ]:
            if name_pattern:
                task = TaskRegistry.get(name_pattern)
                if task is not None:
                    logger.debug(f"Task found in registry: {name_pattern}")
                    return task, "registry"

        # All strategies failed
        task_ref = f"{module_name}.{task_name}"
        raise TaskResolutionError(f"Cannot resolve task: {task_ref}")

    @staticmethod
    def _deserialize_reference(task_data: Dict[str, Any]) -> Tuple[Executable, str]:
        """Deserialize task from importable reference."""
        module_name = task_data["module"]
        task_name = task_data["name"]

        module = importlib.import_module(module_name)
        task = getattr(module, task_name)

        logger.debug(f"Task imported: {module_name}.{task_name}")
        return task, "import"

    @staticmethod
    def _deserialize_source(task_data: Dict[str, Any]) -> Tuple[Executable, str]:
        """Deserialize task from source code."""
        source_code = task_data["source"]
        task_name = task_data["name"]
        task_module = task_data.get("module", "__main__")

        # Create temporary module namespace
        module_namespace = {"__name__": task_module}

        # Import commonly used modules into the namespace
        try:
            import builtins
            module_namespace.update(vars(builtins))
        except ImportError:
            pass

        try:
            # Execute the source code in the temporary namespace
            exec(source_code, module_namespace)

            # Find the task by name
            if task_name not in module_namespace:
                raise TaskResolutionError(f"Task {task_name} not found in source")

            task = module_namespace[task_name]

            # Check if it's callable instead of isinstance check
            if not callable(task):
                raise TaskResolutionError(f"Deserialized object is not callable: {type(task)}")

            # For stricter checking, ensure it has the expected Executable interface
            from graflow.core.task import Executable
            if not isinstance(task, Executable):
                raise TaskResolutionError(f"Deserialized object is not an Executable: {type(task)}")

            return task, "source"

        except Exception as e:
            raise TaskResolutionError(f"Failed to execute source code for {task_name}: {e}") from e

    @staticmethod
    def _deserialize_pickle(task_data: Dict[str, Any]) -> Tuple[Executable, str]:
        """Deserialize task from cloudpickle data."""
        encoded_data = task_data["data"]
        pickled_data = base64.b64decode(encoded_data.encode('utf-8'))
        task = cloudpickle.loads(pickled_data)

        task_name = task_data.get("name", "unknown")
        logger.debug(f"Task unpickled: {task_name}")
        return task, "pickle"

    @staticmethod
    def try_deserialize_task(task_data: Dict[str, Any]) -> Tuple[Optional[Executable], str]:
        """Safe wrapper that returns None instead of raising exception.

        Args:
            task_data: Serialized task data

        Returns:
            Tuple of (task_object_or_none, resolution_method)
            resolution_method: "import", "pickle", "registry", or "failed"
        """
        try:
            task, method = TaskSerializer.deserialize_task(task_data)
            return task, method
        except TaskResolutionError:
            return None, "failed"


class TaskResolver:
    """High-level resolver for task registration and resolution."""

    def __init__(self, default_strategy: str = "reference"):
        """Initialize with default serialization strategy.

        Args:
            default_strategy: Default strategy for task serialization
        """
        self.default_strategy = default_strategy
        self.serializer = TaskSerializer()

    def register_task(self, task_id: str, task: Executable) -> None:
        """Register a task for later resolution.

        Args:
            task_id: Task identifier
            task: Task to register
        """
        # Register with both task_id and task reference
        TaskRegistry.register(task_id, task)

        # Also register with module.name pattern for import resolution
        if hasattr(task, '__module__') and hasattr(task, '__name__'):
            task_ref = f"{task.__module__}.{task.__name__}"
            TaskRegistry.register(task_ref, task)

    def serialize_task(self, task: Executable,
                      strategy: Optional[str] = None) -> Dict[str, Any]:
        """Serialize task for storage.

        Args:
            task: Task to serialize
            strategy: Override default strategy

        Returns:
            Serialized task data
        """
        strategy = strategy or self.default_strategy
        return self.serializer.serialize_task(task, strategy)

    def resolve_task(self, task_data: Dict[str, Any]) -> Executable:
        """Resolve task from serialized data.

        Args:
            task_data: Serialized task data

        Returns:
            Resolved task object

        Raises:
            TaskResolutionError: If task cannot be resolved
        """
        task, method = self.serializer.deserialize_task(task_data)
        logger.debug(f"Resolved task via {method}")
        return task

    def get_registered_tasks(self) -> Dict[str, Executable]:
        """Get all registered tasks."""
        return TaskRegistry._registry.copy()


# Global instance for convenience
default_task_resolver = TaskResolver()

# Convenience functions
def register_task(task_id: str, task: Executable) -> None:
    """Register a task globally."""
    default_task_resolver.register_task(task_id, task)

def serialize_task(task: Executable, strategy: str = "reference") -> Dict[str, Any]:
    """Serialize task globally."""
    return default_task_resolver.serialize_task(task, strategy)

def resolve_task(task_data: Dict[str, Any]) -> Executable:
    """Resolve task globally."""
    return default_task_resolver.resolve_task(task_data)

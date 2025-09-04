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


class FunctionResolutionError(GraflowRuntimeError):
    """Raised when a function cannot be resolved."""
    pass


class FunctionRegistry:
    """Global function registry for fallback resolution."""

    _registry: ClassVar[Dict[str, Executable]] = {}

    @classmethod
    def register(cls, name: str, func: Executable) -> None:
        """Register function in global registry.

        Args:
            name: Function name or key for registration
            func: Callable function to register
        """
        cls._registry[name] = func
        logger.debug(f"Registered function: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Executable]:
        """Get function from registry.

        Args:
            name: Function name to retrieve

        Returns:
            Callable function or None if not found
        """
        return cls._registry.get(name)

    @classmethod
    def clear(cls) -> None:
        """Clear registry (mainly for testing)."""
        cls._registry.clear()

    @classmethod
    def list_functions(cls) -> list[str]:
        """List all registered function names."""
        return list(cls._registry.keys())


class FunctionSerializer:
    """Handles function serialization and deserialization."""

    @staticmethod
    def serialize_function(func: Executable,
                          strategy: str = "reference") -> Dict[str, Any]:
        """Serialize function using specified strategy.

        Args:
            func: Function to serialize
            strategy: Serialization strategy - "reference", "pickle", or "source"

        Returns:
            Dictionary containing serialized function data

        Raises:
            ValueError: If strategy is unsupported
        """
        if strategy == "reference":
            return FunctionSerializer._serialize_reference(func)
        elif strategy == "pickle":
            return FunctionSerializer._serialize_pickle(func)
        elif strategy == "source":
            return FunctionSerializer._serialize_source(func)
        else:
            raise ValueError(f"Unsupported serialization strategy: {strategy}")

    @staticmethod
    def _serialize_reference(func: Executable) -> Dict[str, Any]:
        """Serialize function as importable reference."""
        return {
            "strategy": "reference",
            "module": func.__module__,
            "name": func.__name__,
            "qualname": getattr(func, "__qualname__", func.__name__)
        }

    @staticmethod
    def _serialize_source(func: Executable) -> Dict[str, Any]:
        """Serialize function as source code."""
        try:
            source_code = inspect.getsource(func)
            return {
                "strategy": "source",
                "source": source_code,
                "name": func.__name__,
                "module": getattr(func, "__module__", "unknown"),
                "qualname": getattr(func, "__qualname__", func.__name__)
            }
        except Exception as e:
            raise FunctionResolutionError(f"Failed to get source for function: {e}") from e

    @staticmethod
    def _serialize_pickle(func: Executable) -> Dict[str, Any]:
        """Serialize function using cloudpickle."""
        try:
            pickled_data = cloudpickle.dumps(func)
            encoded_data = base64.b64encode(pickled_data).decode('utf-8')
            return {
                "strategy": "pickle",
                "data": encoded_data,
                "name": getattr(func, "__name__", "unknown"),
                "module": getattr(func, "__module__", "unknown")
            }
        except Exception as e:
            raise FunctionResolutionError(f"Failed to pickle function: {e}") from e

    @staticmethod
    def deserialize_function(func_data: Dict[str, Any]) -> Tuple[Executable, str]:
        """Deserialize function with fallback strategy.

        Strategy resolution order:
        1. Try the specified strategy
        2. Try registry lookup as fallback
        3. Raise FunctionResolutionError if all fail

        Args:
            func_data: Serialized function data

        Returns:
            Tuple of (function_object, resolution_method)

        Raises:
            FunctionResolutionError: When function cannot be resolved
        """
        strategy = func_data.get("strategy", "reference")

        # Try primary strategy
        try:
            if strategy == "reference":
                return FunctionSerializer._deserialize_reference(func_data)
            elif strategy == "pickle":
                return FunctionSerializer._deserialize_pickle(func_data)
            elif strategy == "source":
                return FunctionSerializer._deserialize_source(func_data)
        except Exception as e:
            logger.debug(f"Primary strategy '{strategy}' failed: {e}")

        # Try registry fallback
        func_name = func_data.get("name", "")
        module_name = func_data.get("module", "")

        # Try different naming patterns
        for name_pattern in [
            func_name,
            f"{module_name}.{func_name}",
            func_data.get("qualname", "")
        ]:
            if name_pattern:
                func = FunctionRegistry.get(name_pattern)
                if func is not None:
                    logger.debug(f"Function found in registry: {name_pattern}")
                    return func, "registry"

        # All strategies failed
        func_ref = f"{module_name}.{func_name}"
        raise FunctionResolutionError(f"Cannot resolve function: {func_ref}")

    @staticmethod
    def _deserialize_reference(func_data: Dict[str, Any]) -> Tuple[Executable, str]:
        """Deserialize function from importable reference."""
        module_name = func_data["module"]
        func_name = func_data["name"]

        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        logger.debug(f"Function imported: {module_name}.{func_name}")
        return func, "import"

    @staticmethod
    def _deserialize_source(func_data: Dict[str, Any]) -> Tuple[Executable, str]:
        """Deserialize function from source code."""
        source_code = func_data["source"]
        func_name = func_data["name"]
        func_module = func_data.get("module", "__main__")

        # Create temporary module namespace
        module_namespace = {"__name__": func_module}

        # Import commonly used modules into the namespace
        try:
            import builtins
            module_namespace.update(vars(builtins))
        except ImportError:
            pass

        try:
            # Execute the source code in the temporary namespace
            exec(source_code, module_namespace)

            # Find the function by name
            if func_name not in module_namespace:
                raise FunctionResolutionError(f"Function {func_name} not found in source")

            func = module_namespace[func_name]

            # Check if it's callable instead of isinstance check
            if not callable(func):
                raise FunctionResolutionError(f"Deserialized object is not callable: {type(func)}")

            # For stricter checking, ensure it has the expected Executable interface
            from graflow.core.task import Executable
            if not isinstance(func, Executable):
                raise FunctionResolutionError(f"Deserialized object is not an Executable: {type(func)}")

            return func, "source"

        except Exception as e:
            raise FunctionResolutionError(f"Failed to execute source code for {func_name}: {e}") from e

    @staticmethod
    def _deserialize_pickle(func_data: Dict[str, Any]) -> Tuple[Executable, str]:
        """Deserialize function from cloudpickle data."""
        encoded_data = func_data["data"]
        pickled_data = base64.b64decode(encoded_data.encode('utf-8'))
        func = cloudpickle.loads(pickled_data)

        func_name = func_data.get("name", "unknown")
        logger.debug(f"Function unpickled: {func_name}")
        return func, "pickle"

    @staticmethod
    def try_deserialize_function(func_data: Dict[str, Any]) -> Tuple[Optional[Executable], str]:
        """Safe wrapper that returns None instead of raising exception.

        Args:
            func_data: Serialized function data

        Returns:
            Tuple of (function_object_or_none, resolution_method)
            resolution_method: "import", "pickle", "registry", or "failed"
        """
        try:
            func, method = FunctionSerializer.deserialize_function(func_data)
            return func, method
        except FunctionResolutionError:
            return None, "failed"


class TaskFunctionManager:
    """High-level manager for task function registration and resolution."""

    def __init__(self, default_strategy: str = "reference"):
        """Initialize with default serialization strategy.

        Args:
            default_strategy: Default strategy for function serialization
        """
        self.default_strategy = default_strategy
        self.serializer = FunctionSerializer()

    def register_task_function(self, task_id: str, func: Executable) -> None:
        """Register a task function for later resolution.

        Args:
            task_id: Task identifier
            func: Task function to register
        """
        # Register with both task_id and function reference
        FunctionRegistry.register(task_id, func)

        # Also register with module.name pattern for import resolution
        if hasattr(func, '__module__') and hasattr(func, '__name__'):
            func_ref = f"{func.__module__}.{func.__name__}"
            FunctionRegistry.register(func_ref, func)

    def serialize_task_function(self, func: Executable,
                               strategy: Optional[str] = None) -> Dict[str, Any]:
        """Serialize task function for storage.

        Args:
            func: Function to serialize
            strategy: Override default strategy

        Returns:
            Serialized function data
        """
        strategy = strategy or self.default_strategy
        return self.serializer.serialize_function(func, strategy)

    def resolve_task_function(self, func_data: Dict[str, Any]) -> Executable:
        """Resolve task function from serialized data.

        Args:
            func_data: Serialized function data

        Returns:
            Resolved function object

        Raises:
            FunctionResolutionError: If function cannot be resolved
        """
        func, method = self.serializer.deserialize_function(func_data)
        logger.debug(f"Resolved function via {method}")
        return func

    def get_registered_functions(self) -> Dict[str, Executable]:
        """Get all registered functions."""
        return FunctionRegistry._registry.copy()


# Global instance for convenience
default_function_manager = TaskFunctionManager()

# Convenience functions
def register_task_function(task_id: str, func: Executable) -> None:
    """Register a task function globally."""
    default_function_manager.register_task_function(task_id, func)

def serialize_function(func: Executable, strategy: str = "reference") -> Dict[str, Any]:
    """Serialize function globally."""
    return default_function_manager.serialize_task_function(func, strategy)

def resolve_function(func_data: Dict[str, Any]) -> Executable:
    """Resolve function globally."""
    return default_function_manager.resolve_task_function(func_data)

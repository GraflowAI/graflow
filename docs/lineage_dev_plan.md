# Graflow Lineage System Development Plan

## Overview

This document outlines the development plan for a new lineage system in graflow that tracks runtime task graph execution, manages task relationships, and provides REST API endpoints for GUI visualization and monitoring.

## Goals

1. **Runtime Graph Tracking**: Monitor and record task execution flow in real-time
2. **Lineage Management**: Maintain complete lineage of task dependencies and data flow
3. **REST API**: Provide HTTP endpoints for external GUI applications
4. **Visualization Support**: Enable graph visualization with execution status and metrics
5. **Historical Analysis**: Support querying past executions and performance analysis

## Architecture Overview

```text
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   ExecutionContext  │    │   LineageManager    │    │    REST API         │
│                     │    │                     │    │                     │
│ - Task execution    │───▶│ - NetworkX graph    │───▶│ - Graph endpoints   │
│ - Channel data      │    │ - Execution events  │    │ - Status endpoints  │
│ - Context state     │    │ - Metrics tracking  │    │ - History endpoints │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                      │                          │
                                      ▼                          ▼
                           ┌─────────────────────┐    ┌─────────────────────┐
                           │   Storage Layer     │    │    Web GUI          │
                           │                     │    │                     │
                           │ - Graph snapshots   │    │ - Real-time view    │
                           │ - Execution logs    │    │ - Historical data   │
                           │ - Performance data  │    │ - Interactive viz   │
                           └─────────────────────┘    └─────────────────────┘
```

## Core Components

### 1. LineageManager (`graflow/lineage/manager.py`)

The central component that manages runtime task graph tracking:

```python
import networkx as nx
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import uuid

@dataclass
class TaskExecutionInfo:
    """Information about a task execution."""
    task_id: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    duration: Optional[float] = None
    result_size: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    """Information about a workflow execution session."""
    execution_id: str
    workflow_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

class LineageManager:
    """Manages runtime task graph and execution lineage."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.executions: Dict[str, WorkflowExecution] = {}
        self.task_executions: Dict[str, TaskExecutionInfo] = {}
        self.current_execution: Optional[str] = None
        self._listeners: List[callable] = []
    
    def start_workflow_execution(self, workflow_name: str, 
                                execution_context: 'ExecutionContext') -> str:
        """Start tracking a new workflow execution."""
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_name=workflow_name,
            start_time=datetime.now()
        )
        
        self.executions[execution_id] = execution
        self.current_execution = execution_id
        
        # Initialize graph from execution context
        self._initialize_graph_from_context(execution_context)
        
        # Notify listeners
        self._notify_listeners('workflow_started', execution)
        
        return execution_id
    
    def add_task_execution(self, task_id: str, execution_context: 'ExecutionContext'):
        """Record the start of a task execution."""
        if not self.current_execution:
            return
        
        execution_id = f"{self.current_execution}_{task_id}_{uuid.uuid4().hex[:8]}"
        task_info = TaskExecutionInfo(
            task_id=task_id,
            execution_id=execution_id,
            start_time=datetime.now(),
            status="running"
        )
        
        self.task_executions[execution_id] = task_info
        
        # Update graph with execution info
        if self.graph.has_node(task_id):
            self.graph.nodes[task_id]['current_execution'] = execution_id
            self.graph.nodes[task_id]['status'] = 'running'
        
        self._notify_listeners('task_started', task_info)
    
    def complete_task_execution(self, task_id: str, result: Any, 
                              execution_context: 'ExecutionContext'):
        """Record the completion of a task execution."""
        # Find current execution for this task
        current_exec_id = None
        for exec_id, task_info in self.task_executions.items():
            if (task_info.task_id == task_id and 
                task_info.status == "running"):
                current_exec_id = exec_id
                break
        
        if not current_exec_id:
            return
        
        task_info = self.task_executions[current_exec_id]
        task_info.end_time = datetime.now()
        task_info.status = "completed"
        task_info.duration = (task_info.end_time - task_info.start_time).total_seconds()
        
        # Calculate result size if possible
        try:
            import sys
            task_info.result_size = sys.getsizeof(result)
        except:
            task_info.result_size = None
        
        # Update graph
        if self.graph.has_node(task_id):
            self.graph.nodes[task_id]['status'] = 'completed'
            self.graph.nodes[task_id]['last_duration'] = task_info.duration
        
        self._notify_listeners('task_completed', task_info)
    
    def fail_task_execution(self, task_id: str, error: Exception):
        """Record a task execution failure."""
        # Find current execution for this task
        current_exec_id = None
        for exec_id, task_info in self.task_executions.items():
            if (task_info.task_id == task_id and 
                task_info.status == "running"):
                current_exec_id = exec_id
                break
        
        if not current_exec_id:
            return
        
        task_info = self.task_executions[current_exec_id]
        task_info.end_time = datetime.now()
        task_info.status = "failed"
        task_info.duration = (task_info.end_time - task_info.start_time).total_seconds()
        task_info.error_message = str(error)
        
        # Update graph
        if self.graph.has_node(task_id):
            self.graph.nodes[task_id]['status'] = 'failed'
            self.graph.nodes[task_id]['error'] = str(error)
        
        self._notify_listeners('task_failed', task_info)
    
    def get_graph_snapshot(self) -> Dict[str, Any]:
        """Get current graph state as JSON-serializable dict."""
        return {
            'nodes': [
                {
                    'id': node_id,
                    'data': dict(data)
                }
                for node_id, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    'data': dict(data)
                }
                for source, target, data in self.graph.edges(data=True)
            ]
        }
    
    def _initialize_graph_from_context(self, execution_context: 'ExecutionContext'):
        """Initialize graph structure from execution context."""
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes from context graph
        for node_id in execution_context.graph.nodes:
            task = execution_context.graph.get_task(node_id)
            self.graph.add_node(node_id, {
                'task_type': type(task).__name__,
                'status': 'pending',
                'created_at': datetime.now().isoformat()
            })
        
        # Add edges from context graph
        for edge in execution_context.graph.edges:
            source, target = edge
            self.graph.add_edge(source, target)
    
    def add_listener(self, callback: callable):
        """Add event listener for lineage events."""
        self._listeners.append(callback)
    
    def _notify_listeners(self, event_type: str, data: Any):
        """Notify all listeners of an event."""
        for callback in self._listeners:
            try:
                callback(event_type, data)
            except Exception as e:
                print(f"Error in lineage listener: {e}")

# Global instance
_lineage_manager = LineageManager()

def get_lineage_manager() -> LineageManager:
    """Get the global lineage manager instance."""
    return _lineage_manager
```

### 2. Integration with ExecutionContext (`graflow/core/context.py`)

Add lineage tracking hooks to the execution context:

```python
# Add to ExecutionContext class

from graflow.lineage.manager import get_lineage_manager

class ExecutionContext:
    def __init__(self, ...):
        # ... existing initialization ...
        self._lineage_manager = get_lineage_manager()
        self._workflow_execution_id: Optional[str] = None
    
    def start_workflow_tracking(self, workflow_name: str):
        """Start lineage tracking for this workflow execution."""
        self._workflow_execution_id = self._lineage_manager.start_workflow_execution(
            workflow_name, self
        )
    
    @contextmanager
    def executing_task(self, task: Executable):
        """Context manager for task execution with lineage tracking."""
        task_ctx = self.create_task_context(task.task_id)
        self.push_task_context(task_ctx)
        
        # Start lineage tracking
        self._lineage_manager.add_task_execution(task.task_id, self)
        
        try:
            task.set_execution_context(self)
            yield task_ctx
            
            # Task completed successfully
            result = self.get_result(task.task_id)
            self._lineage_manager.complete_task_execution(task.task_id, result, self)
            
        except Exception as e:
            # Task failed
            self._lineage_manager.fail_task_execution(task.task_id, e)
            raise
        finally:
            self.pop_task_context()
```

### 3. REST API Server (`graflow/api/server.py`)

FastAPI-based REST server for lineage data:

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import json
from datetime import datetime

from graflow.lineage.manager import get_lineage_manager

app = FastAPI(title="Graflow Lineage API", version="1.0.0")

# Enable CORS for web GUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lineage_manager = get_lineage_manager()

# WebSocket connections for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Connection might be closed
                self.disconnect(connection)

manager = ConnectionManager()

# Set up real-time event broadcasting
def lineage_event_listener(event_type: str, data: Any):
    """Listen to lineage events and broadcast to WebSocket clients."""
    import asyncio
    
    message = {
        'type': event_type,
        'timestamp': datetime.now().isoformat(),
        'data': data.__dict__ if hasattr(data, '__dict__') else data
    }
    
    # Broadcast to WebSocket clients
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(manager.broadcast(message))
    except:
        pass  # No event loop running

lineage_manager.add_listener(lineage_event_listener)

# REST Endpoints

@app.get("/api/graph/current")
async def get_current_graph():
    """Get the current task graph state."""
    return lineage_manager.get_graph_snapshot()

@app.get("/api/executions")
async def get_executions():
    """Get all workflow executions."""
    return {
        'executions': [
            {
                'execution_id': exec_id,
                **execution.__dict__
            }
            for exec_id, execution in lineage_manager.executions.items()
        ]
    }

@app.get("/api/executions/{execution_id}")
async def get_execution_details(execution_id: str):
    """Get details of a specific workflow execution."""
    if execution_id not in lineage_manager.executions:
        return {"error": "Execution not found"}, 404
    
    execution = lineage_manager.executions[execution_id]
    
    # Get task executions for this workflow
    task_executions = [
        task_info.__dict__
        for task_info in lineage_manager.task_executions.values()
        if task_info.execution_id.startswith(execution_id)
    ]
    
    return {
        'execution': execution.__dict__,
        'task_executions': task_executions
    }

@app.get("/api/tasks/{task_id}/history")
async def get_task_history(task_id: str):
    """Get execution history for a specific task."""
    task_history = [
        task_info.__dict__
        for task_info in lineage_manager.task_executions.values()
        if task_info.task_id == task_id
    ]
    
    return {
        'task_id': task_id,
        'executions': sorted(task_history, key=lambda x: x['start_time'], reverse=True)
    }

@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get summary metrics for all executions."""
    total_executions = len(lineage_manager.executions)
    total_tasks = len(lineage_manager.task_executions)
    
    completed_tasks = len([
        t for t in lineage_manager.task_executions.values()
        if t.status == "completed"
    ])
    
    failed_tasks = len([
        t for t in lineage_manager.task_executions.values()
        if t.status == "failed"
    ])
    
    # Average task duration
    durations = [
        t.duration for t in lineage_manager.task_executions.values()
        if t.duration is not None
    ]
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    return {
        'total_executions': total_executions,
        'total_tasks': total_tasks,
        'completed_tasks': completed_tasks,
        'failed_tasks': failed_tasks,
        'success_rate': completed_tasks / total_tasks if total_tasks > 0 else 0,
        'average_task_duration': avg_duration
    }

@app.websocket("/ws/lineage")
async def websocket_lineage(websocket: WebSocket):
    """WebSocket endpoint for real-time lineage updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4. CLI Integration (`graflow/cli/lineage.py`)

Command-line interface for lineage operations:

```python
import click
import uvicorn
from graflow.api.server import app
from graflow.lineage.manager import get_lineage_manager

@click.group()
def lineage():
    """Lineage tracking and API commands."""
    pass

@lineage.command()
@click.option('--host', default='0.0.0.0', help='Host to bind the API server')
@click.option('--port', default=8000, help='Port to bind the API server')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host: str, port: int, reload: bool):
    """Start the lineage API server."""
    click.echo(f"Starting Graflow Lineage API server on {host}:{port}")
    uvicorn.run(
        "graflow.api.server:app",
        host=host,
        port=port,
        reload=reload
    )

@lineage.command()
def status():
    """Show current lineage tracking status."""
    manager = get_lineage_manager()
    
    click.echo("Graflow Lineage Status:")
    click.echo(f"  Active executions: {len(manager.executions)}")
    click.echo(f"  Total task executions: {len(manager.task_executions)}")
    click.echo(f"  Current execution: {manager.current_execution or 'None'}")

@lineage.command()
def clear():
    """Clear all lineage tracking data."""
    manager = get_lineage_manager()
    manager.graph.clear()
    manager.executions.clear()
    manager.task_executions.clear()
    manager.current_execution = None
    click.echo("Lineage data cleared.")
```

## Integration with Existing Components

### 1. WorkflowContext Integration

```python
# In graflow/core/workflow.py

class WorkflowContext:
    def execute(self, start_node: Optional[str] = None, max_steps: int = 10) -> None:
        """Execute the workflow with lineage tracking."""
        exec_context = ExecutionContext.create(self.graph, start_node, max_steps=max_steps)
        
        # Start lineage tracking
        exec_context.start_workflow_tracking(self.name)
        
        engine = WorkflowEngine()
        engine.execute(exec_context)
```

### 2. Task Decorator Enhancement

```python
# In graflow/core/decorators.py

def task(name: str = None, inject_context: bool = False, track_lineage: bool = True):
    """Enhanced task decorator with lineage tracking option."""
    def decorator(func):
        task_id = name or func.__name__
        wrapper = TaskWrapper(task_id, func, inject_context=inject_context)
        wrapper._track_lineage = track_lineage
        return wrapper
    return decorator
```

## Development Phases

### Phase 1: Core Lineage Manager (Week 1-2)
- [ ] Implement `LineageManager` class
- [ ] Add NetworkX graph management
- [ ] Create execution tracking data structures
- [ ] Add event system for real-time updates

### Phase 2: ExecutionContext Integration (Week 2-3)
- [ ] Integrate lineage tracking into `ExecutionContext`
- [ ] Add task execution hooks
- [ ] Update `WorkflowContext.execute()` method
- [ ] Test with existing workflows

### Phase 3: REST API Development (Week 3-4)
- [ ] Implement FastAPI server
- [ ] Create REST endpoints for graph data
- [ ] Add WebSocket support for real-time updates
- [ ] Implement metrics and history endpoints

### Phase 4: CLI and Tools (Week 4)
- [ ] Add CLI commands for lineage management
- [ ] Create API server startup command
- [ ] Add status and debugging commands

### Phase 5: Testing and Documentation (Week 5)
- [ ] Unit tests for all components
- [ ] Integration tests with existing workflows
- [ ] API documentation with OpenAPI/Swagger
- [ ] Usage examples and tutorials

## Future Enhancements

### Advanced Features
1. **Persistent Storage**: Add database backend for historical data
2. **Performance Metrics**: Detailed task performance analysis
3. **Alerting**: Notifications for failed tasks or performance issues
4. **Graph Analytics**: Bottleneck detection and optimization suggestions
5. **Multi-tenancy**: Support for multiple workflow projects

### Web GUI Features
1. **Interactive Graph Visualization**: Real-time graph rendering with D3.js or similar
2. **Execution Dashboard**: Overview of running and completed workflows
3. **Historical Analysis**: Timeline view and trend analysis
4. **Performance Profiling**: Task performance charts and bottleneck identification

## API Documentation

Once implemented, the REST API will provide the following endpoints:

- `GET /api/graph/current` - Current graph state
- `GET /api/executions` - List all executions  
- `GET /api/executions/{id}` - Execution details
- `GET /api/tasks/{id}/history` - Task execution history
- `GET /api/metrics/summary` - Performance metrics
- `WS /ws/lineage` - Real-time updates via WebSocket

This will enable rich web-based visualization and monitoring tools to be built on top of the graflow execution engine.
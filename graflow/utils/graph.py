"""Graph construction functionality for graflow."""

from __future__ import annotations

import base64
import logging
import math
import os
import random
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import networkx as nx

from graflow.core.graph import TaskGraph
from graflow.core.workflow import WorkflowContext, current_workflow_context

if TYPE_CHECKING:
    from graflow.core.task import Executable

logger = logging.getLogger(__name__)


def build_graph(start_node: Executable, context: Optional[WorkflowContext] = None) -> TaskGraph:
    """Build a NetworkX directed graph from an executable."""
    if context is None:
        # Use the current workflow context if not provided
        context = current_workflow_context()

    task_graph = context.graph
    cur_graph = task_graph.nx_graph()
    new_graph: nx.DiGraph = nx.DiGraph()
    visited: set[str] = set()

    def _build_graph_recursive(node: Executable) -> None:
        """Recursively build the graph from the executable."""
        if node.task_id in visited:
            return
        visited.add(node.task_id)

        new_graph.add_node(node.task_id, task=node)

        from graflow.core.task import ParallelGroup  # local import to avoid cycles

        # Only traverse successors if node exists in cur_graph
        if node.task_id in cur_graph.nodes:
            for successor in cur_graph.successors(node.task_id):
                successor_task = cur_graph.nodes[successor]["task"]
                new_graph.add_edge(node.task_id, successor)
                _build_graph_recursive(successor_task)

        # Handle ParallelGroup members - add all members to graph
        if isinstance(node, ParallelGroup):
            for member_task in node.tasks:
                member_id = member_task.task_id
                # Add member node
                new_graph.add_node(member_id, task=member_task)
                # Add edge from ParallelGroup to member
                new_graph.add_edge(node.task_id, member_id)
                # Recursively build graph for member
                _build_graph_recursive(member_task)

        # Only traverse predecessors if node exists in cur_graph
        if node.task_id in cur_graph.nodes:
            for predecessor in cur_graph.predecessors(node.task_id):
                predecessor_task = cur_graph.nodes[predecessor]["task"]
                new_graph.add_edge(predecessor, node.task_id)
                _build_graph_recursive(predecessor_task)

    _build_graph_recursive(start_node)
    task_graph._graph = new_graph
    return task_graph


def draw_task_graph(graph: nx.DiGraph, title: str = "Task Graph") -> None:
    """Draw a task graph using matplotlib."""
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        edge_color="black",
        arrows=True,
    )
    plt.title(title)
    plt.show()


def visualize_dependencies(graph: nx.DiGraph) -> None:
    """Visualize task dependencies."""
    print("=== Dependencies ===")
    for node in graph.nodes():
        successors = list(graph.successors(node))
        if successors:
            print(f"{node} >> {' >> '.join(str(s) for s in successors)}")
        else:
            print(f"{node} (no dependencies)")


def show_graph_info(graph: nx.DiGraph) -> None:
    """Display information about the task graph."""

    print("=== Graph Information ===")
    print(f"Nodes: {list(graph.nodes())}")
    print(f"Edges: {list(graph.edges())}")

    # Cycle detection
    try:
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            print(f"Cycles detected: {cycles}")
        else:
            print("No cycles detected")
    except Exception as e:
        logger.error("Error detecting cycles in graph", exc_info=True)
        print(f"Error detecting cycles: {e}")


# ASCII Drawing Utilities


class VertexViewer:
    """VertexViewer class for ASCII graph drawing."""

    HEIGHT = 3

    def __init__(self, name: str) -> None:
        self._h = self.HEIGHT
        self._w = len(name) + 2

    @property
    def h(self) -> int:
        return self._h

    @property
    def w(self) -> int:
        return self._w


class AsciiCanvas:
    """Class for drawing in ASCII."""

    def __init__(self, cols: int, lines: int) -> None:
        if cols <= 1 or lines <= 1:
            msg = "Canvas dimensions should be > 1"
            raise ValueError(msg)

        self.cols = cols
        self.lines = lines
        self.width = cols  # Alias for compatibility
        self.canvas: list[list[str]] = [[" "] * cols for line in range(lines)]

    def draw(self) -> str:
        lines = map("".join, self.canvas)
        return os.linesep.join(lines)

    def point(self, x: int, y: int, char: str) -> None:
        if len(char) != 1:
            msg = "char should be a single character"
            raise ValueError(msg)
        if x >= self.cols or x < 0:
            msg = "x should be >= 0 and < number of columns"
            raise ValueError(msg)
        if y >= self.lines or y < 0:
            msg = "y should be >= 0 and < number of lines"
            raise ValueError(msg)

        self.canvas[y][x] = char

    def line(self, x0: int, y0: int, x1: int, y1: int, char: str) -> None:
        if x0 > x1:
            x1, x0 = x0, x1
            y1, y0 = y0, y1

        dx = x1 - x0
        dy = y1 - y0

        if dx == 0 and dy == 0:
            self.point(x0, y0, char)
        elif abs(dx) >= abs(dy):
            for x in range(x0, x1 + 1):
                y = y0 if dx == 0 else y0 + round((x - x0) * dy / float(dx))
                self.point(x, y, char)
        elif y0 < y1:
            for y in range(y0, y1 + 1):
                x = x0 if dy == 0 else x0 + round((y - y0) * dx / float(dy))
                self.point(x, y, char)
        else:
            for y in range(y1, y0 + 1):
                x = x0 if dy == 0 else x1 + round((y - y1) * dx / float(dy))
                self.point(x, y, char)

    def text(self, x: int, y: int, text: str) -> None:
        for i, char in enumerate(text):
            self.point(x + i, y, char)

    def box(self, x0: int, y0: int, width: int, height: int) -> None:
        if width <= 1 or height <= 1:
            msg = "Box dimensions should be > 1"
            raise ValueError(msg)

        width -= 1
        height -= 1

        for x in range(x0, x0 + width):
            self.point(x, y0, "-")
            self.point(x, y0 + height, "-")

        for y in range(y0, y0 + height):
            self.point(x0, y, "|")
            self.point(x0 + width, y, "|")

        self.point(x0, y0, "+")
        self.point(x0 + width, y0, "+")
        self.point(x0, y0 + height, "+")
        self.point(x0 + width, y0 + height, "+")


class _EdgeViewer:
    def __init__(self) -> None:
        self.pts: list[tuple[float, float]] = []

    def setpath(self, pts: list[tuple[float, float]]) -> None:
        self.pts = pts


def _build_sugiyama_layout(vertices: dict[str, str], edges: list[tuple[str, str]]) -> Any:
    try:
        from grandalf.graphs import Edge, Graph, Vertex
        from grandalf.layouts import SugiyamaLayout
        from grandalf.routing import route_with_lines
    except ImportError as exc:
        msg = "Install grandalf to draw graphs: `pip install grandalf`."
        raise ImportError(msg) from exc

    vertices_ = {id_: Vertex(f" {data} ") for id_, data in vertices.items()}
    edges_ = [Edge(vertices_[s], vertices_[e]) for s, e in edges]
    vertices_list = vertices_.values()
    graph = Graph(vertices_list, edges_)

    for vertex in vertices_list:
        vertex.view = VertexViewer(vertex.data)  # type: ignore

    # NOTE: determine min box length to create the best layout
    minw = min(v.view.w for v in vertices_list)  # type: ignore

    for edge in edges_:
        edge.view = _EdgeViewer()  # type: ignore

    sug = SugiyamaLayout(graph.C[0])
    graph = graph.C[0]
    roots = list(filter(lambda x: len(x.e_in()) == 0, graph.sV))

    sug.init_all(roots=roots, optimize=True)

    sug.yspace = VertexViewer.HEIGHT
    sug.xspace = minw
    sug.route_edge = route_with_lines  # type: ignore

    sug.draw()

    return sug


def _is_parallel_group(node_id: str, graph: nx.DiGraph) -> bool:
    """Check if a node is a ParallelGroup.

    Args:
        node_id: Node identifier
        graph: NetworkX graph containing the node

    Returns:
        True if the node is a ParallelGroup, False otherwise
    """
    # Get node data to check the actual task type
    node_data = graph.nodes.get(node_id)
    if node_data:
        task = node_data.get("task")
        if task is not None:
            # Import here to avoid circular dependency
            from graflow.core.task import ParallelGroup

            return isinstance(task, ParallelGroup)
    return False


def _draw_double_box(canvas: AsciiCanvas, x0: int, y0: int, width: int, height: int) -> None:
    """Draw a double-line box for ParallelGroup nodes.

    Args:
        canvas: ASCII canvas to draw on
        x0: X coordinate of top-left corner
        y0: Y coordinate of top-left corner
        width: Box width
        height: Box height
    """
    if width <= 1 or height <= 1:
        msg = "Box dimensions should be > 1"
        raise ValueError(msg)

    width -= 1
    height -= 1

    # Top and bottom lines with double-line style
    for x in range(x0, x0 + width):
        canvas.point(x, y0, "═")
        canvas.point(x, y0 + height, "═")

    # Left and right lines
    for y in range(y0, y0 + height):
        canvas.point(x0, y, "║")
        canvas.point(x0 + width, y, "║")

    # Corners
    canvas.point(x0, y0, "╔")
    canvas.point(x0 + width, y0, "╗")
    canvas.point(x0, y0 + height, "╚")
    canvas.point(x0 + width, y0 + height, "╝")


def _classify_parallel_group_edges(group_node_id: str, graph: nx.DiGraph) -> tuple[list[str], list[str]]:
    """Classify edges from a ParallelGroup into internal and external.

    Args:
        group_node_id: Node ID of the ParallelGroup
        graph: NetworkX graph containing the node

    Returns:
        Tuple of (internal_tasks, external_tasks)
        - internal_tasks: Tasks that are executed in parallel (in group.tasks)
        - external_tasks: Tasks that are executed after the group completes
    """
    from graflow.core.task import ParallelGroup

    node_data = graph.nodes.get(group_node_id)
    if not node_data:
        return [], []

    task = node_data.get("task")
    if not isinstance(task, ParallelGroup):
        return [], []

    # Get the list of parallel tasks from the ParallelGroup
    parallel_task_ids = [t.task_id for t in task.tasks]

    # Get external successors of this ParallelGroup
    successors = list(graph.successors(group_node_id))
    external_tasks = [s for s in successors if s not in parallel_task_ids]

    # All members are considered internal tasks regardless of explicit edges
    internal_tasks = parallel_task_ids

    return internal_tasks, external_tasks


def _collect_parallel_group_internal_tasks(group_node_id: str, graph: nx.DiGraph) -> set[str]:
    """Collect all tasks that should be inside the ParallelGroup container.

    This includes:
    1. Direct child tasks of the ParallelGroup (group.tasks)
    2. All tasks reachable from child tasks, up to (but not including) external successors

    Args:
        group_node_id: Node ID of the ParallelGroup
        graph: NetworkX graph containing the node

    Returns:
        Set of task IDs that should be inside the container
    """
    from graflow.core.task import ParallelGroup

    node_data = graph.nodes.get(group_node_id)
    if not node_data:
        return set()

    task = node_data.get("task")
    if not isinstance(task, ParallelGroup):
        return set()

    # Get the list of parallel tasks from the ParallelGroup
    parallel_task_ids = {t.task_id for t in task.tasks}

    # Get external successors (tasks that execute after the group completes)
    _, external_tasks = _classify_parallel_group_edges(group_node_id, graph)
    external_task_set = set(external_tasks)

    # Collect all tasks reachable from parallel tasks, excluding external tasks
    internal_tasks = set()
    visited = set()

    def dfs(task_id: str) -> None:
        """Depth-first search to collect internal tasks."""
        if task_id in visited:
            return
        if task_id in external_task_set:
            # Stop at external successors (don't include them)
            return
        if task_id not in graph.nodes():
            return

        visited.add(task_id)
        internal_tasks.add(task_id)

        # Recursively visit successors
        for successor in graph.successors(task_id):
            dfs(successor)

    # Start DFS from each parallel task
    for task_id in parallel_task_ids:
        if task_id in graph.nodes():
            dfs(task_id)

    return internal_tasks


def _transform_graph_with_start_end_nodes(graph: nx.DiGraph) -> tuple[nx.DiGraph, set[str]]:
    """Transform graph to replace ParallelGroup nodes with start and end nodes.

    For each ParallelGroup:
    - Remove the ParallelGroup node
    - Create a "ParallelGroup_X" node (reusing the original name)
    - Create a "↣ ParallelGroup_X" node (merge/join point)
    - Add edges: predecessors -> start node (ParallelGroup_X)
    - Add edges: start node -> parallel tasks
    - Add edges: parallel tasks -> end node (↣ ParallelGroup_X)
    - Add edges: end node -> external successors

    Args:
        graph: Original graph with ParallelGroup nodes

    Returns:
        Tuple of (transformed graph, set of ParallelGroup node names for double-box rendering)
    """
    from graflow.core.task import ParallelGroup

    transformed = graph.copy()
    parallel_group_nodes: set[str] = set()

    for node in list(graph.nodes()):
        if not _is_parallel_group(str(node), graph):
            continue

        node_data = graph.nodes.get(node)
        if not node_data:
            continue

        task = node_data.get("task")
        if not isinstance(task, ParallelGroup):
            continue

        # Get parallel tasks and external tasks
        parallel_task_ids = [t.task_id for t in task.tasks]
        predecessors = list(graph.predecessors(node))
        successors = list(graph.successors(node))

        # External successors exclude parallel members (tracked separately)
        external_tasks = [s for s in successors if s not in parallel_task_ids]
        internal_tasks = parallel_task_ids

        # Create start and end node names
        start_node_name = str(node)  # Use the original ParallelGroup name
        end_node_name = f"↣ {node}"  # ↣ indicates merge/join point

        # Track these nodes for double-box rendering
        parallel_group_nodes.add(start_node_name)
        parallel_group_nodes.add(end_node_name)

        # Remove ParallelGroup node
        transformed.remove_node(node)

        # Add start node (reusing ParallelGroup name) and end node (merge/join point)
        transformed.add_node(start_node_name)
        transformed.add_node(end_node_name)

        # Add edges: predecessors -> start node
        for pred in predecessors:
            transformed.add_edge(pred, start_node_name)

        # Add edges: start node -> parallel tasks
        for task_id in internal_tasks:
            if task_id in transformed.nodes():
                transformed.add_edge(start_node_name, task_id)

        # Add edges: parallel tasks -> end node
        for task_id in internal_tasks:
            if task_id in transformed.nodes():
                transformed.add_edge(task_id, end_node_name)

        # Add edges: end node -> external successors
        for succ in external_tasks:
            if succ in transformed.nodes():
                transformed.add_edge(end_node_name, succ)

    return transformed, parallel_group_nodes


def draw_ascii(graph: nx.DiGraph) -> str:  # noqa: PLR0912
    """Draw a NetworkX DiGraph in ASCII format.

    Args:
        graph: NetworkX directed graph to draw

    Returns:
        ASCII representation of the graph
    """
    if not graph.nodes():
        return "Empty graph"

    if not graph.edges():
        # Handle graph with nodes but no edges - create simple vertical layout
        nodes = list(graph.nodes())
        max_width = max(len(str(node)) for node in nodes) + 2
        separator = "+" + "-" * max_width + "+\n"
        node_boxes = []

        for node in nodes:
            node_str = str(node)[: max_width - 2]  # Truncate if too long
            padding = max_width - len(node_str)
            left_pad = padding // 2
            right_pad = padding - left_pad
            node_boxes.append(f"|{' ' * left_pad}{node_str}{' ' * right_pad}|")

        return separator + f"\n{separator}".join(node_boxes) + f"\n{separator}"

    # Transform graph to use start/end nodes for ParallelGroups
    graph, parallel_group_nodes = _transform_graph_with_start_end_nodes(graph)

    # Convert NetworkX graph to format expected by _build_sugiyama_layout
    vertices = {str(node): str(node) for node in graph.nodes()}
    edges = [(str(u), str(v)) for u, v in graph.edges()]

    # NOTE: coordinates might be negative, so we need to shift
    # everything to the positive plane before we actually draw it.
    xlist: list[float] = []
    ylist: list[float] = []

    sug = _build_sugiyama_layout(vertices, edges)

    for vertex in sug.g.sV:
        # NOTE: moving boxes w/2 to the left
        xlist.extend(
            (
                vertex.view.xy[0] - vertex.view.w / 2.0,
                vertex.view.xy[0] + vertex.view.w / 2.0,
            )
        )
        ylist.extend((vertex.view.xy[1], vertex.view.xy[1] + vertex.view.h))

    for edge in sug.g.sE:
        for x, y in edge.view.pts:
            xlist.append(x)
            ylist.append(y)

    if not xlist or not ylist:
        msg = "No valid layout coordinates found"
        raise ValueError(msg)

    minx = min(xlist)
    miny = min(ylist)
    maxx = max(xlist)
    maxy = max(ylist)

    canvas_cols = math.ceil(math.ceil(maxx) - math.floor(minx)) + 1
    canvas_lines = round(maxy - miny)

    canvas = AsciiCanvas(canvas_cols, canvas_lines)

    # FIRST: Draw edges
    # NOTE: Draw edges first so that node boxes can overwrite them
    for edge in sug.g.sE:
        if len(edge.view.pts) <= 1:
            msg = "Not enough points to draw an edge"
            raise ValueError(msg)

        # Use '*' for all edges uniformly
        edge_char = "*"

        for index in range(1, len(edge.view.pts)):
            start = edge.view.pts[index - 1]
            end = edge.view.pts[index]

            start_x = round(start[0] - minx)
            start_y = round(start[1] - miny)
            end_x = round(end[0] - minx)
            end_y = round(end[1] - miny)

            if start_x < 0 or start_y < 0 or end_x < 0 or end_y < 0:
                msg = f"Invalid edge coordinates: start_x={start_x}, start_y={start_y}, end_x={end_x}, end_y={end_y}"
                raise ValueError(msg)

            canvas.line(start_x, start_y, end_x, end_y, edge_char)

    # SECOND: Draw task boxes (which will overwrite edges)
    for vertex in sug.g.sV:
        # NOTE: moving boxes w/2 to the left
        x = vertex.view.xy[0] - vertex.view.w / 2.0
        y = vertex.view.xy[1]

        # Get node ID from vertex data (strip spaces added by _build_sugiyama_layout)
        node_id = vertex.data.strip()

        # Use double-line box for ParallelGroup start/end nodes
        if node_id in parallel_group_nodes:
            _draw_double_box(
                canvas,
                round(x - minx),
                round(y - miny),
                vertex.view.w,
                vertex.view.h,
            )
        else:
            canvas.box(
                round(x - minx),
                round(y - miny),
                vertex.view.w,
                vertex.view.h,
            )

        canvas.text(round(x - minx) + 1, round(y - miny) + 1, vertex.data)

    return canvas.draw()


def _transform_graph_for_container_view(graph: nx.DiGraph) -> tuple[nx.DiGraph, dict[str, dict[str, Any]]]:
    """Transform graph to support container-based ParallelGroup view.

    Keeps ParallelGroup nodes and maintains edges at the group level (not individual tasks).

    Args:
        graph: Original graph with ParallelGroup nodes

    Returns:
        - Transformed graph with group-level edges
        - Metadata dict mapping group_id -> {tasks, predecessors, successors, label}
    """
    transformed_graph = graph.copy()
    parallel_groups: dict[str, dict[str, Any]] = {}

    for node in list(graph.nodes()):
        if _is_parallel_group(str(node), graph):
            # Extract group metadata
            node_data = graph.nodes[node]
            task = node_data.get("task")
            if task is None:
                continue

            from graflow.core.task import ParallelGroup

            if not isinstance(task, ParallelGroup):
                continue

            # Get predecessors and successors of the ParallelGroup
            predecessors = list(graph.predecessors(node))

            # Classify successors
            _internal_tasks, external_tasks = _classify_parallel_group_edges(str(node), graph)

            # Collect all tasks that should be inside the container
            # This includes direct child tasks and all reachable internal tasks
            tasks_in_container = _collect_parallel_group_internal_tasks(str(node), graph)

            # Store metadata
            parallel_groups[str(node)] = {
                "tasks": list(tasks_in_container),
                "predecessors": predecessors,
                "successors": external_tasks,
                "label": str(node),
            }

            # Remove edges to internal tasks from external nodes (including ParallelGroup)
            # But keep edges between internal tasks (e.g., transform_a -> subtask_a)
            for ptask in tasks_in_container:
                for pred in list(transformed_graph.predecessors(ptask)):
                    # Keep edges from other internal tasks
                    if pred not in tasks_in_container:
                        transformed_graph.remove_edge(pred, ptask)

            # Remove edges from internal tasks ONLY to the ParallelGroup's external successors
            # (these edges should come from the ParallelGroup instead)
            # But keep edges to other downstream tasks (e.g., transform_a -> subtask_a)
            for ptask in tasks_in_container:
                for succ in list(transformed_graph.successors(ptask)):
                    # Only remove if successor is a direct external successor of ParallelGroup
                    if succ in external_tasks:
                        transformed_graph.remove_edge(ptask, succ)

    return transformed_graph, parallel_groups


def draw_mermaid(  # noqa: PLR0912
    graph: nx.DiGraph,
    title: str = "Graph",
    *,
    with_styles: bool = True,
    node_colors: Optional[dict[Any, str]] = None,
    wrap_label_n_words: int = 9,
) -> str:
    """Draw a NetworkX DiGraph in Mermaid format.

    Args:
        graph: NetworkX directed graph to draw
        title: Title for the graph
        with_styles: Whether to include node styling
        node_colors: Custom colors for nodes (optional)
        wrap_label_n_words: Words to wrap edge labels at

    Returns:
        Mermaid diagram syntax
    """
    # Mermaid reserved keywords that cannot be used as node IDs
    reserved_keywords = {
        "graph",
        "subgraph",
        "direction",
        "click",
        "class",
        "classDef",
        "style",
        "linkStyle",
        "fill",
        "stroke",
        "color",
        "end",
        "start",
        "TD",
        "TB",
        "BT",
        "RL",
        "LR",
        "flowchart",
    }

    # Create unique IDs for all node labels
    node_id_map: dict[str, str] = {}
    # Pre-populate used_ids with all escaped node labels to avoid conflicts
    used_ids: set[str] = {re.sub(r"[^a-zA-Z-_0-9]", "_", str(node)) for node in graph.nodes()}

    def _escape_node_label(node_label: str) -> str:
        """Generate a unique Mermaid-compatible ID for a node label."""
        if node_label in node_id_map:
            return node_id_map[node_label]

        # Escape non-alphanumeric characters
        escaped = re.sub(r"[^a-zA-Z-_0-9]", "_", node_label)

        # Start with the escaped label as candidate
        candidate = escaped

        # If it's a reserved keyword or already used, add suffix
        if escaped in reserved_keywords or candidate in used_ids:
            counter = 1
            while True:
                candidate = f"{escaped}_{counter}"
                if candidate not in used_ids:
                    break
                counter += 1

        # Store mapping and mark as used
        node_id_map[node_label] = candidate
        used_ids.add(candidate)
        return candidate

    if not graph.nodes():
        return "graph TD;\n    EmptyGraph[Empty Graph];\n"

    # Transform graph for container view
    display_graph, parallel_groups = _transform_graph_for_container_view(graph)

    # Initialize Mermaid graph
    if with_styles:
        mermaid = "---\nconfig:\n  flowchart:\n    curve: linear\n---\n"
    else:
        mermaid = ""

    mermaid += "graph TD;\n"

    # Add title comment
    if title != "Graph":
        mermaid += f"    %% {title}\n"

    # Add nodes with proper formatting
    first_node = None
    last_node = None

    # Try to identify first and last nodes based on graph structure
    nodes_list = list(display_graph.nodes())
    if nodes_list:
        # Find nodes with no predecessors (potential start nodes)
        start_candidates = [n for n in nodes_list if display_graph.in_degree(n) == 0]
        if start_candidates:
            first_node = start_candidates[0]

        # Find nodes with no successors (potential end nodes)
        end_candidates = [n for n in nodes_list if display_graph.out_degree(n) == 0]
        if end_candidates:
            last_node = end_candidates[-1]  # select last node

    # Track which nodes are in parallel groups
    nodes_in_groups: set[str] = set()
    for group_info in parallel_groups.values():
        nodes_in_groups.update(str(task_id) for task_id in group_info["tasks"])

    # Track parallel group IDs
    parallel_group_ids = set(parallel_groups.keys())

    # Add parallel group subgraphs
    for group_id, group_info in parallel_groups.items():
        group_id_escaped = _escape_node_label(group_id)

        mermaid += f'    subgraph {group_id_escaped}[" {group_info["label"]} "]\n'
        mermaid += "        direction TB\n"

        # Add tasks within the subgraph
        for task_id in group_info["tasks"]:
            task_str = str(task_id)
            if task_str not in display_graph.nodes():
                continue
            task_node_id = _escape_node_label(task_str)
            task_label = task_str.replace('"', '\\"')

            mermaid += f"        {task_node_id}[{task_label}];\n"

        mermaid += "    end\n"
        # Apply parallel group styling to subgraph
        mermaid += f"    class {group_id_escaped} parallelGroup;\n"

    # Add regular nodes (not in parallel groups and not ParallelGroup nodes themselves)
    for node in display_graph.nodes():
        node_str = str(node)

        if node_str in nodes_in_groups:
            continue  # Already added in subgraph

        # Skip ParallelGroup nodes - they are represented as subgraphs, not individual nodes
        if node_str in parallel_group_ids:
            continue

        node_id = _escape_node_label(node_str)
        node_label = node_str.replace('"', '\\"')  # Escape quotes

        # Handle special formatting for different node types
        if node == first_node:
            mermaid += f"    {node_id}([{node_label}]):::first;\n"
        elif node == last_node:
            mermaid += f"    {node_id}([{node_label}]):::last;\n"
        else:
            mermaid += f"    {node_id}[{node_label}];\n"

    # Add edges
    # Handle edges involving ParallelGroup nodes specially - connect directly to subgraph
    for u, v in display_graph.edges():
        u_str = str(u)
        v_str = str(v)

        # Case 1: external_node -> ParallelGroup
        # Draw as: external_node -> subgraph (direct edge to subgraph)
        if v_str in parallel_group_ids and u_str not in parallel_group_ids:
            u_id = _escape_node_label(u_str)
            v_id = _escape_node_label(v_str)
            edge_data = display_graph.get_edge_data(u, v)

            # Connect directly to the subgraph
            if edge_data and edge_data.get("label"):
                label = str(edge_data["label"])
                mermaid += f"    {u_id} -- {label} --> {v_id};\n"
            else:
                mermaid += f"    {u_id} --> {v_id};\n"
            continue

        # Case 2: ParallelGroup -> external_node
        # Draw as: subgraph -> external_node (direct edge from subgraph)
        if u_str in parallel_group_ids and v_str not in parallel_group_ids and v_str not in nodes_in_groups:
            u_id = _escape_node_label(u_str)
            v_id = _escape_node_label(v_str)
            edge_data = display_graph.get_edge_data(u, v)

            # Connect directly from the subgraph
            if edge_data and edge_data.get("label"):
                label = str(edge_data["label"])
                mermaid += f"    {u_id} -- {label} --> {v_id};\n"
            else:
                mermaid += f"    {u_id} --> {v_id};\n"
            continue

        # Case 3: ParallelGroup -> internal_task
        # Skip these edges - internal structure handled by subgraph
        if u_str in parallel_group_ids and v_str in nodes_in_groups:
            continue

        # Case 4: Regular edges (not involving ParallelGroup nodes)
        if u_str not in parallel_group_ids and v_str not in parallel_group_ids:
            u_id = _escape_node_label(u_str)
            v_id = _escape_node_label(v_str)

            # Check if edge has data/label
            edge_data = display_graph.get_edge_data(u, v)
            if edge_data and edge_data.get("label"):
                label = str(edge_data["label"])
                # Wrap long labels
                if len(label.split()) > wrap_label_n_words:
                    words = label.split()
                    wrapped_label = "<br>".join(
                        " ".join(words[i : i + wrap_label_n_words]) for i in range(0, len(words), wrap_label_n_words)
                    )
                    mermaid += f"    {u_id} -- {wrapped_label} --> {v_id};\n"
                else:
                    mermaid += f"    {u_id} -- {label} --> {v_id};\n"
            else:
                # Normal edge
                mermaid += f"    {u_id} --> {v_id};\n"

    # Add custom styles
    if with_styles:
        mermaid += "    classDef first fill:#e1f5fe,stroke:#01579b,stroke-width:2px;\n"
        mermaid += "    classDef last fill:#fff3e0,stroke:#e65100,stroke-width:2px;\n"
        mermaid += "    classDef parallelGroup fill:#f3e5f5,stroke:#4a148c,stroke-width:3px;\n"
        mermaid += "    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;\n"

        # Add custom node colors if provided
        if node_colors:
            for node, color in node_colors.items():
                node_id = _escape_node_label(str(node))
                mermaid += f"    classDef {node_id}_style fill:{color};\n"
                mermaid += f"    class {node_id} {node_id}_style;\n"

    return mermaid


def create_agraph(  # noqa: PLR0912
    graph: nx.DiGraph,
    node_labels: Optional[dict[Any, str]] = None,
    node_colors: Optional[dict[Any, str]] = None,
) -> Any:
    """Create a pygraphviz AGraph object from a NetworkX DiGraph.

    Args:
        graph: NetworkX directed graph to draw
        node_labels: Custom labels for nodes (optional)
        node_colors: Custom colors for nodes (optional)

    Returns:
        pygraphviz AGraph object

    Raises:
        ImportError: If pygraphviz is not installed
    """
    try:
        import pygraphviz as pgv
    except ImportError as exc:
        msg = "Install pygraphviz to draw graphs: `pip install pygraphviz`."
        raise ImportError(msg) from exc

    if not graph.nodes():
        # Create a simple "empty graph" visualization
        viz = pgv.AGraph(directed=True)
        viz.add_node("Empty", label="Empty Graph", style="filled", fillcolor="lightgray")
        return viz

    # Transform graph for container view
    display_graph, parallel_groups = _transform_graph_for_container_view(graph)

    # Create pygraphviz graph with compound=true to support edges to/from clusters
    viz = pgv.AGraph(directed=True, nodesep=0.5, ranksep=1.2, compound=True)

    # Track which nodes are in parallel groups
    nodes_in_groups: set[str] = set()
    for group_info in parallel_groups.values():
        nodes_in_groups.update(group_info["tasks"])

    # Track parallel group IDs and map them to cluster names
    parallel_group_ids = set(parallel_groups.keys())
    group_to_cluster: dict[str, str] = {}

    # Add parallel group clusters
    for idx, (group_id, group_info) in enumerate(parallel_groups.items()):
        cluster_name = f"cluster_{idx}"
        group_to_cluster[group_id] = cluster_name

        viz.add_subgraph(
            name=cluster_name,
            label=group_info["label"],
            style="filled",
            fillcolor="#f3e5f5",
            color="#4a148c",
            penwidth=2.5,
            labeljust="l",  # Left-justify label
            labelloc="t",  # Place label at top
        )

        # Add tasks to the cluster
        cluster = viz.get_subgraph(cluster_name)
        assert cluster is not None
        for task_id in group_info["tasks"]:
            if task_id not in display_graph.nodes():
                continue

            label = node_labels.get(task_id, str(task_id)) if node_labels else str(task_id)
            color = node_colors.get(task_id, "lightblue") if node_colors else "lightblue"

            cluster.add_node(
                str(task_id),
                label=f"<<B>{label}</B>>",
                style="filled",
                fillcolor=color,
                fontsize=12,
                fontname="arial",
            )

    # Add regular nodes (not in parallel groups and not ParallelGroup nodes themselves)
    for node in display_graph.nodes():
        node_str = str(node)

        if node_str in nodes_in_groups:
            continue  # Already added in cluster

        # Skip ParallelGroup nodes - they are represented as clusters, not individual nodes
        if node_str in parallel_group_ids:
            continue

        label = node_labels.get(node, node_str) if node_labels else node_str
        color = node_colors.get(node, "lightblue") if node_colors else "lightblue"

        viz.add_node(
            node_str,
            label=f"<<B>{label}</B>>",
            style="filled",
            fillcolor=color,
            fontsize=12,
            fontname="arial",
        )

    # Add edges with special handling for ParallelGroup nodes
    for u, v in display_graph.edges():
        u_str = str(u)
        v_str = str(v)

        # Determine source and target for the edge
        # If source/target is a ParallelGroup, use lhead/ltail to connect to cluster
        edge_attrs = {"fontsize": "10", "fontname": "arial"}

        # Case 1: external_node -> ParallelGroup
        if v_str in parallel_group_ids and u_str not in parallel_group_ids:
            cluster_name = group_to_cluster[v_str]
            # Find a representative node in the cluster to use as the target
            tasks = parallel_groups[v_str]["tasks"]
            if tasks:
                representative = tasks[0]
                edge_attrs["lhead"] = cluster_name
                viz.add_edge(u_str, representative, **edge_attrs)
            continue

        # Case 2: ParallelGroup -> external_node
        if u_str in parallel_group_ids and v_str not in parallel_group_ids:
            cluster_name = group_to_cluster[u_str]
            # Find a representative node in the cluster to use as the source
            tasks = parallel_groups[u_str]["tasks"]
            if tasks:
                representative = tasks[0]
                edge_attrs["ltail"] = cluster_name
                viz.add_edge(representative, v_str, **edge_attrs)
            continue

        # Case 3: Regular edges (not involving ParallelGroup nodes)
        if u_str not in parallel_group_ids and v_str not in parallel_group_ids:
            viz.add_edge(u_str, v_str, **edge_attrs)

    return viz


def draw_dot(
    graph: nx.DiGraph,
    node_labels: Optional[dict[Any, str]] = None,
    node_colors: Optional[dict[Any, str]] = None,
) -> str:
    """Generate DOT format string from a NetworkX DiGraph.

    Args:
        graph: NetworkX directed graph to draw
        node_labels: Custom labels for nodes (optional)
        node_colors: Custom colors for nodes (optional)

    Returns:
        DOT format string

    Raises:
        ImportError: If pygraphviz is not installed
    """
    viz = create_agraph(graph, node_labels=node_labels, node_colors=node_colors)
    try:
        return viz.string()
    finally:
        viz.close()


def draw_png(
    graph: nx.DiGraph,
    output_path: Optional[str] = None,
    node_labels: Optional[dict[Any, str]] = None,
    node_colors: Optional[dict[Any, str]] = None,
) -> Optional[bytes]:
    """Draw a NetworkX DiGraph as PNG using pygraphviz.

    Args:
        graph: NetworkX directed graph to draw
        output_path: Path to save PNG file (optional)
        node_labels: Custom labels for nodes (optional)
        node_colors: Custom colors for nodes (optional)

    Returns:
        PNG bytes if output_path is None, otherwise None
    """
    viz = create_agraph(graph, node_labels=node_labels, node_colors=node_colors)
    try:
        return viz.draw(output_path, format="png", prog="dot")
    finally:
        viz.close()


def draw_mermaid_png(
    graph: nx.DiGraph,
    output_path: Optional[str] = None,
    *,
    title: str = "Graph",
    with_styles: bool = True,
    node_colors: Optional[dict[Any, str]] = None,
    wrap_label_n_words: int = 9,
    background_color: Optional[str] = "white",
    max_retries: int = 1,
    retry_delay: float = 1.0,
) -> bytes:
    """Draw a NetworkX DiGraph as PNG using Mermaid API rendering.

    Args:
        graph: NetworkX directed graph to draw
        output_path: Path to save PNG file (optional)
        title: Title for the graph
        with_styles: Whether to include node styling
        node_colors: Custom colors for nodes (optional)
        wrap_label_n_words: Words to wrap edge labels at
        background_color: Background color of the image
        max_retries: Maximum number of retries
        retry_delay: Delay between retries

    Returns:
        PNG image bytes

    Raises:
        ImportError: If required dependencies are not installed
    """
    # Generate Mermaid syntax
    mermaid_syntax = draw_mermaid(
        graph,
        title=title,
        with_styles=with_styles,
        node_colors=node_colors,
        wrap_label_n_words=wrap_label_n_words,
    )

    img_bytes = _render_mermaid_using_api(
        mermaid_syntax,
        output_path=output_path,
        background_color=background_color,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    return img_bytes


def _render_mermaid_using_api(
    mermaid_syntax: str,
    *,
    output_path: Optional[str] = None,
    background_color: Optional[str] = "white",
    file_type: str = "png",
    max_retries: int = 1,
    retry_delay: float = 1.0,
) -> bytes:
    """Renders Mermaid graph using the Mermaid.INK API."""
    try:
        import requests
    except ImportError as e:
        msg = "Install the `requests` module to use the Mermaid.INK API: `pip install requests`."
        raise ImportError(msg) from e

    # Use Mermaid API to render the image
    mermaid_syntax_encoded = base64.b64encode(mermaid_syntax.encode("utf8")).decode("ascii")

    # Check if the background color is a hexadecimal color code using regex
    if background_color is not None:
        hex_color_pattern = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")
        if not hex_color_pattern.match(background_color):
            background_color = f"!{background_color}"

    image_url = f"https://mermaid.ink/img/{mermaid_syntax_encoded}?type={file_type}&bgColor={background_color}"

    error_msg_suffix = (
        "To resolve this issue:\n"
        "1. Check your internet connection and try again\n"
        "2. Try with higher retry settings: "
        "`draw_mermaid_png(..., max_retries=5, retry_delay=2.0)`\n"
        "3. Try using the pygraphviz PNG renderer instead: `draw_png(...)`"
    )

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code == requests.codes.ok:
                img_bytes = response.content
                if output_path is not None:
                    Path(output_path).write_bytes(response.content)

                return img_bytes

            # If we get a server error (5xx), retry
            if 500 <= response.status_code < 600 and attempt < max_retries:
                # Exponential backoff with jitter
                sleep_time = retry_delay * (2**attempt) * (0.5 + 0.5 * random.random())
                time.sleep(sleep_time)
                continue

            # For other status codes, fail immediately
            msg = (
                "Failed to reach https://mermaid.ink/ API while trying to render "
                f"your graph. Status code: {response.status_code}.\n\n"
            ) + error_msg_suffix
            raise ValueError(msg)

        except (requests.RequestException, requests.Timeout) as e:
            if attempt < max_retries:
                # Exponential backoff with jitter
                sleep_time = retry_delay * (2**attempt) * (0.5 + 0.5 * random.random())
                time.sleep(sleep_time)
            else:
                msg = (
                    "Failed to reach https://mermaid.ink/ API while trying to render "
                    f"your graph after {max_retries} retries. "
                ) + error_msg_suffix
                raise ValueError(msg) from e

    # This should not be reached, but just in case
    msg = (
        f"Failed to reach https://mermaid.ink/ API while trying to render your graph after {max_retries} retries. "
    ) + error_msg_suffix
    raise ValueError(msg)

"""Graph construction functionality for graflow."""

from __future__ import annotations

import base64
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

        for successor in cur_graph.successors(node.task_id):
            successor_task = cur_graph.nodes[successor]["task"]
            new_graph.add_edge(node.task_id, successor)
            _build_graph_recursive(successor_task)

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
    except Exception:
        print("Error detecting cycles")


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
        from grandalf.graphs import Edge, Graph, Vertex  # noqa: PLC0415
        from grandalf.layouts import SugiyamaLayout  # noqa: PLC0415
        from grandalf.routing import route_with_lines  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install grandalf to draw graphs: `pip install grandalf`."
        raise ImportError(msg) from exc

    vertices_ = {id_: Vertex(f" {data} ") for id_, data in vertices.items()}
    edges_ = [Edge(vertices_[s], vertices_[e]) for s, e in edges]
    vertices_list = vertices_.values()
    graph = Graph(vertices_list, edges_)

    for vertex in vertices_list:
        vertex.view = VertexViewer(vertex.data) # type: ignore

    # NOTE: determine min box length to create the best layout
    minw = min(v.view.w for v in vertices_list) # type: ignore

    for edge in edges_:
        edge.view = _EdgeViewer() # type: ignore

    sug = SugiyamaLayout(graph.C[0])
    graph = graph.C[0]
    roots = list(filter(lambda x: len(x.e_in()) == 0, graph.sV))

    sug.init_all(roots=roots, optimize=True)

    sug.yspace = VertexViewer.HEIGHT
    sug.xspace = minw
    sug.route_edge = route_with_lines # type: ignore

    sug.draw()

    return sug


def draw_ascii(graph: nx.DiGraph) -> str:
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
            node_str = str(node)[:max_width-2]  # Truncate if too long
            padding = max_width - len(node_str)
            left_pad = padding // 2
            right_pad = padding - left_pad
            node_boxes.append(f"|{' ' * left_pad}{node_str}{' ' * right_pad}|")

        return separator + f"\n{separator}".join(node_boxes) + f"\n{separator}"

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
        xlist.extend((
            vertex.view.xy[0] - vertex.view.w / 2.0,
            vertex.view.xy[0] + vertex.view.w / 2.0,
        ))
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

    # NOTE: first draw edges so that node boxes could overwrite them
    for edge in sug.g.sE:
        if len(edge.view.pts) <= 1:
            msg = "Not enough points to draw an edge"
            raise ValueError(msg)
        for index in range(1, len(edge.view.pts)):
            start = edge.view.pts[index - 1]
            end = edge.view.pts[index]

            start_x = round(start[0] - minx)
            start_y = round(start[1] - miny)
            end_x = round(end[0] - minx)
            end_y = round(end[1] - miny)

            if start_x < 0 or start_y < 0 or end_x < 0 or end_y < 0:
                msg = (
                    "Invalid edge coordinates: "
                    f"start_x={start_x}, "
                    f"start_y={start_y}, "
                    f"end_x={end_x}, "
                    f"end_y={end_y}"
                )
                raise ValueError(msg)

            canvas.line(start_x, start_y, end_x, end_y, "*")

    for vertex in sug.g.sV:
        # NOTE: moving boxes w/2 to the left
        x = vertex.view.xy[0] - vertex.view.w / 2.0
        y = vertex.view.xy[1]

        canvas.box(
            round(x - minx),
            round(y - miny),
            vertex.view.w,
            vertex.view.h,
        )

        canvas.text(round(x - minx) + 1, round(y - miny) + 1, vertex.data)

    return canvas.draw()


def draw_mermaid(  # noqa: PLR0912, PLR0915
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
        'graph', 'subgraph', 'direction', 'click', 'class', 'classDef',
        'style', 'linkStyle', 'fill', 'stroke', 'color', 'end', 'start',
        'TD', 'TB', 'BT', 'RL', 'LR', 'flowchart'
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
    nodes_list = list(graph.nodes())
    if nodes_list:
        # Find nodes with no predecessors (potential start nodes)
        start_candidates = [n for n in nodes_list if graph.in_degree(n) == 0]
        if start_candidates:
            first_node = start_candidates[0]

        # Find nodes with no successors (potential end nodes)
        end_candidates = [n for n in nodes_list if graph.out_degree(n) == 0]
        if end_candidates:
            last_node = end_candidates[0]

    for node in graph.nodes():
        node_id = _escape_node_label(str(node))
        node_label = str(node).replace('"', '\\"')  # Escape quotes

        # Handle special formatting for first/last nodes
        if node == first_node:
            mermaid += f"    {node_id}([{node_label}]):::first;\n"
        elif node == last_node:
            mermaid += f"    {node_id}([{node_label}]):::last;\n"
        else:
            mermaid += f"    {node_id}[{node_label}];\n"

    # Add edges
    for u, v in graph.edges():
        u_id = _escape_node_label(str(u))
        v_id = _escape_node_label(str(v))

        # Check if edge has data/label
        edge_data = graph.get_edge_data(u, v)
        if edge_data and edge_data.get('label'):
            label = str(edge_data['label'])
            # Wrap long labels
            if len(label.split()) > wrap_label_n_words:
                words = label.split()
                wrapped_label = "<br>".join(
                    " ".join(words[i:i + wrap_label_n_words])
                    for i in range(0, len(words), wrap_label_n_words)
                )
                mermaid += f"    {u_id} -- {wrapped_label} --> {v_id};\n"
            else:
                mermaid += f"    {u_id} -- {label} --> {v_id};\n"
        else:
            mermaid += f"    {u_id} --> {v_id};\n"

    # Add custom styles
    if with_styles:
        mermaid += "    classDef first fill:#e1f5fe,stroke:#01579b,stroke-width:2px;\n"
        mermaid += "    classDef last fill:#fff3e0,stroke:#e65100,stroke-width:2px;\n"
        mermaid += "    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;\n"

        # Add custom node colors if provided
        if node_colors:
            for node, color in node_colors.items():
                node_id = _escape_node_label(str(node))
                mermaid += f"    classDef {node_id}_style fill:{color};\n"
                mermaid += f"    class {node_id} {node_id}_style;\n"

    return mermaid


def draw_png(graph: nx.DiGraph, output_path: Optional[str] = None,
             node_labels: Optional[dict[Any, str]] = None,
             node_colors: Optional[dict[Any, str]] = None) -> Optional[bytes]:
    """Draw a NetworkX DiGraph as PNG using pygraphviz.

    Args:
        graph: NetworkX directed graph to draw
        output_path: Path to save PNG file (optional)
        node_labels: Custom labels for nodes (optional)
        node_colors: Custom colors for nodes (optional)

    Returns:
        PNG bytes if output_path is None, otherwise None
    """
    try:
        import pygraphviz as pgv  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install pygraphviz to draw PNG graphs: `pip install pygraphviz`."
        raise ImportError(msg) from exc

    if not graph.nodes():
        # Create a simple "empty graph" visualization
        viz = pgv.AGraph(directed=True)
        viz.add_node("Empty", label="Empty Graph", style="filled", fillcolor="lightgray")
        try:
            return viz.draw(output_path, format="png", prog="dot")
        finally:
            viz.close()

    # Create pygraphviz graph
    viz = pgv.AGraph(directed=True, nodesep=0.9, ranksep=1.0)

    # Add nodes
    for node in graph.nodes():
        label = node_labels.get(node, str(node)) if node_labels else str(node)
        color = node_colors.get(node, "lightblue") if node_colors else "lightblue"

        viz.add_node(
            str(node),
            label=f"<<B>{label}</B>>",
            style="filled",
            fillcolor=color,
            fontsize=12,
            fontname="arial",
        )

    # Add edges
    for u, v in graph.edges():
        viz.add_edge(str(u), str(v), fontsize=10, fontname="arial")

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
        import requests  # noqa: PLC0415
    except ImportError as e:
        msg = (
            "Install the `requests` module to use the Mermaid.INK API: "
            "`pip install requests`."
        )
        raise ImportError(msg) from e

    # Use Mermaid API to render the image
    mermaid_syntax_encoded = base64.b64encode(mermaid_syntax.encode("utf8")).decode(
        "ascii"
    )

    # Check if the background color is a hexadecimal color code using regex
    if background_color is not None:
        hex_color_pattern = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")
        if not hex_color_pattern.match(background_color):
            background_color = f"!{background_color}"

    image_url = (
        f"https://mermaid.ink/img/{mermaid_syntax_encoded}"
        f"?type={file_type}&bgColor={background_color}"
    )

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
        "Failed to reach https://mermaid.ink/ API while trying to render "
        f"your graph after {max_retries} retries. "
    ) + error_msg_suffix
    raise ValueError(msg)

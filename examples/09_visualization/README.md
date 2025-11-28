# Workflow Visualization

This directory contains examples demonstrating how to visualize Graflow workflows and graphs using various methods and formats.

## Overview

Visualization is crucial for:
- **Understanding** complex workflow structures
- **Debugging** workflow logic and dependencies
- **Documenting** workflows for team collaboration
- **Presenting** workflow designs in reports and slides
- **Analyzing** workflow complexity and optimization opportunities

## Examples

### 1. workflow_visualization.py
**Difficulty**: Intermediate
**Time**: 20 minutes

Demonstrates how to visualize actual Graflow workflows extracted from TaskGraph objects.

**Key Concepts**:
- Extracting NetworkX graphs from workflows
- ASCII visualization for terminal output
- Mermaid diagram generation for documentation
- PNG generation for presentations
- Workflow structure analysis

**What You'll Learn**:
- How to extract and visualize workflow graphs
- Different visualization methods and their use cases
- Customizing visualizations with colors and labels
- Analyzing workflow structure programmatically

**Run**:
```bash
python examples/08_visualization/workflow_visualization.py
```

**Expected Output**:
```
=== EXAMPLE 1: Simple Sequential Pipeline ===

📊 Workflow Structure:
=== Workflow 'simple_pipeline' Information ===
  +-----------+
  | load_data |
  +-----------+
        *
        *
        *
+---------------+
| validate_data |
+---------------+
...

🌊 Mermaid Diagram:
graph TD
    load_data --> validate_data
    validate_data --> transform_data
    ...
```

**Real-World Applications**:
- Workflow documentation generation
- CI/CD pipeline visualization
- Team onboarding materials
- Architecture diagrams

---

### 2. graph_utilities.py
**Difficulty**: Intermediate
**Time**: 25 minutes

Explores low-level graph visualization utilities for working directly with NetworkX DiGraph objects.

**Key Concepts**:
- Creating custom NetworkX graphs
- Multiple visualization formats (ASCII, Mermaid, PNG)
- Custom node labels and colors
- Graph analysis utilities
- Handling edge cases

**What You'll Learn**:
- Direct use of graph visualization utilities
- Creating custom visualizations for documentation
- Color coding and labeling strategies
- Graph structure analysis
- Working with complex DAGs

**Run**:
```bash
python examples/08_visualization/graph_utilities.py
```

**Expected Output**:
```
=== EXAMPLE 1: Basic Graph Visualization ===

📊 Graph Structure:
Nodes: 4
Edges: 3
...

🎨 ASCII Visualization:
  start
    *
    *
    *
  task_a
    *
...

🌊 Mermaid Diagram:
graph TD
    start --> task_a
    ...
```

**Real-World Applications**:
- Data pipeline documentation
- Dependency graph visualization
- System architecture diagrams
- Process flow documentation

---

## Learning Path

**Recommended Order**:
1. Start with `workflow_visualization.py` to understand Graflow-specific visualization
2. Move to `graph_utilities.py` for low-level control and custom graphs

**Prerequisites**:
- Complete examples from 01-02 (basics and workflows)
- Basic understanding of graph theory (helpful but not required)

**Total Time**: ~45 minutes

---

## Visualization Methods

### 1. ASCII Visualization

**Purpose**: Quick visualization in terminal or logs

**Pros**:
- Works in any terminal
- No external dependencies (except grandalf)
- Fast generation
- Great for debugging

**Cons**:
- Limited formatting options
- Not suitable for complex graphs
- Requires grandalf package

**Example**:
```python
from graflow.utils.graph import draw_ascii

ascii_output = draw_ascii(graph)
print(ascii_output)
```

**Use When**:
- Debugging workflows in terminal
- Adding to log files
- Quick structure checks
- CI/CD output

---

### 2. Mermaid Diagrams

**Purpose**: Documentation and markdown-friendly diagrams

**Pros**:
- Works in markdown/GitHub
- Highly customizable
- No dependencies
- Good for documentation

**Cons**:
- Requires mermaid-compatible viewer
- Manual rendering needed for some platforms

**Example**:
```python
from graflow.utils.graph import draw_mermaid

mermaid_code = draw_mermaid(
    graph,
    title="My Workflow",
    node_colors={"start": "#90EE90"}
)
print(mermaid_code)
```

**Use When**:
- Creating documentation
- README files
- Wiki pages
- GitHub/GitLab markdown

**Mermaid in Markdown**:
````markdown
```mermaid
graph TD
    start --> process
    process --> end
```
````

---

### 3. PNG Generation

**Purpose**: High-quality images for presentations and reports

**Pros**:
- Professional appearance
- Full customization (colors, labels, layout)
- Works everywhere
- High resolution

**Cons**:
- Requires pygraphviz installation
- Slower than other methods
- Binary file (not text)

**Example**:
```python
from graflow.utils.graph import draw_png

png_bytes = draw_png(
    graph,
    output_path="/tmp/workflow.png",
    node_colors={"start": "lightgreen"},
    node_labels={"start": "Begin Process"}
)
```

**Use When**:
- Presentations
- Technical reports
- Architecture documentation
- Marketing materials

---

### 4. Mermaid PNG (API)

**Purpose**: PNG generation without local dependencies

**Pros**:
- No local dependencies except requests
- Works anywhere with internet
- Consistent rendering

**Cons**:
- Requires internet connection
- Depends on external API
- Rate limits may apply

**Example**:
```python
from graflow.utils.graph import draw_mermaid_png

png_bytes = draw_mermaid_png(
    graph,
    output_path="/tmp/workflow.png",
    background_color="white"
)
```

**Use When**:
- Cannot install pygraphviz
- Need consistent rendering
- Cloud/serverless environments

---

## Installation

### Core Functionality

No extra dependencies needed for basic workflow visualization using `show_info()` and dependency visualization.

### Optional Dependencies

For full visualization capabilities:

```bash
# ASCII visualization
pip install grandalf

# PNG generation (local)
pip install pygraphviz

# Mermaid PNG via API
pip install requests
```

### Platform-Specific Notes

**macOS (pygraphviz)**:
```bash
brew install graphviz
pip install pygraphviz
```

**Ubuntu/Debian (pygraphviz)**:
```bash
sudo apt-get install graphviz graphviz-dev
pip install pygraphviz
```

**Windows (pygraphviz)**:
```bash
# Download Graphviz from https://graphviz.org/download/
# Add to PATH
pip install pygraphviz
```

---

## Customization Guide

### Color Schemes

**By Task Type**:
```python
colors = {
    "load_*": "lightblue",      # Data loading
    "validate_*": "lightyellow", # Validation
    "process_*": "lightgreen",   # Processing
    "save_*": "lightcoral"       # Output
}
```

**By Workflow Stage**:
```python
stage_colors = {
    # Ingestion
    "extract": "#E6F3FF",

    # Transformation
    "transform": "#FFF4E6",

    # Output
    "load": "#F3E6FF"
}
```

**By Status**:
```python
status_colors = {
    "completed": "#90EE90",  # Green
    "running": "#FFE4B5",    # Orange
    "pending": "#D3D3D3",    # Gray
    "failed": "#FFB6C1"      # Red
}
```

### Custom Labels

**Readable Names**:
```python
labels = {
    "ld_usr_data": "Load User\nDatabase",
    "val_schema": "Validate\nSchema",
    "xfm_enrich": "Transform &\nEnrich Data"
}

draw_png(graph, node_labels=labels)
```

**With Icons** (PNG only):
```python
labels = {
    "load": "📥 Load Data",
    "process": "⚙️  Process",
    "save": "💾 Save Results"
}
```

---

## Common Patterns

### 1. Documentation Generation

Generate workflow documentation automatically:

```python
from graflow.core.workflow import workflow
from graflow.utils.graph import draw_mermaid

with workflow("my_workflow") as ctx:
    # Define workflow...

    # Generate documentation
    mermaid_code = draw_mermaid(
        ctx.graph.nx_graph(),
        title="My Workflow Architecture"
    )

    # Save to file
    with open("docs/workflow.md", "w") as f:
        f.write("# Workflow Documentation\n\n")
        f.write("```mermaid\n")
        f.write(mermaid_code)
        f.write("\n```\n")
```

### 2. Debugging Complex Workflows

Quick terminal visualization:

```python
from graflow.utils.graph import draw_ascii, show_graph_info

# Quick structure check
show_graph_info(ctx.graph.nx_graph())

# ASCII view in terminal
print(draw_ascii(ctx.graph.nx_graph()))
```

### 3. Presentation Materials

Generate high-quality PNGs:

```python
# Color code by layer
layer_colors = {
    "input": "lightblue",
    "processing": "lightyellow",
    "output": "lightgreen"
}

# Generate PNG for presentation
draw_png(
    graph,
    output_path="presentation/workflow_diagram.png",
    node_colors=layer_colors
)
```

### 4. CI/CD Integration

Automatically generate workflow diagrams in CI/CD:

```bash
# In CI/CD pipeline
python -c "
from my_workflow import create_workflow
from graflow.utils.graph import draw_mermaid

with create_workflow() as ctx:
    mermaid = draw_mermaid(ctx.graph.nx_graph())
    with open('docs/workflow.md', 'w') as f:
        f.write('```mermaid\n' + mermaid + '\n```')
"
```

---

## Best Practices

### ✅ DO:

- Use ASCII for quick debugging in terminal
- Use Mermaid for documentation in markdown
- Use PNG for presentations and reports
- Color code nodes by type/stage/status
- Keep node labels concise
- Generate visualizations automatically in CI/CD
- Include visualizations in PR descriptions

### ❌ DON'T:

- Use PNG for version control (use Mermaid instead)
- Create overly complex visualizations (split into multiple diagrams)
- Use too many colors (stick to 3-5 color scheme)
- Hard-code file paths (use environment variables)
- Generate visualizations in production code

---

## Troubleshooting

### ASCII Drawing Errors

**Problem**: `ModuleNotFoundError: No module named 'grandalf'`

**Solution**:
```bash
pip install grandalf
```

### PNG Generation Errors

**Problem**: `ImportError: cannot import name 'AGraph' from 'pygraphviz'`

**Solution**:
```bash
# macOS
brew install graphviz
pip install --force-reinstall pygraphviz

# Ubuntu
sudo apt-get install graphviz graphviz-dev
pip install --force-reinstall pygraphviz
```

### Mermaid API Errors

**Problem**: `requests.exceptions.ConnectionError`

**Solution**:
- Check internet connection
- Install requests: `pip install requests`
- Use local PNG generation instead

### Graph Too Large

**Problem**: Visualization is cluttered or unreadable

**Solution**:
- Split into multiple diagrams (by stage/component)
- Increase PNG size/resolution
- Use abbreviated node names
- Focus on critical path only

---

## Advanced Topics

### Custom Visualization Functions

Create reusable visualization helpers:

```python
def visualize_workflow(ctx, output_dir="docs/workflows"):
    """Generate all visualization formats for a workflow."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    graph = ctx.graph.nx_graph()
    name = ctx.name

    # Mermaid for documentation
    mermaid = draw_mermaid(graph, title=name)
    with open(f"{output_dir}/{name}.md", "w") as f:
        f.write(f"# {name}\n\n```mermaid\n{mermaid}\n```\n")

    # PNG for presentations
    draw_png(graph, output_path=f"{output_dir}/{name}.png")

    # ASCII for logs
    with open(f"{output_dir}/{name}.txt", "w") as f:
        f.write(draw_ascii(graph))
```

### Animated Visualizations

Track workflow execution progress:

```python
# Generate snapshots during execution
snapshots = []

@task(inject_context=True)
def my_task(context):
    # Capture current state
    graph = context.execution_context.graph.nx_graph()
    snapshot = draw_mermaid(graph)
    snapshots.append(snapshot)

    # ... task logic
```

### Interactive Visualizations

Use with Jupyter notebooks:

```python
from IPython.display import Image, display
from graflow.utils.graph import draw_png

# In Jupyter
png_bytes = draw_png(graph)
display(Image(png_bytes))
```

---

## Next Steps

After mastering visualization, explore:

1. **Custom Handlers** (`../04_execution/`)
   - Visualize handler selection
   - Task execution flow

2. **Distributed Workflows** (`../05_distributed/`)
   - Visualize task distribution
   - Worker allocation

3. **Real-World Examples** (`../07_real_world/`)
   - Document production workflows
   - Create architecture diagrams

---

## Additional Resources

- **Mermaid Documentation**: https://mermaid.js.org/
- **NetworkX Documentation**: https://networkx.org/
- **Graphviz Documentation**: https://graphviz.org/
- **Pygraphviz**: https://pygraphviz.github.io/

---

**Ready to visualize?** Start with `workflow_visualization.py` to see how easy it is!

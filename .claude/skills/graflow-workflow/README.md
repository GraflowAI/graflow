# Graflow Workflow Builder Skill

Claude Code skill for building executable task graph workflows using Graflow.

## Overview

This skill guides users through a structured 3-phase process to create Python workflow pipelines:

1. **Plan** - Requirements gathering, design document creation, and iterative feedback
2. **Implement** - Code generation based on approved design
3. **Review** - Validation and documentation

## Trigger Keywords

This skill activates when users mention:
- "workflow", "pipeline", "task graph"
- "Graflow"
- Building automated data/AI pipelines

## Workflow Process

### Phase 1: Plan (Design with Feedback Loop)

```
Requirements Gathering
        ↓
Create Design Document (workflow_design.md)
        ↓
Present Design to User
        ↓
    ┌───────────────────┐
    │  User Feedback?   │
    └───────────────────┘
      ↓ Yes         ↓ No
  Update Design   Confirm Approval
      ↓                 ↓
   Re-present     → Phase 2
```

Key outputs:
- `workflow_design.md` with task definitions, graph structure, and data flow

### Phase 2: Implement

Based on the approved design:
- Create workflow file with `@task` decorators
- Use `>>` (sequential) and `|` (parallel) operators
- Set up channel communication
- Add type hints and docstrings

### Phase 3: Review

- Validate implementation against design
- Check for common issues (context injection, parameter priority, etc.)
- Create `README.md` with usage instructions and workflow documentation

## File Structure

```
.claude/skills/graflow-workflow/
├── SKILL.md                          # Main skill definition
├── README.md                         # This file
└── references/
    ├── workflow-patterns.md          # Core Graflow patterns
    └── advanced-patterns.md          # Dynamic tasks, HITL, distributed execution
```

## References

The `references/` directory contains detailed documentation:

- **workflow-patterns.md**: Task definitions, composition operators, channels, LLM integration
- **advanced-patterns.md**: Dynamic task generation, checkpoints, HITL, prompt management, distributed execution

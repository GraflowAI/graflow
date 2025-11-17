import { FormEvent, useMemo, useState } from "react";
import {
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Link,
  Stack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography
} from "@mui/material";
import { supportedLayouts, supportedWorkflows } from "../api/client";
import type { LayoutOption, WorkflowOption } from "../api/types";
import originalWorkflowImg from "../assets/original_wf.png";
import dynamicWorkflowImg from "../assets/dynamic_wf.svg";

export interface QueryFormPayload {
  queries: string[];
  layout: LayoutOption;
  outputDir?: string;
  workflow: WorkflowOption;
}

export interface QueryFormProps {
  onSubmit: (payload: QueryFormPayload) => void;
  disabled?: boolean;
}

const DEFAULT_QUERIES = ["technology", "climate change", "artificial intelligence"];

const QueryForm = ({ onSubmit, disabled = false }: QueryFormProps) => {
  const [layout, setLayout] = useState<LayoutOption>("two-column");
  const [rawQueries, setRawQueries] = useState<string>(DEFAULT_QUERIES.join("\n"));
  const [outputDir, setOutputDir] = useState<string>("");
  const [workflow, setWorkflow] = useState<WorkflowOption>("original");
  const [previewWorkflow, setPreviewWorkflow] = useState<WorkflowOption | null>(null);

  const workflowImages: Record<WorkflowOption, string> = {
    original: originalWorkflowImg,
    dynamic: dynamicWorkflowImg,
    agent: dynamicWorkflowImg  // Using dynamic workflow image as placeholder for agent workflow
  };

  const workflowTitles: Record<WorkflowOption, string> = {
    original: "Original Workflow",
    dynamic: "Dynamic Workflow",
    agent: "Agent Workflow"
  };

  const workflowDescriptions: Record<WorkflowOption, string> = {
    original: "Simple LLM tasks with basic critique loop",
    dynamic: "Complex parallel tasks with quality gates and runtime task generation",
    agent: "LLM Agents with autonomous tool calling and ReAct pattern"
  };

  const queries = useMemo(
    () =>
      rawQueries
        .split("\n")
        .map((query) => query.trim())
        .filter(Boolean),
    [rawQueries]
  );

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!queries.length) {
      return;
    }
    onSubmit({
      queries,
      layout,
      workflow,
      outputDir: outputDir || undefined
    });
  };

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <Stack spacing={3}>
        <Stack spacing={1}>
          <Typography variant="subtitle2" component="label" htmlFor="queries-input">
            Queries
          </Typography>
          <TextField
            id="queries-input"
            value={rawQueries}
            onChange={(event) => setRawQueries(event.target.value)}
            placeholder="Enter one query per line"
            multiline
            rows={4}
            disabled={disabled}
          />
          <Stack direction="row" spacing={1} flexWrap="wrap">
            {queries.map((query) => (
              <Chip key={query} label={query} size="small" color="primary" variant="outlined" />
            ))}
          </Stack>
        </Stack>

        <Stack spacing={1}>
          <Typography variant="subtitle2" component="span">
            Layout
          </Typography>
          <ToggleButtonGroup
            value={layout}
            exclusive
            onChange={(_, value: LayoutOption | null) => {
              if (value) {
                setLayout(value);
              }
            }}
            size="small"
          >
            {supportedLayouts.map((item) => (
              <ToggleButton key={item} value={item}>
                {item}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Stack>

        <Stack spacing={1}>
          <Typography variant="subtitle2" component="span">
            Workflow
          </Typography>
          <ToggleButtonGroup
            value={workflow}
            exclusive
            onChange={(_, value: WorkflowOption | null) => {
              if (value) {
                setWorkflow(value);
              }
            }}
            size="small"
          >
            {supportedWorkflows.map((item) => (
              <ToggleButton key={item} value={item}>
                {item}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
          <Typography variant="body2" color="text.secondary">
            <strong>Original:</strong> {workflowDescriptions.original}<br />
            <strong>Dynamic:</strong> {workflowDescriptions.dynamic}<br />
            <strong>Agent:</strong> {workflowDescriptions.agent}
            <br /><br />
            <Link
              component="button"
              type="button"
              onClick={() => setPreviewWorkflow("original")}
              underline="hover"
            >
              View original
            </Link>
            {" / "}
            <Link
              component="button"
              type="button"
              onClick={() => {
                window.open(dynamicWorkflowImg, "_blank", "noopener,noreferrer");
              }}
              underline="hover"
            >
              View dynamic
            </Link>
            {" / "}
            <Link
              component="button"
              type="button"
              onClick={() => setPreviewWorkflow("agent")}
              underline="hover"
            >
              View agent
            </Link>
          </Typography>
        </Stack>

        <TextField
          label="Output directory override (optional)"
          value={outputDir}
          onChange={(event) => setOutputDir(event.target.value)}
          disabled={disabled}
        />

        <Box>
          <Button type="submit" variant="contained" disabled={disabled || !queries.length}>
            Generate newspaper
          </Button>
        </Box>
      </Stack>

      <Dialog
        open={previewWorkflow !== null}
        onClose={() => setPreviewWorkflow(null)}
        maxWidth="md"
        fullWidth
      >
        {previewWorkflow && (
          <>
            <DialogTitle>{workflowTitles[previewWorkflow]}</DialogTitle>
            <DialogContent>
              <Stack spacing={2}>
                <Typography variant="body2" color="text.secondary">
                  {workflowDescriptions[previewWorkflow]}
                </Typography>
                {previewWorkflow === "agent" ? (
                  <Box>
                    <Typography variant="h6" gutterBottom>Agent Workflow Architecture</Typography>
                    <Typography variant="body2" component="div">
                      <strong>Research Agent:</strong>
                      <ul>
                        <li>Autonomous web search with Tavily API</li>
                        <li>Extract key facts from multiple sources</li>
                        <li>Refine queries based on findings (ReAct pattern)</li>
                      </ul>
                      <strong>Editorial Agent:</strong>
                      <ul>
                        <li>Check factual claims against sources</li>
                        <li>Assess readability with textstat metrics</li>
                        <li>Verify source credibility</li>
                        <li>Autonomously decide: APPROVE or REVISE</li>
                      </ul>
                      <strong>Flow:</strong> topic_intake → research_agent → curate → write → editorial_agent → design
                    </Typography>
                  </Box>
                ) : (
                  <Box sx={{ display: "flex", justifyContent: "center" }}>
                    <Box
                      component="img"
                      src={workflowImages[previewWorkflow]}
                      alt={`${previewWorkflow} workflow diagram`}
                      sx={{ width: "100%", maxHeight: 500, objectFit: "contain" }}
                    />
                  </Box>
                )}
              </Stack>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setPreviewWorkflow(null)}>Close</Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  );
};

export default QueryForm;

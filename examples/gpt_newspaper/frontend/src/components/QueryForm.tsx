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
    dynamic: dynamicWorkflowImg
  };

  const workflowTitles: Record<WorkflowOption, string> = {
    original: "Original Workflow",
    dynamic: "Dynamic Workflow"
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
            Choose between the original static flow or the new dynamic workflow with runtime branching. {" "}
            <Link
              component="button"
              type="button"
              onClick={() => setPreviewWorkflow("original")}
              underline="hover"
            >
              View original flow
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
              View dynamic flow
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
        {previewWorkflow && previewWorkflow === "original" && (
          <>
            <DialogTitle>{workflowTitles[previewWorkflow]}</DialogTitle>
            <DialogContent>
              <Box sx={{ display: "flex", justifyContent: "center" }}>
                <Box
                  component="img"
                  src={workflowImages[previewWorkflow]}
                  alt={`${previewWorkflow} workflow diagram`}
                  sx={{ width: "100%", maxHeight: 500, objectFit: "contain" }}
                />
              </Box>
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

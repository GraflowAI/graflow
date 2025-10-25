import { FormEvent, useMemo, useState } from "react";
import {
  Box,
  Button,
  Chip,
  Stack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography
} from "@mui/material";
import { supportedLayouts } from "../api/client";
import type { LayoutOption } from "../api/types";

export interface QueryFormPayload {
  queries: string[];
  layout: LayoutOption;
  outputDir?: string;
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
    </Box>
  );
};

export default QueryForm;

/**
 * GPT Newspaper - Main Application Component
 * Graflow Edition with Material-UI
 */

import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  Typography,
  Paper,
  Button,
  Link,
  Alert
} from '@mui/material';
import NewspaperIcon from '@mui/icons-material/Newspaper';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

import { TopicInput, LayoutSelector, LoadingSpinner, ErrorMessage } from './components';
import { apiClient } from './api';
import { theme } from './theme';
import type { LayoutOption } from './types';

interface Topic {
  id: number;
  value: string;
}

const MAX_TOPICS = 10;

function App() {
  const [topics, setTopics] = useState<Topic[]>([{ id: 1, value: '' }]);
  const [selectedLayout, setSelectedLayout] = useState<LayoutOption>('layout_1.html');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'ok' | 'warning' | 'error'>('ok');

  // Check API health on mount
  useEffect(() => {
    checkAPIHealth();
  }, []);

  const checkAPIHealth = async () => {
    try {
      const health = await apiClient.checkHealth();
      if (health.status === 'warning') {
        setApiStatus('warning');
        console.warn('API Warning:', health.message);
      } else {
        setApiStatus('ok');
      }
    } catch (err) {
      setApiStatus('error');
      console.error('Failed to connect to API:', err);
    }
  };

  const handleTopicChange = (id: number, value: string) => {
    setTopics((prev) => prev.map((t) => (t.id === id ? { ...t, value } : t)));
  };

  const handleAddTopic = () => {
    if (topics.length < MAX_TOPICS) {
      const newId = Math.max(...topics.map((t) => t.id)) + 1;
      setTopics((prev) => [...prev, { id: newId, value: '' }]);
    }
  };

  const handleRemoveTopic = (id: number) => {
    if (topics.length > 1) {
      setTopics((prev) => prev.filter((t) => t.id !== id));
    }
  };

  const handleProduceNewspaper = async () => {
    // Collect non-empty topics
    const filledTopics = topics
      .filter((t) => t.value.trim())
      .map((t) => t.value.trim());

    if (filledTopics.length === 0) {
      setError('Please fill in at least one topic.');
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      const response = await apiClient.generateNewspaper({
        topics: filledTopics,
        layout: selectedLayout,
        max_workers: null,
      });

      // Redirect to the generated newspaper
      window.location.href = response.path;
    } catch (err) {
      setIsLoading(false);
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred');
      }
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="md" sx={{ py: 4 }}>
        {/* Header */}
        <Paper elevation={2} sx={{ p: 4, mb: 4, textAlign: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
            <NewspaperIcon sx={{ fontSize: 48, mr: 2, color: 'primary.main' }} />
            <Typography variant="h1" component="h1" color="primary">
              GPT Newspaper
            </Typography>
          </Box>

          <Typography variant="h6" color="text.secondary" gutterBottom>
            Powered by Graflow
          </Typography>

          <Typography variant="body1" color="text.secondary" sx={{ mt: 2 }}>
            Search powered by{' '}
            <Link href="https://tavily.com" target="_blank" rel="noopener noreferrer">
              Tavily
            </Link>
            {' | '}
            Workflow powered by{' '}
            <Link
              href="https://github.com/yourusername/graflow"
              target="_blank"
              rel="noopener noreferrer"
            >
              Graflow
            </Link>
          </Typography>

          <Typography variant="h6" sx={{ mt: 3, fontWeight: 500 }}>
            Type topics of interest and get your personalized newspaper
          </Typography>

          {apiStatus === 'error' && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              ⚠️ Warning: Unable to connect to backend API. Make sure the server is running on
              port 8000.
            </Alert>
          )}
        </Paper>

        {/* Topic Selection */}
        <Paper elevation={2} sx={{ p: 4, mb: 3 }}>
          <Typography variant="h5" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
            Select Topics
          </Typography>

          <Box component="form" onSubmit={(e) => e.preventDefault()}>
            {topics.map((topic, index) => (
              <TopicInput
                key={topic.id}
                id={topic.id}
                value={topic.value}
                onChange={handleTopicChange}
                onAdd={handleAddTopic}
                onRemove={handleRemoveTopic}
                isFirst={index === 0}
                isLast={index === topics.length - 1}
                canAddMore={topics.length < MAX_TOPICS}
                disabled={isLoading}
              />
            ))}
          </Box>

          <Box sx={{ mt: 4, mb: 3 }}>
            <LayoutSelector
              selected={selectedLayout}
              onChange={setSelectedLayout}
              disabled={isLoading}
            />
          </Box>

          <Button
            variant="contained"
            size="large"
            fullWidth
            onClick={handleProduceNewspaper}
            disabled={isLoading}
            sx={{ mt: 3, py: 1.5 }}
            aria-label="Produce newspaper"
          >
            {isLoading ? 'Generating...' : 'Produce Newspaper'}
          </Button>
        </Paper>

        {/* Loading State */}
        {isLoading && <LoadingSpinner loading={isLoading} />}

        {/* Error State */}
        {error && (
          <ErrorMessage
            message={error}
            title="Error"
            onDismiss={() => setError(null)}
            open={Boolean(error)}
          />
        )}
      </Container>
    </ThemeProvider>
  );
}

export default App;

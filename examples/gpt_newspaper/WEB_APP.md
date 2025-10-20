# GPT Newspaper Web Application

FastAPI + React/TypeScript web application for the GPT Newspaper generator.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Application                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Frontend (React/TypeScript)         Backend (FastAPI)      │
│  ├── Vite Dev Server (port 5173)    ├── API Server (8000)  │
│  ├── React Components                ├── Pydantic Models    │
│  ├── TypeScript API Client           ├── CORS Middleware    │
│  └── Modern UI/UX                    └── Static Files       │
│                                                              │
│  ┌─────────────┐                    ┌─────────────┐        │
│  │   User UI   │ ──── HTTP ───────▶ │  REST API   │        │
│  └─────────────┘                    └─────────────┘        │
│                                              │               │
│                                              ▼               │
│                                    ┌──────────────────┐     │
│                                    │  Graflow Engine  │     │
│                                    │  - Search Agent  │     │
│                                    │  - Curator Agent │     │
│                                    │  - Writer Agent  │     │
│                                    │  - Critique      │     │
│                                    │  - Designer      │     │
│                                    └──────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Setup

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   cd examples/gpt_newspaper/backend
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export TAVILY_API_KEY=<your_tavily_key>
   export OPENAI_API_KEY=<your_openai_key>
   ```

3. **Start the FastAPI server:**
   ```bash
   # From examples/gpt_newspaper/backend directory
   uvicorn api:app --reload --port 8000

   # Or directly with Python
   python api.py
   ```

   The backend will be available at:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - OpenAPI schema: http://localhost:8000/openapi.json

### Frontend Setup

1. **Install Node.js dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Configure API URL (optional):**
   ```bash
   # Create .env file from template
   cp .env.example .env

   # Edit .env if needed (defaults to http://localhost:8000)
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

   The frontend will be available at: http://localhost:5173

## Usage

1. **Start both servers:**
   ```bash
   # Terminal 1: Backend
   cd examples/gpt_newspaper/backend
   uvicorn api:app --reload

   # Terminal 2: Frontend
   cd examples/gpt_newspaper/frontend
   npm run dev
   ```

2. **Open browser:**
   - Go to http://localhost:5173
   - Enter topics of interest (1-10 topics)
   - Select a newspaper layout (1, 2, or 3)
   - Click "Produce Newspaper"
   - Wait for the generation process (shows progress)
   - Browser will redirect to the generated newspaper

## API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "GPT Newspaper API is running"
}
```

### `POST /api/generate`
Generate a newspaper from topics.

**Request Body:**
```json
{
  "topics": ["AI developments", "Climate change"],
  "layout": "layout_1.html",
  "max_workers": null
}
```

**Response:**
```json
{
  "path": "/outputs/run_1234567890/newspaper.html",
  "article_count": 2,
  "timestamp": 1234567890
}
```

### `GET /outputs/{path}`
Serve generated newspaper files.

## Frontend Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── TopicInput.tsx       # Individual topic input field
│   │   ├── LayoutSelector.tsx   # Layout selection UI
│   │   ├── LoadingSpinner.tsx   # Loading animation
│   │   └── ErrorMessage.tsx     # Error display
│   ├── api.ts                   # API client
│   ├── types.ts                 # TypeScript type definitions
│   ├── App.tsx                  # Main application component
│   ├── App.css                  # Application styles
│   └── main.tsx                 # Entry point
├── index.html                   # HTML template
├── package.json                 # NPM dependencies
├── tsconfig.json                # TypeScript configuration
└── vite.config.ts               # Vite configuration
```

## Production Build

### Build Frontend
```bash
cd frontend
npm run build
```

This creates an optimized production build in `frontend/dist/`.

### Serve Production Build
The FastAPI backend can serve the production build by mounting the `dist` directory:

```python
# Add to api.py
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
```

Then run:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Access the app at: http://localhost:8000

## Development Tips

### Hot Reload
Both backend and frontend support hot reload:
- Backend: `uvicorn api:app --reload`
- Frontend: `npm run dev`

### Type Safety
The frontend uses TypeScript for type safety:
- API types defined in `src/types.ts`
- Props interfaces in each component
- Strict TypeScript configuration

### CORS
The backend enables CORS for development:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, update `allow_origins` to specific domains.

## Troubleshooting

### Frontend can't connect to backend
- Ensure backend is running on port 8000
- Check `VITE_API_URL` in frontend `.env`
- Verify CORS settings in `api.py`

### API Key errors
- Set `TAVILY_API_KEY` environment variable
- Set `OPENAI_API_KEY` (or your LLM provider key)
- Check the health endpoint: http://localhost:8000

### Build errors
```bash
# Clear npm cache
cd frontend
rm -rf node_modules package-lock.json
npm install

# Clear Python cache
cd ..
find . -type d -name __pycache__ -exec rm -rf {} +
```

## Technologies Used

### Backend
- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation using Python type hints
- **Uvicorn**: ASGI server
- **Graflow**: Workflow engine for agent orchestration

### Frontend
- **React 19**: UI library
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool and dev server
- **Material-UI (MUI)**: Component library and design system
- **Axios**: Type-safe HTTP client
- **Storybook**: Component development and documentation
- **ESLint with jsx-a11y**: Accessibility linting

## Contributing

See the main Graflow repository for contribution guidelines.

## License

Same as Graflow main project.

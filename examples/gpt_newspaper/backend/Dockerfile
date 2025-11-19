# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

# Install system deps needed for uvicorn[standard] (uvloop) and building wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy graflow package metadata first for better caching.
COPY pyproject.toml README.md /workspace/
COPY graflow /workspace/graflow

# Copy example backend code and requirements.
COPY examples/gpt_newspaper/backend /workspace/examples/gpt_newspaper/backend

RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install -r examples/gpt_newspaper/backend/requirements.txt

WORKDIR /workspace/examples/gpt_newspaper/backend

EXPOSE 8000

# nest_asyncio: Can't patch loop of type <class 'uvloop.Loop'>
# https://github.com/NVIDIA-NeMo/Guardrails/issues/112#issuecomment-1744171237
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]

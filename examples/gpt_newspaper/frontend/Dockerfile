# syntax=docker/dockerfile:1.7

FROM node:20-slim AS runtime

ENV NODE_ENV=development \
    HOST=0.0.0.0 \
    PORT=5173

WORKDIR /workspace/examples/gpt_newspaper/frontend

# Install dependencies based on lockfile for reproducibility. Use
# --legacy-peer-deps because Storybook currently targets Vite <=6 while this
# project uses Vite 7 – the versions are known to work together despite the
# peer warning.
COPY examples/gpt_newspaper/frontend/package*.json ./
RUN npm ci --legacy-peer-deps

# Copy the remaining frontend source after dependencies restore.
COPY examples/gpt_newspaper/frontend ./

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]

# Build context is the repository root: the backend serves ../files as its
# static frontend, so both directories have to be inside the image.
FROM node:24-slim

# Prisma's query engine needs openssl; curl is used by the compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssl curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/backend

COPY backend/package.json backend/package-lock.json ./
RUN npm ci --fetch-retries 5 --fetch-timeout 120000

COPY backend/ ./
COPY files/ /workspace/files/

# `prisma generate` reads prisma.config.ts, which requires DATABASE_URL to be
# set. The real value is injected at runtime by compose.
ENV DATABASE_URL="postgresql://placeholder:placeholder@localhost:5432/placeholder"
RUN npx prisma generate

EXPOSE 4000

CMD ["sh", "-c", "npx prisma migrate deploy && npx prisma db seed && node server.js"]

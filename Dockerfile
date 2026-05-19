FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

# Non-root for k8s compatibility (matches HelmRelease runAsUser: 1000)
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app
USER 1000

ENTRYPOINT ["memory-mcp-web"]

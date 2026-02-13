# Use a modern Python base
FROM python:3.12-slim-bookworm

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Julia
ENV JULIA_VERSION=1.10.2
RUN curl -sSL "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION%.*}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
    | tar -xzC /opt --strip-components=0
ENV PATH="/opt/julia-${JULIA_VERSION}/bin:${PATH}"

# 3. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 4. Install Dependencies (Leverage caching)
COPY pyproject.toml uv.lock Project.toml Manifest.toml ./
RUN uv sync --frozen --no-dev && \
    julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# 5. Copy the source code (Logic is baked in, data/configs stay out)
COPY . .

# 6. NO DEFAULT EXECUTION
# This ensures that if you just 'run' the container, it doesn't start training.
# It waits for your job script to tell it what to do.
CMD ["/bin/bash"]
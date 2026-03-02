# Use a modern Python base
FROM python:3.13-slim-bookworm

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Julia
ENV JULIA_VERSION=1.10.2
RUN curl -sSL "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION%.*}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
    | tar -xzC /opt --strip-components=0
ENV PATH="/opt/julia-${JULIA_VERSION}/bin:${PATH}"

# 3. Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

# 4. Install Dependencies
# Copy only the files needed for installation first
# REMOVED Manifest.toml from this list because we want to generate a fresh one
COPY pyproject.toml uv.lock Project.toml ./

# FIX: Remove the local Manifest.toml (if it existed) and instantiate
RUN uv sync --frozen --no-dev && \
    rm -f Manifest.toml && \
    julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# 5. Copy the source code
# (With .dockerignore, this won't overwrite the fresh Manifest)
COPY . .

# FIX: Set permissions so you don't need 'chmod +x' manually
RUN chmod +x *.sh

# FIX: Tell Julia to ALWAYS use the project in the current directory
# This prevents the "Package HDF5 not found" error when running scripts
ENV JULIA_PROJECT=@.

# 6. NO DEFAULT EXECUTION
CMD ["/bin/bash"]
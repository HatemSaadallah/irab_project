FROM python:3.10-slim

WORKDIR /app

# System deps for torch + scientific Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the whole package (this Dockerfile is meant to be built from the repo root,
# not from app/). If building from app/ only, use the git+https install in requirements.
COPY pyproject.toml ./
COPY src/ ./src/
COPY app/requirements.txt ./app_requirements.txt

RUN pip install --no-cache-dir -r app_requirements.txt
RUN pip install --no-cache-dir -e .

COPY app/ ./app/

# HF Spaces cache dirs need to be writable
ENV HF_HOME=/tmp/huggingface
ENV STREAMLIT_HOME=/tmp/.streamlit
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

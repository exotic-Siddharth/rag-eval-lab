FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/hf

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY raglab ./raglab
RUN pip install --no-cache-dir ".[ml,claude]"
COPY configs ./configs

# Corpus and models download on first start; mount /app/.cache and /app/data to keep them.
RUN python -m raglab download
EXPOSE 8000
CMD ["python", "-m", "raglab", "serve", "--host", "0.0.0.0", "--port", "8000"]

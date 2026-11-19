# Image for the live demo on Hugging Face Spaces (free CPU: 2 vCPU, 16 GB).
# The deploy workflow copies raglab/, configs/, pyproject.toml and LICENSE next to this file.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    OMP_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY raglab ./raglab
RUN pip install --no-cache-dir ".[ml]"
COPY configs ./configs
RUN chown -R user:user /home/user
USER user

# Download SQuAD 2.0 dev and the three models, embed every passage and run one question through each
# reader now, so visitors never wait for a download.
RUN python -m raglab demo --config configs/default.yaml --warm-only

EXPOSE 7860
CMD ["python", "-m", "raglab", "demo", "--config", "configs/default.yaml", "--host", "0.0.0.0", "--port", "7860"]

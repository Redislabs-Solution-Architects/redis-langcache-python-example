# =========================
# Dockerfile
# =========================
# Imagem base enxuta
FROM python:3.12-slim AS runtime

# Evita prompts interativos e acelera pip
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Dependências do sistema (ca-certificates + curl p/ healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Cria usuário não-root
RUN useradd -m -u 10001 appuser
WORKDIR /app

# Copia apenas o requirements primeiro (melhor cache de camadas)
COPY requirements.txt /app/requirements.txt

# Instala dependências Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copia o restante do projeto (NÃO copie .env; use --env-file no run/compose)
COPY . /app

# Ajusta permissões
RUN chown -R appuser:appuser /app
USER appuser

# Exponha a porta do Gradio
EXPOSE 7860

# Healthcheck simples (Gradio serve HTML na raiz)
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl -fsS http://127.0.0.1:7860/ >/dev/null || exit 1

# Entry-point: rode o seu main
ENTRYPOINT ["python", "-u", "main_demo_released.py"]
FROM python:3.13-slim

WORKDIR /code

# Installer uv (gestionnaire de dépendances)
RUN pip install uv

# Copier les fichiers nécessaires à uv
COPY pyproject.toml uv.lock ./

# Installer les dépendances via uv
RUN uv sync --frozen

# Copier le reste du code
COPY . .

EXPOSE 7860

# Lancer FastAPI via uv
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

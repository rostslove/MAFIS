# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc python3-dev libffi-dev libssl-dev libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY --from=fedot_industrial pyproject.toml poetry.lock setup.py requirements.txt MANIFEST.in README*.rst LICENSE.md /opt/Fedot.Industrial/
COPY --from=fedot_industrial fedot_ind /opt/Fedot.Industrial/fedot_ind
WORKDIR /opt/Fedot.Industrial
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

ENV FEDOT_INDUSTRIAL_PATH=/opt/Fedot.Industrial

EXPOSE 8001

CMD ["python", "app.py"]

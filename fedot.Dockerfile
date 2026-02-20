FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        gcc \
        python3-dev \
        libffi-dev \
        libssl-dev \
        make \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=$POETRY_HOME python3 -

WORKDIR /app

RUN git clone https://github.com/aimclub/Fedot.Industrial.git && \
    cd Fedot.Industrial && \
    poetry config virtualenvs.create false && \
    poetry install

RUN pip install --no-cache-dir fastapi

COPY fedot_server/fedot_app.py .

EXPOSE 8000

CMD ["python", "fedot_app.py"]

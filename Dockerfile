FROM python:3.11 AS builder
SHELL ["/bin/bash", "-c"]
WORKDIR /app
COPY requirements.txt requirements.txt
RUN python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
COPY . .
RUN source .venv/bin/activate && pip install .

FROM python:3.11 AS runner
SHELL ["/bin/bash", "-c"]
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY workdir/model.json model.json
COPY workdir/data.csv data.csv
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
CMD ["python3", "-m", "fm_training_estimator.ui.cli", "--model_path", "model.json", "--lookup_data_path", "data.csv", "input.json"]

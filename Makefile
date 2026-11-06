SHELL := /bin/bash
NVM   := export NVM_DIR="$$HOME/.nvm" && . "$$HOME/.nvm/nvm.sh"

.PHONY: setup ingest experiment demo mlflow test api frontend dev

# ─── Setup ────────────────────────────────────────────────────────────────────
setup:
	.venv/bin/python src/setup_models.py

# ─── Pipeline ─────────────────────────────────────────────────────────────────
ingest:
	.venv/bin/python src/ingest.py

ingest-reset:
	.venv/bin/python src/ingest.py --reset

experiment:
	.venv/bin/python src/experiment.py

ground-truth:
	.venv/bin/python scripts/build_expected_answers.py

# ─── Demo ─────────────────────────────────────────────────────────────────────
demo:
	.venv/bin/streamlit run demo/app.py

api:
	.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	$(NVM) && cd visualization && npm run dev

dev:
	@$(NVM); \
	trap 'kill 0' SIGINT; \
	.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload 2>&1 | sed 's/^/[api] /' & \
	(cd visualization && npm run dev 2>&1 | sed 's/^/[ui]  /') & \
	wait

# ─── MLflow ───────────────────────────────────────────────────────────────────
mlflow:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ─── Tests ────────────────────────────────────────────────────────────────────
test:
	.venv/bin/pytest tests/ -v

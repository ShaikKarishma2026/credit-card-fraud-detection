"""
Unified Runner for Credit Card Fraud Detection Platform
Provides simple CLI controls to manage database migrations, model training,
backend API server, frontend dashboard, and automated test execution.
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_backend():
    print("[RUN] Starting FastAPI Backend on http://localhost:8000...")
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    subprocess.run(cmd, cwd=BASE_DIR)

def run_frontend():
    print("[RUN] Starting Streamlit Frontend on http://localhost:8501...")
    cmd = [sys.executable, "-m", "streamlit", "run", "frontend/app.py"]
    subprocess.run(cmd, cwd=BASE_DIR)

def setup_db():
    print("[SETUP] Ingesting transactions and setting up database...")
    cmd = [sys.executable, "database/db_setup.py"]
    subprocess.run(cmd, cwd=BASE_DIR)

def train_models():
    print("[TRAIN] Running ML training pipeline (SMOTE, 4 algorithms, evaluation)...")
    cmd = [sys.executable, "models/train_models.py"]
    subprocess.run(cmd, cwd=BASE_DIR)

def run_tests():
    print("[TEST] Running end-to-end test suite...")
    cmd = [sys.executable, "tests/test_pipeline.py"]
    subprocess.run(cmd, cwd=BASE_DIR)

def print_help():
    print("""
=============================================================
💳 Credit Card Fraud Detection Platform - Command Runner
=============================================================
Usage: python run.py [COMMAND]

Available Commands:
  frontend    Start the Streamlit interactive dashboard (port 8501)
  backend     Start the FastAPI REST backend microservice (port 8000)
  setup-db    Ingest transactions and initialize MySQL/SQLite database
  train       Retrain ML models and log evaluation metrics
  test        Execute comprehensive end-to-end test suite
  help        Display this help message
=============================================================
""")

if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "help"
    actions = {
        "frontend": run_frontend,
        "backend": run_backend,
        "setup-db": setup_db,
        "train": train_models,
        "test": run_tests,
        "help": print_help
    }

    action = actions.get(arg, print_help)
    action()

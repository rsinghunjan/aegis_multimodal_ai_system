#!/usr/bin/env python3
"""
Minimal governance Flask API to list MLflow runs and promote a run to 'production'.
Promotion triggers an Argo Workflow (or sets MLflow model stage) and records evidence.

Usage:
  pip install flask mlflow requests
  export MLFLOW_TRACKING_URI=...
  python3 governance/api.py
"""
from flask import Flask, jsonify, request
import os
import mlflow
from mlflow.tracking import MlflowClient
import sqlite3
import subprocess
import json

app = Flask(__name__)
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI","http://mlflow.aegis.svc.cluster.local:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

DB = os.environ.get("GOV_DB","./governance.db")
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS promotions (id INTEGER PRIMARY KEY, run_id TEXT, promoted_by TEXT, notes TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()
init_db()

@app.route("/runs/<experiment_name>")
def list_runs(experiment_name):
    exp = client.get_experiment_by_name(experiment_name)
    if not exp: return jsonify({"error":"experiment not found"}), 404
    runs = client.search_runs([exp.experiment_id], max_results=50)
    out = [{"run_id": r.info.run_id, "status": r.info.status, "metrics": r.data.metrics, "params": r.data.params} for r in runs]
    return jsonify(out)

@app.route("/promote", methods=["POST"])
def promote():
    payload = request.json or {}
    run_id = payload.get("run_id")
    user = payload.get("user","system")
    notes = payload.get("notes","")
    if not run_id:
        return jsonify({"error":"run_id required"}), 400
    # Example: set MLflow model stage if model saved under artifact path. This is a placeholder.
    # In production, create a promotion workflow: deploy, or mark registry stage, and attach signature evidence.
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO promotions (run_id,promoted_by,notes) VALUES (?,?,?)",(run_id,user,notes))
    conn.commit(); conn.close()
    return jsonify({"ok":True, "run_id": run_id})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

#!/usr/bin/env python3
"""
Simple drift detector using Evidently.

- Loads baseline dataset (from S3 features), compares with recent production features (from S3).
- If drift metric exceeds threshold, POSTs to Alertmanager webhook to trigger retrain sensor.

Usage:
  ALERTMANAGER_URL=http://alertmanager.svc.cluster.local:9093 python3 monitoring/evidently_monitor.py --baseline s3://.../features/<snapshot> --prod s3://.../features/<recent>
"""
import os, argparse, json, tempfile
import boto3, pandas as pd, requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

S3 = boto3.client("s3")

def download_s3_file(s3_uri, local):
    bucket, key = s3_uri.replace("s3://","").split("/",1)
    S3.download_file(bucket, key, local)
    return local

def s3_parquet_to_df(s3_uri):
    _, key = s3_uri.replace("s3://","").split("/",1)
    bucket = s3_uri.replace("s3://","").split("/",1)[0]
    tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".parquet")
    download_s3_file(s3_uri, tmp.name)
    return pd.read_parquet(tmp.name)

def post_alert(alertname, summary, labels=None):
    url = os.environ.get("ALERTMANAGER_URL")
    if not url:
        raise SystemExit("Set ALERTMANAGER_URL")
    payload = [{
      "labels": {"alertname": alertname, **(labels or {})},
      "annotations": {"summary": summary}
    }]
    requests.post(url + "/api/v1/alerts", json=payload, timeout=10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--prod", required=True)
    ap.add_argument("--threshold", type=float, default=0.1)
    args = ap.parse_args()
    df_base = pd.read_parquet(args.baseline) if args.baseline.endswith(".parquet") else s3_parquet_to_df(args.baseline)
    df_prod = pd.read_parquet(args.prod) if args.prod.endswith(".parquet") else s3_parquet_to_df(args.prod)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_base, current_data=df_prod)
    res = report.as_dict()
    # Simplified: look for dataset_drift metric
    drift_score = res.get("metrics",{}).get("DataDriftPreset",{}).get("dataset_drift",{}).get("drift_score",0)
    print("Drift score:", drift_score)
    if drift_score >= args.threshold:
        post_alert("ModelDrift", f"Drift detected: score={drift_score}", labels={"severity":"critical"})
        print("Posted drift alert to Alertmanager")
    else:
        print("No significant drift")

if __name__ == "__main__":
    main()

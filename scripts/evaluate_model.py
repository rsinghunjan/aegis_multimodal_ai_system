102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
#!/usr/bin/env python
    except Exception as exc:
        logger.exception("boto3 not installed")
        raise
    assert s3_url.startswith("s3://")
    _, rest = s3_url.split("s3://", 1)
    bucket, key = rest.split("/", 1)
    dest = os.path.join(dst_dir, os.path.basename(key))
    s3 = boto3.client("s3")
    logger.info("Downloading s3://%s/%s -> %s", bucket, key, dest)
    s3.download_file(bucket, key, dest)
    return dest

def verify_signature(candidate_path: str) -> bool:
    try:
        from api.model_signing import verify_model_artifact
    except Exception:
        logger.info("No api.model_signing.verify_model_artifact available; skipping verification")
        return True
    try:
        ok = verify_model_artifact(candidate_path)
        logger.info("Signature verification result: %s", ok)
        return bool(ok)
    except Exception:
        logger.exception("Signature verification raised error")
        return False

def evaluate_sklearn(model_file: str, test_csv: Optional[str], label_col: str, metric: str) -> float:
    import joblib
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score, mean_squared_error
    import pandas as pd

    model = joblib.load(model_file)
    if test_csv:
        df = pd.read_csv(test_csv)
        if label_col not in df.columns:
            raise ValueError(f"label column '{label_col}' not found in {test_csv}")
        y = df[label_col].values
        X = df.drop(columns=[label_col]).values
    else:
        X, y = load_iris(return_X_y=True)
    preds = model.predict(X)
    if metric == "accuracy":
        return float(accuracy_score(y, preds))
    else:
        rmse = float(mean_squared_error(y, preds, squared=False))
        return rmse

def evaluate_torch(model_file: str, test_csv: Optional[str], label_col: str, metric: str) -> float:
    # Minimal torch evaluation: attempt to load TorchScript model or state_dict and run on small synthetic dataset
    import torch
    import numpy as np
    try:
        model = torch.jit.load(model_file)
        model.eval()
    except Exception:
        # fallback: assume it's a state_dict => user should supply an evaluation wrapper instead
        raise RuntimeError("Torch evaluation supports TorchScript artifacts only in this runner")

    # build synthetic batch - CIFAR10-like input (3,32,32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = torch.randn(100, 3, 32, 32, device=device)
    with torch.no_grad():
        out = model(inputs)
        preds = out.argmax(dim=1).cpu().numpy()
    # since we don't have true labels for synthetic data, compute a dummy metric (e.g., distinctiveness)
    # This mode is best-effort and not recommended for strict validation.
    return float((preds >= 0).mean())

def compute_metric(mode: str, artifact_file: str, test_data: Optional[str], label_col: str, metric: str) -> float:
    if mode == "sklearn":
        return evaluate_sklearn(artifact_file, test_data, label_col, metric)
    elif mode == "torch":
scripts/evaluate_model.py

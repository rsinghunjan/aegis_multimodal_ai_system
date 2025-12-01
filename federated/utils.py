import numpy as np
from sklearn.datasets import make_classification
from typing import Tuple


def make_toy_client_data(n_samples: int = 200, n_features: int = 20, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a small toy classification dataset for a federated client.
    Returns (X, y)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(10, n_features),
        n_redundant=0,
        n_classes=2,
        random_state=random_state,
    )
    return X.astype("float32"), y.astype("int64")
  

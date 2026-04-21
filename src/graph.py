from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def build_adjacency_matrix(stations: List[dict], threshold_km: float) -> np.ndarray:
    n = len(stations)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                adj[i, j] = 1.0
            else:
                d = haversine_km(
                    stations[i]["latitude"],
                    stations[i]["longitude"],
                    stations[j]["latitude"],
                    stations[j]["longitude"],
                )
                if d <= threshold_km:
                    adj[i, j] = 1.0
    return adj


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    degree = np.sum(adj, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    degree_inv_sqrt[degree == 0] = 0.0
    d_hat = np.diag(degree_inv_sqrt)
    return d_hat @ adj @ d_hat
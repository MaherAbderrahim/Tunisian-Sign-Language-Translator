"""Hand skeleton topology used by the GNN experiments.

The 21 MediaPipe hand landmarks are indexed as:

    0: wrist
    1-4:  thumb (CMC, MCP, IP, TIP)
    5-8:  index (MCP, PIP, DIP, TIP)
    9-12: middle (MCP, PIP, DIP, TIP)
    13-16: ring (MCP, PIP, DIP, TIP)
    17-20: pinky (MCP, PIP, DIP, TIP)

The graph connects the wrist to each finger base and follows each finger
chain, plus the palm connections (index-middle-ring-pinky bases) — the same
connectivity MediaPipe uses to render the hand skeleton.
"""
import numpy as np

WRIST = 0
FINGER_TIPS = [4, 8, 12, 16, 20]

# Same edges as src/models/landmark_extractor.py::HAND_CONNECTIONS
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

# Finger chains for hierarchical features (each finger: base -> tip)
FINGER_CHAINS = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}


def adjacency_matrix(include_self: bool = True) -> np.ndarray:
    """Return the (21, 21) binary adjacency matrix of the hand graph."""
    A = np.zeros((21, 21), dtype=np.float32)
    for i, j in HAND_CONNECTIONS:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if include_self:
        A += np.eye(21, dtype=np.float32)
    return A


def normalized_adjacency(include_self: bool = True) -> np.ndarray:
    """Symmetric normalization D^-1/2 A D^-1/2 (Kipf & Welling)."""
    A = adjacency_matrix(include_self=include_self)
    D = A.sum(axis=1)
    D_inv_sqrt = np.zeros_like(A)
    nz = D > 0
    D_inv_sqrt[nz, nz] = 1.0 / np.sqrt(D[nz])
    return D_inv_sqrt @ A @ D_inv_sqrt


def neighbors(idx: int) -> list:
    """Neighbor list for one node (used by GAT)."""
    return [j for i, j in HAND_CONNECTIONS if i == idx] + [
        i for i, j in HAND_CONNECTIONS if j == idx
    ]
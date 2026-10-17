"""Train the deep parity network's odd network on simulated markets, then freeze it.

The training data come only from ``softedge.parity.world``. Each sample is the network's input
after week t of a simulated market and the target is week t + 1's normalised excess return,
clipped at 4. The loss is mean squared error, so the network learns the expected next return
given the history, averaged over the prior. The best epoch on validation markets is kept.

    python scripts/parity_deep_train.py --out <weights.npz> --log <log.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time

import numpy as np

from softedge.parity.deep import AdamW, OddNet
from softedge.parity.deep_network import N_EVEN, N_ODD, MarketMemory
from softedge.parity.world import simulate_world

TRAIN_SEEDS = (1001, 1002)
PATHS_PER_SEED = 4096
VAL_SEED = 2001
VAL_PATHS = 1024
N_STEPS = 1040
FIRST = 51
STRIDE = 3
TARGET_CLIP = 4.0
HIDDEN = (64, 64, 64)
EPOCHS = 40
BATCH = 4096
LR = 1e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 4
INIT_SEED = 0


def samples(seed: int, n_paths: int, stride: int):
    """Inputs, targets and null flags from one batch of simulated markets."""
    world = simulate_world(n_paths, N_STEPS, seed)
    excess = world.logret - world.cash
    memory = MarketMemory(n_paths)
    xs, ys = [], []
    for t in range(N_STEPS - 1):
        memory.observe(excess[:, t])
        if t >= FIRST and (t - FIRST) % stride == 0:
            xs.append(memory.inputs())
            ys.append(np.clip(excess[:, t + 1] / memory.sigma, -TARGET_CLIP, TARGET_CLIP))
    k = len(xs)
    return np.concatenate(xs), np.concatenate(ys), np.tile(world.null, k)


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    """Share of the target's second moment explained, against a forecast of zero."""
    return float(1.0 - np.mean((pred - y) ** 2) / np.mean(y * y))


def linear_baseline(x_tr, y_tr, x_va, ridge: float = 1.0) -> np.ndarray:
    """Ridge on the odd inputs alone, the linear model of the same odd form."""
    a = x_tr[:, :N_ODD]
    beta = np.linalg.solve(a.T @ a + ridge * np.eye(N_ODD), a.T @ y_tr)
    return x_va[:, :N_ODD] @ beta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="weights file (.npz)")
    parser.add_argument("--log", required=True, help="training log (.json)")
    args = parser.parse_args(argv)
    t0 = time.time()
    parts = [samples(s, PATHS_PER_SEED, STRIDE) for s in TRAIN_SEEDS]
    x_tr = np.concatenate([p[0] for p in parts])
    y_tr = np.concatenate([p[1] for p in parts])
    x_va, y_va, null_va = samples(VAL_SEED, VAL_PATHS, 1)
    log = {"train_samples": int(y_tr.size), "val_samples": int(y_va.size),
           "data_seconds": round(time.time() - t0, 1), "epochs": []}
    rng = np.random.default_rng(INIT_SEED)
    net = OddNet(N_ODD, N_EVEN, HIDDEN, rng)
    net.fit_scaling(x_tr)
    opt = AdamW(net.mlp.params, lr=LR, weight_decay=WEIGHT_DECAY)
    steps = EPOCHS * math.ceil(y_tr.size / BATCH)
    step = 0
    best, best_state, stale = math.inf, None, 0
    for epoch in range(EPOCHS):
        order = rng.permutation(y_tr.size)
        total = 0.0
        for i in range(0, y_tr.size, BATCH):
            idx = order[i: i + BATCH]
            loss, grads = net.loss_and_grads(x_tr[idx], y_tr[idx])
            lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + math.cos(math.pi * step / steps))
            opt.step(grads, lr)
            total += loss * idx.size
            step += 1
        pred = net.predict(x_va)
        val = float(np.mean((pred - y_va) ** 2))
        row = {"epoch": epoch + 1, "train_mse": total / y_tr.size, "val_mse": val,
               "val_r2": r2(pred, y_va), "val_r2_edge": r2(pred[~null_va], y_va[~null_va]),
               "val_r2_null": r2(pred[null_va], y_va[null_va]),
               "null_mean_abs_forecast": float(np.mean(np.abs(pred[null_va]))),
               "edge_mean_abs_forecast": float(np.mean(np.abs(pred[~null_va])))}
        log["epochs"].append(row)
        print(json.dumps(row), flush=True)
        if val < best:
            best, stale = val, 0
            best_state = {k: np.array(v, copy=True) for k, v in net.state().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    np.savez(args.out, **best_state)
    with open(args.out, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    lin = linear_baseline(x_tr, y_tr, x_va)
    best_net = OddNet.from_state(best_state)
    pred = best_net.predict(x_va)
    log.update({
        "n_params": best_net.n_params, "hidden": list(HIDDEN), "sha256": digest,
        "best_val_mse": best, "best_val_r2": r2(pred, y_va),
        "best_val_r2_edge": r2(pred[~null_va], y_va[~null_va]),
        "best_val_r2_null": r2(pred[null_va], y_va[null_va]),
        "linear_val_r2": r2(lin, y_va), "linear_val_r2_edge": r2(lin[~null_va], y_va[~null_va]),
        "seconds": round(time.time() - t0, 1),
    })
    with open(args.log, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=1)
    print(json.dumps({k: v for k, v in log.items() if k != "epochs"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

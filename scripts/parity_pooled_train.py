"""Train the pooled parity network end to end on simulated panels, then freeze it.

Every step draws fresh panels from ``softedge.parity.panel_world`` and takes one AdamW step on
minus the mean weekly net log growth of the panel backtest, with costs, the band and the crash
governor, and both gates open. The premium channel's weight is drawn per panel (0, 1, or uniform
between them, a third each) and is an input to the network, so the network learns to act with
and without a deployed premium. Fixed validation panels are scored every 100 steps against one
half under the same governor, with the premium weight at 0 and at 1; the parameters with the
best mean gain over the two are kept and written as a flat npz.

    python scripts/parity_pooled_train.py --out <weights.npz> --log <log.json> --ckpt <folder>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from softedge.parity import jaxnet as jn
from softedge.parity import panel_world as pw
from softedge.parity import pooled as pl

HIDDEN = (64, 64, 64)
STEPS = 2000
PANELS = 32
ASSETS = 12
N_STEPS = 1040
COST = 0.001
PEAK_LR = 1e-3
END_LR = 1e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 1e-4
CLIP_NORM = 1.0
EVAL_EVERY = 100
VAL_PANELS = 128
INIT_SEED = 0
SCALING_KEY = 5002
TRAIN_KEY = 7002
VAL_KEY = 9002


def premium_weights(key, n):
    """A third 0, a third 1, a third uniform on (0, 1)."""
    k1, k2 = jax.random.split(key)
    kind = jax.random.randint(k1, (n,), 0, 3)
    return jnp.where(kind == 0, 0.0, jnp.where(kind == 1, 1.0, jax.random.uniform(k2, (n,))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--panels", type=int, default=PANELS)
    parser.add_argument("--val-panels", type=int, default=VAL_PANELS)
    args = parser.parse_args(argv)
    t0 = time.time()

    model = pl.PooledPolicy(HIDDEN, rngs=nnx.Rngs(INIT_SEED))
    ks = jax.random.split(jax.random.key(SCALING_KEY))
    r, c, lst, _ = pw.simulate(ks[0], 32, ASSETS, N_STEPS)
    wp = premium_weights(ks[1], 32)

    @jax.jit
    def sample_inputs(r, c, lst, wp):
        def one(a, b, m, w):
            out = pl.rollout(model, a, b, m, COST, pl.Config(gate=False), w, inputs=True)
            return out["inputs"]
        return jax.vmap(one)(r, c, lst, wp)

    us = np.asarray(sample_inputs(r, c, lst, wp))
    ready = np.asarray(lst) & (np.cumsum(np.asarray(lst), axis=-1) > pl.FIRST_DECISION)
    model.set_scaling(us[ready])
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    decay_mask = jax.tree_util.tree_map_with_path(
        lambda path, _: "kernel" in jn._key(path), params)
    warmup = min(WARMUP_STEPS, args.steps // 2)
    schedule = optax.warmup_cosine_decay_schedule(0.0, PEAK_LR, warmup, args.steps, END_LR)
    tx = optax.chain(optax.clip_by_global_norm(CLIP_NORM),
                     optax.adamw(schedule, weight_decay=WEIGHT_DECAY, mask=decay_mask))
    opt_state = tx.init(params)

    def loss_fn(p, r, c, lst, wp):
        return -pl.panel_growth(nnx.merge(graphdef, p, rest), r, c, lst, COST, wp)

    @jax.jit
    def train_step(p, opt_state, key):
        k1, k2 = jax.random.split(key)
        r, c, lst, _ = pw.simulate(k1, args.panels, ASSETS, N_STEPS)
        wp = premium_weights(k2, args.panels)
        loss, grads = jax.value_and_grad(loss_fn)(p, r, c, lst, wp)
        updates, opt_state = tx.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss, optax.global_norm(grads)

    vr, vc, vl, vnull = pw.simulate(jax.random.key(VAL_KEY), args.val_panels, ASSETS, N_STEPS)
    vnull = np.asarray(vnull)

    def per_panel(p, cfg, w):
        m = nnx.merge(graphdef, p, rest)

        def one(a, b, k):
            out = pl.rollout(m, a, b, k, COST, cfg, w)
            return (out["growth"] * out["scored"]).sum() / jnp.maximum(out["scored"].sum(), 1)
        return jax.vmap(one)(vr, vc, vl)

    open_cfg = pl.Config(gate=False)
    run_val = jax.jit(lambda p, w: per_panel(p, open_cfg, w))
    base = np.asarray(jax.jit(lambda p: per_panel(
        p, pl.Config(gate=False, fixed=0.5), 0.0))(params))

    def gains(p):
        out = {}
        for w in (0.0, 1.0):
            d = 52.0 * (np.asarray(run_val(p, w)) - base)
            out[f"premium_{w:.0f}"] = {"all": float(d.mean()), "edge": float(d[~vnull].mean()),
                                       "null": float(d[vnull].mean())}
        out["score"] = 0.5 * (out["premium_0"]["all"] + out["premium_1"]["all"])
        return out

    ckpt = ocp.CheckpointManager(os.path.abspath(args.ckpt),
                                 options=ocp.CheckpointManagerOptions(max_to_keep=2))
    train_key = jax.random.key(TRAIN_KEY)
    log = {"evals": []}
    best, best_params, best_step = -np.inf, params, 0
    window = []
    for step in range(1, args.steps + 1):
        params, opt_state, loss, gnorm = train_step(params, opt_state,
                                                    jax.random.fold_in(train_key, step))
        window.append((float(loss), float(gnorm)))
        if step % EVAL_EVERY == 0 or step == args.steps:
            g = gains(params)
            row = {"step": step, "train_loss": float(np.mean([w[0] for w in window])),
                   "grad_norm": float(np.mean([w[1] for w in window])), "val_gain": g,
                   "seconds": round(time.time() - t0, 1)}
            window = []
            log["evals"].append(row)
            print(json.dumps(row), flush=True)
            ckpt.save(step, args=ocp.args.StandardSave(jax.tree_util.tree_map(np.asarray,
                                                                               params)))
            if g["score"] > best:
                best, best_params, best_step = g["score"], params, step
    ckpt.wait_until_finished()
    model = nnx.merge(graphdef, best_params, rest)
    pl.save(model, args.out)
    with open(args.out, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    log.update({"sha256": digest, "best_step": best_step, "best_score": best,
                "best_val_gain": gains(best_params),
                "n_params": int(sum(x.size for x in jax.tree_util.tree_leaves(best_params))),
                "hidden": list(HIDDEN), "steps_run": args.steps, "panels": args.panels,
                "assets": ASSETS, "seconds": round(time.time() - t0, 1),
                "jax": jax.__version__})
    with open(args.log, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=1)
    print(json.dumps({k: v for k, v in log.items() if k != "evals"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

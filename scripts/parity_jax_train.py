"""Train the JAX parity network end to end on simulated markets, then freeze it.

Every step draws fresh simulated markets from ``softedge.parity.jax_world`` and takes one AdamW
step on minus the mean weekly net log growth of the differentiable backtest, costs, band and
governor included. Fixed validation markets are scored every 100 steps against the one-half
policy under the same governor; the parameters with the best validation gain are kept and
written as a flat npz. Orbax keeps the latest checkpoints.

    python scripts/parity_jax_train.py --out <weights.npz> --log <log.json> --ckpt <folder>
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

from softedge.parity import jax_world as jw
from softedge.parity import jaxnet as jn

HIDDEN = (64, 64, 64)
STEPS = 2500
BATCH = 256
N_STEPS = 1040
COST = 0.001
PEAK_LR = 1e-3
END_LR = 1e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 1e-4
CLIP_NORM = 1.0
EVAL_EVERY = 100
VAL_PATHS = 1024
INIT_SEED = 0
SCALING_KEY = 5001
TRAIN_KEY = 7001
VAL_KEY = 9001


def path_inputs(model, r, c):
    """The model's inputs after every week of one market."""
    def body(mem, inp):
        x, t = inp
        mem, _, u = model.step(mem, x, t)
        return mem, u

    _, us = jax.lax.scan(body, jn.init_memory(), (r - c, jnp.arange(r.shape[0])))
    return us


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--val-paths", type=int, default=VAL_PATHS)
    args = parser.parse_args(argv)
    t0 = time.time()

    model = jn.ParityPolicy(HIDDEN, rngs=nnx.Rngs(INIT_SEED))
    r, c, _ = jw.simulate(jax.random.key(SCALING_KEY), 1024, N_STEPS)
    us = jax.jit(jax.vmap(lambda a, b: path_inputs(model, a, b)))(r, c)
    model.set_scaling(np.asarray(us[:, jn.FIRST_DECISION:, :]).reshape(-1, us.shape[-1]))
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    decay_mask = jax.tree_util.tree_map_with_path(
        lambda path, _: "kernel" in jn._key(path), params)
    warmup = min(WARMUP_STEPS, args.steps // 2)
    schedule = optax.warmup_cosine_decay_schedule(0.0, PEAK_LR, warmup, args.steps, END_LR)
    tx = optax.chain(optax.clip_by_global_norm(CLIP_NORM),
                     optax.adamw(schedule, weight_decay=WEIGHT_DECAY, mask=decay_mask))
    opt_state = tx.init(params)

    def growth(p, r, c):
        return -jn.mean_growth(nnx.merge(graphdef, p, rest), r, c, COST)

    @jax.jit
    def train_step(p, opt_state, key):
        r, c, _ = jw.simulate(key, args.batch, N_STEPS)
        loss, grads = jax.value_and_grad(growth)(p, r, c)
        updates, opt_state = tx.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss, optax.global_norm(grads)

    vr, vc, vnull = jw.simulate(jax.random.key(VAL_KEY), args.val_paths, N_STEPS)
    vnull = np.asarray(vnull)

    @jax.jit
    def per_market(p):
        m = nnx.merge(graphdef, p, rest)
        g, _ = jax.vmap(lambda a, b: jn.rollout(m, a, b, COST))(vr, vc)
        return g[:, jn.WARMUP:].mean(axis=1)

    half = jax.tree_util.tree_map_with_path(
        lambda path, v: jnp.zeros_like(v) if jn._key(path).startswith("out") else v, params)
    base = np.asarray(per_market(half))

    def gains(p):
        d = 52.0 * (np.asarray(per_market(p)) - base)
        return {"all": float(d.mean()), "edge": float(d[~vnull].mean()),
                "null": float(d[vnull].mean())}

    ckpt = ocp.CheckpointManager(os.path.abspath(args.ckpt),
                                 options=ocp.CheckpointManagerOptions(max_to_keep=2))
    train_key = jax.random.key(TRAIN_KEY)
    log = {"steps": [], "evals": []}
    best, best_params = -np.inf, params
    window = []
    for step in range(1, args.steps + 1):
        params, opt_state, loss, gnorm = train_step(params, opt_state,
                                                    jax.random.fold_in(train_key, step))
        window.append((float(loss), float(gnorm)))
        if step % EVAL_EVERY == 0 or step == args.steps:
            g = gains(params)
            row = {"step": step, "train_loss": float(np.mean([w[0] for w in window])),
                   "grad_norm": float(np.mean([w[1] for w in window])), "val_gain": g,
                   "decays_odd": np.asarray(jax.nn.sigmoid(params["odd_logit"][...])).tolist(),
                   "seconds": round(time.time() - t0, 1)}
            window = []
            log["evals"].append(row)
            print(json.dumps(row), flush=True)
            ckpt.save(step, args=ocp.args.StandardSave(jax.tree_util.tree_map(np.asarray,
                                                                               params)))
            if g["all"] > best:
                best, best_params = g["all"], params
    ckpt.wait_until_finished()
    model = nnx.merge(graphdef, best_params, rest)
    jn.save(model, args.out)
    with open(args.out, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    log.update({"sha256": digest, "best_val_gain": best, "final_val_gain": gains(best_params),
                "n_params": int(sum(x.size for x in jax.tree_util.tree_leaves(best_params))),
                "hidden": list(HIDDEN), "steps_run": args.steps, "batch": args.batch,
                "seconds": round(time.time() - t0, 1), "jax": jax.__version__})
    with open(args.log, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=1)
    print(json.dumps({k: v for k, v in log.items() if k not in ("steps", "evals")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

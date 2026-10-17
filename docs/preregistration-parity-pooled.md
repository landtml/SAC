# Pooled parity network: pre-registered evaluation

Registered on 2026-10-17, before the candidate below is run on any panel listed under
[Evaluation sets](#evaluation-sets). The candidate is frozen in this commit:

| Model | File | SHA-256 |
| --- | --- | --- |
| Pooled parity network, trained end to end in JAX | `models/parity_pooled.npz` | `b55701890462f762eb645e773a6096d60ac052b58bb450abc02a249da5f86d3a` |

The blind synthetic panels come from a generator built to a short brief, without access to the
parity code or any parity result. Nobody who worked on this network has read the generator
before this commit. Its SHA-256 is
`e3bf00b60b4019861005f381321a8e1c2e2388411cc5b6027f7db8dab4cbea84`; it is added to the
repository after the test, unchanged, as `research/parity/blind_panels.py`.

What was seen before registration: training and validation on simulated panels; development
results on sets that earlier registered tests had already used (the crypto cross-section, the
currency panel, and the held-out families of the deep test), listed in
[Development](#development); the coverage counts of the macro panel (dates and weeks only); and
the blind module's self-check report (shapes, finiteness, volatility range), which names no
family.

## Question

The deep parity network failed its registered test because its evidence gate never opened on
short or noisy histories, its governor let 93 of 406 coins breach, and its default ignored the
premium over cash. Does pooling evidence and memory across a panel of assets, a governor sized
to each asset's crash risk, and a gated premium fix those failures without trading noise?

## The candidate

Code: `src/softedge/parity/pooled.py`, trained by `scripts/parity_pooled_train.py` on panels from
`src/softedge/parity/panel_world.py`.

- **Inputs.** Each asset's memories as in the deep network (25 inputs), the mean of those
  memories over the assets listed that week (25), log(1 + listed assets), and the premium
  channel's weight. The odd network (three hidden layers of 64 SiLU units) is odd in the 20 odd
  inputs, so mirroring every excess return flips its output.
- **Premium.** Each asset's Sharpe ratio of excess log return (EWMA mean with a 260-week decay,
  260-week volatility, effective sample size from the EWMA weights), shrunk towards the panel's
  precision-weighted mean by tau^2 / (tau^2 + 1 / n). The logit is s / sigma, half Kelly,
  clipped to +-3.
- **Exposure.** (1 + tanh(w_p p + w_d g)) / 2, where p is the premium logit, g the network's
  conviction, and w_p, w_d the weights of two pooled ledgers.
- **Pooled ledgers.** Each listed asset bets clip(forecast / (2 x RMS of its past forecasts),
  -1, 1) on psi(z) = clip(z, -2, 2) / 2 of next week. The ledger compounds 1 + lambda u-bar,
  where u-bar is the average settlement over the assets that bet; lambda =
  clip(sum u-bar / (sum u-bar^2 + 1), 0, 0.5) from earlier weeks. Evidence is a CUSUM of log
  capital and the weight is (1 - E0 / E)+ with E0 = log(2 / 0.1).
- **Crash governor.** cap = clip((W - 0.76 H) / (W c), 0, 1), with the crash size
  c = clip(max(worst own weekly loss, m sigma), 0.10, 1), sigma the asset's EWMA volatility and
  m the median, over the panel's assets, of each asset's largest weekly loss so far in
  volatility units, clipped to [8, 32]. An asset above its cap is traded down whatever the band.
- **Training.** 2,000 AdamW steps on 32 fresh simulated panels of 12 assets and 1,040 weeks,
  gates open, premium weight drawn per panel, cost 10 bp; peak learning rate 1e-3, 100 warm-up
  steps, cosine decay to 1e-5, weight decay 1e-4 on kernels, gradient clipping at 1. The best of
  20 validation checkpoints on 128 fixed simulated panels is the candidate.
- **Training result, on simulated panels only.** The best checkpoint is step 1,700 of 2,000.
  With the gates open it gains 0.53 points a year of net log growth over one half under the
  same governor on the validation panels: 1.08 on panels with an edge and -0.33 on null
  panels, which the gates exist to prevent. The model has 11,790 trained parameters. Training
  took 95 minutes on CPU, from commit c102082.

## Development

Every development choice and its result, on burned sets only, is listed here. The governor
constants were chosen with a fixed exposure of one half, before any network was trained.

| Step | Choice | Result on development sets |
| --- | --- | --- |
| D1 | Governor family, with a fixed exposure of one half on the crypto panel | Breaching coins: no governor 405, first network's governor 70, crash governor 1, CPPI 9 |
| D2 | Tail statistics | Worst weekly loss in volatility units: crypto coins median 2.9, largest 10.6; currencies up to 48.5 (AUD) |
| D3 | Crash floor in {0.05, 0.10, 0.15} and tail cap in {12, 16, 24, 32, none}, at exposures of one half and one | The floor made no difference; currencies breached once at full exposure with caps up to 24. Chosen: floor 0.10, cap 32 |
| D4 | First trained network, panel maximum of tail multiples | Against CPPI on the three families: -0.05, +0.12, -0.01. Currencies: the same network under the first governor earned 0.51 more |
| D5 | Tail rule: panel maximum, own, panel median (network of D4 held fixed) | Crypto 5.07%, 5.35%, 5.35% (1 breach each); currencies 4.37%, 4.65%, 5.07% (0 breaches each). Chosen: panel median, and the network retrained under it |
| D6 | Retrained network (the candidate) | See the table below |

The candidate on the development sets, in points of annual net log growth (95% intervals):

| Set | Candidate minus best simple rule | Minus the same network on each asset alone | Minus the deep network | Breaches | Gate deployed |
| --- | --- | --- | --- | --- | --- |
| `regime_switching`, 50 panels of 10 | -0.05 vs CPPI (-0.09 to -0.00) | +0.22 (+0.17 to +0.27) | +0.24 | 0 | 38% |
| `ar_returns`, 50 panels | +0.08 vs CPPI (+0.02 to +0.14) | +0.37 (+0.30 to +0.44) | +0.38 | 0 | 46% |
| `ou_mean_reverting`, 50 panels | -0.01 vs CPPI (-0.02 to +0.01) | +0.24 (+0.23 to +0.26) | +0.24 | 0 | 40% |
| Crypto, 406 coins | -0.34 vs CPPI (-2.52 to +1.70) | 0.00 | -0.18 | 1 (CPPI 9) | 0% |
| Currencies, 13 | +1.10 vs Grossman-Zhou (+0.41 to +1.84) | +0.76 (+0.23 to +1.27) | +0.73 | 0 | 98% |

On crypto neither forecast predicts next week's return (correlations -0.005 and 0.010 over
scored coin-weeks), so the gate stays shut and the candidate holds one half under the crash
governor. The tail rule was chosen partly on the currency panel, so the currency result is
flattered. Trial count for this registration: two trained networks and the development choices
above; one candidate is registered.

## Evaluation sets

| Set | Content | Accounting |
| --- | --- | --- |
| Blind synthetic | The four families of the blind module, seeds 1 to 100, panels of 10 assets and 1,040 weeks | The module's cash path; cost 5 bp |
| Macro | NASDAQ, NIKKEI, WTI, Brent, and 10-, 5- and 2-year Treasury total-return proxies, weekly to 2005-12-30 (`scripts/parity_macro.py`) | DTB3 cash; costs 5 bp for equities and oil, 2 bp for Treasuries |
| Null | The first 25 blind panels of each family and the macro panel, with each week's excess returns multiplied by one random sign common to all assets (seed 20261017) | As above |

The macro panel stops at 2005-12-30 so that it does not touch 2006 to 2016-09, the test window
of the long-only weekly search. Its series were used by other studies of this project, and no
parity network has been run on them.

Accounting as in the earlier parity tests: 52 weeks of warm-up per asset from its own first
week, the 0.05 band, and a breach when wealth falls more than 25% below its peak.

## Claims

**P1, blind synthetic.** Over the 400 blind panels, the per-panel annual difference in net log
growth between the candidate and its family's best feasible simple rule has a 95% t-interval
above zero. The best rule is the feasible one (at most 1% of the family's assets breach) with
the highest growth. The candidate breaches on at most 1% of each family's assets.

**P2, macro.** The pooled difference against the panel's best feasible simple rule has a 95%
block-bootstrap interval above zero (calendar weeks, blocks of 52, 10,000 draws), and the
candidate breaches on no more series than that rule. If no rule is feasible, the rule with the
fewest breaches is used.

**P3, pooling earns its place.** Over the 400 blind panels, the per-panel difference between
the candidate and the same network run on each asset alone has a 95% t-interval above zero.

**P4, null and leaks.** On the null panels either ledger is deployed in at most 2% of scored
asset-weeks. Flipping the excess return of every asset in every week after a cut leaves every
earlier target of every asset unchanged, on the first 8 blind panels of each family and on the
macro panel, at cuts 60, 200, 400, 600, 800 and 1,000.

Reported with no pass rule: the ungated network, premium off, direction off, one half under the
crash governor, the network under the old governor, the first parity network, the deep network,
the six simple rules and the governed trend gate; per-series results on the macro panel.

## Decision rules

- The candidate, its code and every constant are fixed by this commit. A changed model is a new
  registration.
- If P4's probe fails, every result is void.
- A diagnostic that beats the candidate is reported, not claimed.
- The blind panels, seeds 1 to 100, and the macro panel to 2005 are not reused to evaluate a
  revised version of this candidate.
- For the long-only weekly search, this run counts as one trial on its development window.

## Software

JAX 0.11.2, Flax 0.12.10 (NNX), Optax 0.2.8, Orbax 0.12.7, on CPU.

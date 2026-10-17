# SoftEdge documentation

This index lists the documentation pages. SoftEdge asks whether a Soft Actor-Critic agent can find
a trading edge that holds up on data it never trained on. The [project README](../README.md) gives
the overview.

## Pages

- [Getting started](getting-started.md): install, run the synthetic check, download data, read a report.
- [Results](results.md): what the agent found on synthetic and real markets, with the limits.
- [Glossary](glossary.md): plain meanings of the terms used across these pages.
- [Configs](../configs/README.md): every config file and what it changes.
- [Roadmap](roadmap.md): open research questions.
- [Research log](research-log.md): every market framing and algorithm change tried, with its outcome.
- [Directional prediction study](preregistration-direction.md): the pre-registered hypotheses, data, controls and pass rule, fixed before any test was run.
- [Drawdown-budget checks](preregistration-drawdown-frontier.md): the pre-registered checks of the crash-budget frontier for the 60/20/20 mix, fixed before any exact check was run.
- [Long-only weekly search](preregistration-long-only-weekly.md): the pre-registered long-only, weekly-rebalance families and pass rule, fixed before any result.
- [Discovery search](preregistration-discovery.md): the pre-registered search for a learner by trial and error on synthetic data, with its kill rules, fixed before any candidate exists.
- [Parity network](parity-network.md): a learner built around the sign symmetry of returns, with evidence ledgers and a drawdown governor.
- [Parity network test](preregistration-parity-network.md): the pre-registered evaluation of the parity network on synthetic families, fixed before any registered tape was scored.
- [Parity network on currencies](preregistration-parity-network-fx.md): the registered real-data check of the same network on 13 currency deposits held by a US-dollar investor.
- [Deep parity network test](preregistration-parity-deep.md): the registered evaluation of the deep version, trained end to end in JAX on simulated markets, on held-out synthetic families and the crypto cross-section.
- [Technical design](design.md): the reasoning behind each design choice, and its known limitations.
- [Supervised forecasting architecture](architecture.md): a design for a forecast-then-size research track.

New to the project? Read the results, then the glossary entries they link to.

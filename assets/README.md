# Brand assets

All files here are MIT licensed, like the code.

The SoftEdge mark is the agent's policy: a Gaussian over target positions, with a vertical line
at zero. The green area is the probability of holding a long position, the edge the agent thinks
it has. When the evidence is weak the policy moves back to zero and the agent stays flat.

- `logo.svg`: the mark on a dark tile, for icons down to 32 px.
- `banner.svg`: mark, wordmark and tagline for the repository header. It has its own background, so it reads on light and dark GitHub themes.
- `hero.svg`: the animated README header; `hero.gif` and `hero.png` are its renderings
  (`scripts/render_hero.mjs`).
- `png/`: 2x PNG renderings of the static SVGs, for viewers that do not draw SVG
  (`scripts/render_assets.mjs`).
- `social-preview.png`: the 1280x640 image GitHub shows in link previews.

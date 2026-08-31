# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository (`rl_lab` — the teaching library, published
to PyPI).

This repo is independent of the `rl-course` course-content repo
(`~/projects/rl-course`, sibling directory) — see that repo's own
`CLAUDE.md` and `directory-structure.md` for how the two relate. Nothing
in this repo should be instructor-only or course-specific: the library is
public by design.

## Audience & style
- Author (Luiz) is a mathematician with deep background in stochastic
  control, dynamic programming, and applied probability.
- Direct, precise, intermediate-to-advanced. Don't explain standard RL
  foundations (MDPs, Bellman equations, etc.) from first principles.
- End users are students running this via `pip install rl_lab` inside
  Google Colab — no local GPU or local Python environment assumptions
  anywhere in library code.

## Commands
Own venv at the repo root (`.venv/`, Python 3.11 — pinned by
`requires-python = ">=3.11,<3.12"` in `pyproject.toml`):

```
source .venv/bin/activate
```

- Install in editable mode after any source change, then smoke-test:
  `pip install -e .` followed by `python -c "import rl_lab"`
- Reinstall dependencies:
  - `pip install -r requirements.txt` — full pinned freeze, reproduces
    the tested dev environment.
  - `pip install -r requirements-core.txt` — loose/unpinned core deps
    only, mirrors `pyproject.toml`.
- Run tests: `pytest` (single test:
  `pytest tests/test_mdp.py::test_recycling_robot_creation`)
- Clean build artifacts (`__pycache__`, `build/`, `dist/`, `*.egg-info`):
  `./clean`

## Architecture
- `src/rl_lab/mdp.py` — the `MDP` class, the only real code so far and
  the core abstraction the rest of the library builds on. States and
  actions are stored as indexed tuples; transition probabilities and
  rewards are dense numpy arrays of shape
  `(n_actions, n_states, n_states)`. Transition specs accept a `'*'`
  sentinel probability meaning "remaining probability mass to reach 1",
  which must be the last entry in a transition list — `__init__`
  validates this along with probability sums, duplicate states/actions,
  and missing transition specs for non-terminal states.
- `tests/test_mdp.py` already exercises a `.P(state, action)` /
  `.R(state, action)` query API on `MDP` that doesn't exist yet (only
  the bulk array properties `transition_probs` / `rewards` are currently
  exposed) — expect this gap to close as the API evolves, don't assume
  it's a regression.
- `src/rl_lab/envs/`, `agents/`, `utils/` are placeholder dirs from the
  planned layout (see the repo's `directory-structure.md` in
  `rl-course` for the original full-layout sketch) — empty for now.

## Packaging & release
- src-layout (`src/rl_lab/...`), not flat — avoids import-path issues
  during local testing, standard with setuptools build backends.
- Version: semver in `pyproject.toml`. Tag releases as `vX.Y.Z` (no more
  `rl_lab-v` prefix — that existed only to disambiguate from course tags
  back when this lived inside the `rl-course` monorepo) and push the
  tag — `.github/workflows/publish.yml` builds and publishes to PyPI via
  Trusted Publishing (OIDC). No path-scoping needed on the trigger since
  this whole repo is the package.
- Before tagging: run the full test suite and the editable-install
  smoke test above.
- To let `rl-course` test against an unreleased version, that repo can
  `pip install -e ../rl_lab` (editable sibling install) instead of
  waiting for a PyPI release.

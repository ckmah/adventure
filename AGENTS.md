# AGENTS.md

## Cursor Cloud specific instructions

This repo (`adventure`) is a [pixi](https://pixi.sh) workspace of [marimo](https://marimo.io)
reactive notebooks ("quests") for spatial-transcriptomics analysis. The interactive core is the
`LandmarksWidget` ("draw a landmark, then measure spatial relationships to cells") from the
`spatial-rx` package.

### Toolchain
- Package manager is `pixi` (installed at `~/.pixi/bin`, on PATH via `~/.bashrc`). There is no
  npm/pip/uv workflow. The environment lives at `.pixi/envs/default` (Python 3.14).
- Run anything inside the env with `pixi run <cmd>` (e.g. `pixi run python ...`,
  `pixi run marimo ...`). `[tasks]` in `pixi.toml` is empty, so there are no predefined tasks.
- `pixi.toml` `platforms` includes both `osx-arm64` (the author's machine) and `linux-64` (cloud).
  Keep `linux-64` present or `pixi install` will fail on this VM.

### Dependencies gotchas
- `spatial-rx` is declared as a git dependency (`{ git = "https://github.com/ckmah/spatial-rx.git" }`).
  Upstream it was an editable local path (`../spatial-rx`), which cannot exist here because the
  parent of `/workspace` is not writable. If you need to hack on `spatial-rx` and `adventure`
  together, clone `spatial-rx` somewhere writable and point the dep at it (`{ path = "...", editable = true }`).
- The older quest `quests/20260811_widgets.py` imports `wigglystuff`, which is NOT declared in
  `pixi.toml` (the newer showcase replaced it with `spatial-rx`). Add `wigglystuff` to
  `pypi-dependencies` if you need to run that notebook.

### Dataset requirement (important)
- Both quests call `sd.read_zarr("data/mouse_liver.zarr")`. That SpatialData store (the real
  Guilliams et al. mouse-liver dataset) is NOT in the repo and has no download script, so the
  notebooks error at the data-loading cell until it is provided at `data/` (gitignored / untracked).
- For local dev without the real data you can generate a small SYNTHETIC placeholder store that
  matches the expected schema (multiscale `raw_image`, `segmentation_mask` labels, `table` AnnData
  with `obs.cell_ID`/`obs.annotation`/`obsm.spatial` and gene var-names, `transcripts` points with
  `x`/`y`/`gene`, and `nucleus_boundaries` shapes indexed by cell ID). This only exercises the
  pipeline; it is not the real dataset and should not be committed.

### Run / lint / test
- Run a quest as an interactive app: `pixi run marimo run quests/<file>.py --headless --port 2718 --host 0.0.0.0`.
- Authoring/dev mode: `pixi run marimo edit quests/<file>.py --headless --port 2718 --no-token --host 0.0.0.0`.
  Note: in this cloud browser, `marimo edit` live cell outputs did not render reliably (cells looked
  stale); `marimo run` and `pixi run marimo export html quests/<file>.py -o out.html` both render
  outputs correctly. Prefer `marimo run` for GUI demos.
- Lint/format check: `pixi run marimo check quests/`.
- `adventure` has no test suite of its own. The core widget logic is tested in the `spatial-rx`
  repo (`pytest tests/`).

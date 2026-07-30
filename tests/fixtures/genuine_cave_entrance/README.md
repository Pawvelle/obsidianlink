# `tests/fixtures/genuine_cave_entrance/` — provenance

This directory holds manually-reviewed, byte-verified copies of MineRL
seed-101 genuine-cave-entrance frames. Provenance is **not** a guess: it
is reproducible by anyone with shell access to the repository.

## `entrance.png` — initial natural entrance

- Source: `runs/manual-findcave/20260723-110256/candidates/candidate-tick-03026.png`
- Source date: 2026-07-23 11:05
- Source size: 144431 bytes
- Source MD5: `a5cf9195457f25a17d1bc527f4b08651`
- Manual validation: `runs/manual-findcave/20260723-110256/positive_validation.md`

This is the same frame referenced in `ROADMAP.md` Phase 4 as the
human-operated real seed-101 entrance. The positive validation in
`runs/manual-findcave/20260723-110256/positive_validation.md` records the
manual review that confirmed the same dark stone opening.

- Fixture size: 144431 bytes
- Fixture MD5: `a5cf9195457f25a17d1bc527f4b08651`
- Fixture date: 2026-07-23 11:12
- Expected ground truth: `cave_visible=true`, direction `center`.

## `after_approach_right.png` — same entrance, post-approach view

- Source: `runs/phase4-true-entrance-approach/20260723-142315/episode-01/decision_frames/tick-0235.png`
- Source size: 207870 bytes
- Source MD5: `8d814f039bfb2983a5e8c1022a04b559`
- Source SHA-256: `e1d4e1318d06be6ba82ea1dbeb40fe000d1e8e282ee97b332437f75a5627d6e9`
- Manual validation: `runs/phase4-true-entrance-approach/20260723-142315/episode-01/manual_review.md`

The two evidence frames in that manual review establish the same natural
stone-bounded entrance at observation ticks 0 and 235. The tick-0235
frame is the same entrance after the player has approached and turned
relative to the target; the opening is now larger and offset to the
right of the current view.

- Fixture size: 207870 bytes
- Fixture MD5: `8d814f039bfb2983a5e8c1022a04b559`
- Fixture SHA-256: `e1d4e1318d06be6ba82ea1dbeb40fe000d1e8e282ee97b332437f75a5627d6e9`
- Expected ground truth: `cave_visible=true`, direction `right`.

## Use in benchmarks

These fixtures are the only positive samples in the Phase 6.3 MiniMax-M3
offline visual benchmark (see `scripts/benchmark_minimax.py`). The round-1
fixture set used only `entrance.png`; the expanded-fixture round
introduces `after_approach_right.png` as a second positive with a
different expected direction to test cross-view generalization. Any
future change that adds, removes, or replaces a positive must update the
matching `ROADMAP.md` Phase 6.3 section and the matching `README.md`
section here, and must keep the round-1 fixture path stable for
comparability.

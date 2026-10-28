# Occasional utilities

Run from `Diss` with the experiment environment active:

| Command | Purpose |
| --- | --- |
| `python -m tools.demo_landmarks` | Preview and export landmarks on the five curated demo images |
| `python -m tools.make_placeholders` | Generate cartoon stimuli and `trials_placeholder.csv` for the placeholder option |

Use `python demo.py` for the five-image experiment and `python analyse.py --test`
for its gaze report. Landmark previews are separate from measured gaze and
reviewed facial AOI masks.

The placeholder generator writes to `stimuli/real` and `stimuli/fake`; check
whether those paths hold images you need before invoking it.

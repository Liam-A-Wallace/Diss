# Occasional utilities

Run from `Diss` with the experiment environment active:

| Command | Purpose |
| --- | --- |
| `python -m tools.demo_landmarks` | Preview and export landmarks on the five curated demo images |
| `python -m tools.diagnose_aois` | Compare eight fixed faces: landmarks, full-frame and face-crop segmentation, confidence, thresholds, resolution and historical v2 masks |
| `python -m tools.make_placeholders` | Generate cartoon stimuli and `trials_placeholder.csv` for the placeholder option |

Use `python demo.py` for the five-image experiment and `python analyse.py --test`
for its gaze report. Landmark previews are separate from measured gaze and
reviewed facial AOI masks.

See [model diagnostic instructions](../docs/AOI_MODEL_DIAGNOSTICS.md). The
comparison writes a new report under `processed/` without changing AOIs or
approvals. See [detailed AOI definitions](../docs/DETAILED_AOIS.md) for the
current masks.

The placeholder generator writes to `stimuli/placeholders/real` and `stimuli/placeholders/fake`; check
whether those paths hold images you need before invoking it.

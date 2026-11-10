# AOI model diagnostics

Run from Diss:

```powershell
python -m pip install -r requirements-diagnostics.txt
python -m tools.diagnose_aois
```

The tool creates a new timestamped directory under `processed/`. It does not
change production masks, geometry, thresholds, reviews or gaze analysis.
It uses the existing local parser and sibling GazeFollower landmark weights;
no files outside Diss are written or deleted.

Eight fixed cases (four real, four OpenAI-generated) cover earlier reported
eye/brow failures, glasses, unequal eye coverage and high uncertainty. This is
a failure-case study, not a representative estimate of model performance.
The selection and reasons are recorded in `selection.csv`.

For each image the report compares:

- Original image and raw MediaPipe points with linked feature contours.
- Unthresholded SegFormer output at 512 and 1024 square input resolution.
- Maximum softmax confidence, with the same black=0 to white=1 display scale.
- Thresholded predictions at 0.4, 0.5 and 0.6 for both resolutions.
- The exact saved historical detailed-v2 mask, when available. It is not
  regenerated during the comparison.
- A separate 512 square parser pass on a square face crop defined by MediaPipe
  bounds plus 25% padding. Crop predictions are mapped back to source pixels;
  this is an experiment and does not alter production masks.

Original-sized arrays and landmark coordinates are saved for later reference
evaluation. The raw model's 19 classes are retained without study-label merging.
`feature_metrics.csv` reports raw/retained class area, retention fraction,
mean confidence and connected-component counts. More components can flag
fragmentation but are not automatically errors (small objects can be valid).
`resolution_agreement.csv` measures agreement between the two predictions;
it is explicitly not ground-truth Dice accuracy. Absent-class metrics are blank
where a denominator would be zero.

Use `manual_review.csv` to record alignment/segmentation judgements separately
for eyes, brows, nose and mouth. All cells begin unrated. Reference mask paths
can be attached after independent human annotation; no accuracy values are
fabricated from model predictions. Document the annotator, rules and disagreements.

Interpreting the comparisons:

1. If raw dots miss feature contours, investigate landmark detection first.
2. If dots align but raw segmentation misses the feature, investigate parser
   inference/model quality. Threshold changes cannot repair raw class placement.
3. If raw features align but the saved detailed AOI is wrong, inspect custom
   geometry or class remapping. Only then change the geometry.
4. Higher confidence, larger regions and 1024-resolution disagreement are not
   evidence of improved accuracy without reference annotations.
5. Validate participant gaze accuracy separately before selecting AOI granularity.

See [detailed AOI definitions](DETAILED_AOIS.md) for the current region geometry.

The report records image, model and script hashes, settings and versions.
Visual judgements remain provisional until reference annotations are available.

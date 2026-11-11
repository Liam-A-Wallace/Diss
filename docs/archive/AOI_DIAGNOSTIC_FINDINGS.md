# Initial model diagnostic findings — 10 November 2026

Completed report: `processed/aoi_diagnostics_20261110_180449/index.html`.
Eight fixed troublesome faces: four real, four OpenAI-generated. All produced
one detected MediaPipe face. Sixteen parser inferences compared the same model
at 512 and 1024 input resolution. Each prediction was compared at confidence
thresholds 0.4, 0.5 and 0.6. Production code, masks and reviews were unchanged.

Historical snapshot: image names refer to the files used on 10 November 2026.
`real_003` has since been replaced, and the active demo and trial list have
changed. These observations and measurements do not describe the replacement.

## Measured retention

At 512 input, the proportion of raw class-labelled pixels surviving each cutoff
was as follows. Counts pool original-image pixels, so larger images contribute
more. These are NOT accuracy or recall measurements against human ground truth.

| Cutoff | Eyes | Eyebrows | Nose | Mouth/lips |
|---|---|---|---|---|
| 0.4 | 97.4% | 98.0% | 100.0% | 99.9% |
| 0.5 | 82.4% | 88.3% | 99.7% | 99.4% |
| 0.6 | 55.1% | 71.2% | 97.1% | 94.8% |

At 1024 and cutoff 0.6, the corresponding figures were 57.0%, 50.8%, 95.3%
and 84.7%. Higher resolution does not automatically increase confident feature
coverage. Neither confidence nor retained area establishes anatomical correctness.

## Provisional visual findings

The eight face crop sheets were inspected alongside originals. These are
assistant visual observations, not independent human reference annotations.

- `real_001`: raw eyebrows cover more of the visible brow than the 0.6 masks;
  the cutoff removes much of the image-right brow. The 1024 prediction also
  changes some left/right eyebrow class assignments.
- `real_003`: narrow eye apertures and eyebrow class mixing are present in raw
  output; cutoff 0.6 removes substantial eye/brow coverage. 1024 is mixed.
- `real_007` and `real_009`: glasses occupy much of the eye vicinity in the parser
  labels, while landmark points are still placed around the visible eyes.
  Thresholding further reduces the separate eye predictions.
- `pair_001_A`: raw core feature placement is substantially more coherent than
  the fragmented impression of the saved detailed skin subdivisions. 1024 adds
  eyebrow class mixing/uncertainty rather than an obvious uniform benefit.
- `pair_002_B`: glasses/eye class competition persists at both resolutions;
  the 0.6 cutoff can remove an entire confident eye region. A lower cutoff does
  not resolve the underlying semantics of an eye behind transparent glasses.
- `pair_004_B`: the 512 raw nose is contiguous, but at 1024 it splits into nasal
  patches with intervening skin. This is direct evidence against assuming that
  higher resolution improves this model on every face. Cutoff 0.6 also removes
  substantial eye/brow pixels.
- `pair_008_B`: reflections/glasses remain a challenge for eye labels at both
  resolutions. Raw nose/mouth placement is more coherent than several saved
  detailed skin-region boundaries.

MediaPipe locates the core feature areas in these examples, but some eyebrow
guides sit above the visible hair. Landmark dots and the linked contour guides
are geometry cues, not pixel-boundary ground truth. No landmark error in pixels
has been established. The nose guide is a selected landmark chain, not an
official semantic nose boundary.

The saved v2 comparison shows arbitrary cheek/lower-face divisions independent
of raw SegFormer class placement. Correcting pixel confidence alone cannot
repair those historical derived-region definitions.

## Decisions supported by this run

1. Preserve the current production configuration during reference evaluation.
   Do not switch globally to 1024 or a lower threshold based on these pictures.
2. Prioritise threshold sensitivity and glasses/eye semantics for core-feature
   failures. The 0.6 cutoff clearly causes substantial additional pixel removal.
3. Evaluate landmark alignment separately, especially brows and occlusions.
   Face detection success is not landmark accuracy.
4. Treat custom cheek/chin/jaw errors as an independent problem. Do not rebuild
   them until the core model/reference comparison is sufficiently reliable.
5. Complete the generated manual_review.csv and attach independently drawn
   reference masks. Ground-truth IoU/Dice, boundary error and a claim of the
   globally dominant source of error cannot be justified without those references.
6. Measure participant gaze accuracy separately before deciding whether small
   adjacent AOIs support the dissertation's comparisons.

Model/image hashes,
settings, arrays, landmark coordinates and comparison metrics are saved in the
run folder.

## Follow-up crop experiment — 10 November 2026

Run: `processed/aoi_diagnostics_20261110_183148/index.html`. The same parser
was run at 512 on square crops bounded by MediaPipe face landmarks with 25%
padding, then projected back to original pixels. The crop condition looked
similar to full-frame 512 across these eight faces; it did not consistently
repair glasses/eye competition or other raw class-placement errors. Pooled
0.6-retention for crop eyes/brows was 51.5%/70.6%, versus 55.1%/71.2% for
full-frame 512. There is no evidence here to switch production inference to
face crops.

The current masks retain the 0.6 cutoff. See [detailed AOI definitions](../DETAILED_AOIS.md)
for the skin-region geometry used in detailed-v4.

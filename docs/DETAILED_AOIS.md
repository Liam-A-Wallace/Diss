# Detailed facial AOIs

Detailed-v4 combines parser labels with landmark-based skin regions. Review
the masks before using them in gaze analysis.

## Generate and review masks

```powershell
python -m pip install -r requirements-aois.txt
python analyse.py aois --detailed --test
python analyse.py review --detailed --test
python analyse.py --detailed --test
```

Outputs use `aois/test/detailed-v4` (or `aois/main/detailed-v4` without `--test`).
Review every overlay before entering `reviewed`. Generation never approves
masks. The same `--detailed` flag selects the correct masks during analysis.
The local landmark weights currently come from the sibling GazeFollower
checkout; missing weights produce an explicit error. Model, geometry and landmark weight hashes identify each set;
changing an existing set's implementation/settings requires a new version.

The mask retains 19 parser feature classes and adds six positional skin labels.
Image-left and image-right always mean the displayed image, not the subject's
anatomical side. Paired raw feature classes are combined and split along the
mesh eye axis to prevent mixed side IDs. Without a valid mesh those sided
features and all derived regions are withheld; the remaining parser labels
remain available. Severe pose asymmetry withholds all derived regions.

Masks are exclusive pixel partitions. Forehead skin is separated using brow
and temple landmarks, extending to connected confident skin at the parser's
hair/hat boundary, rather than stopping at the mesh upper oval. Under-eye
ribbons follow lower-eyelid contours. The chin follows the lower outer lip
and lower jaw. Cheeks follow each side's face outline, eye corners, nasal wing
and mouth corner landmarks. Existing parser features and uncertain pixels
are never overwritten. Remaining perioral skin can stay other_skin. These
boundaries are operational AOI definitions, not learned anatomy. See
[v4 investigation and validation](AOI_V4_FINDINGS.md).
The unchanged 0.6 cutoff applies to all parser classes. Missing eyes, brows,
nose or mouth, fragmented side labels, insufficient skin and unsupported pose
are recorded as review flags. Eye/parser disagreement withholds that side's
under-eye region. Other skin is intentionally retained between small regions.

The manifest records label sources, an explicit broad parent mapping, and raw
landmark CSVs. Parent mapping is metadata for downstream aggregation; current
gaze analysis reports detailed labels and does not also count parent labels.


## Limits

Derived skin labels describe approximate positions, not eye bags, wrinkles or
other appearance details. Anatomical accuracy needs independent reference masks.
Measure participant gaze error before comparing small adjacent regions.

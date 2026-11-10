# Detailed-v4 geometry checks

## Causes established from the implementation and saved outputs

- V3 used explicit ellipses for all six derived skin regions. Their small
  radii caused the incomplete coverage and circular appearance. This was an
  AOI construction problem independent of parser detection.
- V3 also restricted forehead skin to the mesh face oval. The top mesh point
  is not a hairline detector, so this prevented forehead coverage from reaching
  the parser's actual skin/hair boundary.
- The 0.6 cutoff rejects many raw eye and eyebrow pixels. The earlier eight-face
  diagnostic retained 55.1% of raw eye pixels and 71.2% of raw brow pixels.
  This explains some missing edges but is not a measurement of accuracy.
- Raw parser predictions still contain narrow eyes, glasses competition,
  simplified nose/lip boundaries and occasional fragmented hair/ear labels.
  Landmark geometry cannot establish the correct replacement for those pixels.
- Teeth have no separate trained class. The parser mouth-interior label can
  include teeth. V4 does not misrepresent this as a teeth detector.
- Yellow marks were the palette colour of label 255 (uncertain). No boundary
  stroke was drawn. Blending every mask colour with the photograph explains
  the difference from opaque legend swatches.

## V4 implementation

The ellipse builders were removed. Forehead uses the brow/temple contour and
connected parser-confirmed skin up to hair/hat/uncertainty boundaries. Its
lateral extent is limited to face width. Under-eye regions are curved ribbons
following each lower eyelid; their lower edge is an explicit positional AOI
definition, not a learned boundary of eye bags. Chin follows the outer lower
lip and jaw. Cheeks use the observed face outline, eye corners, nasal wings
and mouth corners. This uses each face's actual coordinates rather than
identically shaped or equally sized patches.

All derived assignment is restricted to confident skin. Existing primary
labels, low-confidence pixels and accessories are preserved. Priority is
forehead, under-eyes, chin, then cheeks, giving one integer label per pixel.
Some remaining skin is intentional, particularly around the mouth. Severe
pose or invalid geometry still withholds derived regions.

The review gallery now includes the overlay, opaque label map, raw parser
overlay, raw mesh and confidence map for every face. Uncertainty is grey.
Opaque maps match the legend colours exactly. The original single-channel
integer mask remains the input to gaze analysis; display colours do not affect
fixation or dwell assignment.

Raw parser arrays are saved as *_parser.npz and landmark coordinates as CSV.
Model weights and confidence threshold were not altered. V3 outputs are kept
for comparison; --detailed now selects the separate detailed-v4 folder.

## Validation and remaining limits

Run `python -m tools.validate_detailed_aois` to compare eight cached raw-model
cases with their historical V3 masks. The comparisons use identical raw parser
predictions and meshes so that geometry changes can be isolated. The tool
asserts exact preservation of primary pixels and uncertainty, and verifies
that derived labels only consume former skin/derived skin. When the generated
V4 test set exists, it also checks every generated mask against V3 and writes
generated_set_checks.json. Area differences are reported as coverage changes,
not accuracy gains. The comparison gallery shows before/after/opaque maps.

Completed run: `processed/detailed_v4_validation_20261110_192644/index.html`.
All 20 generated masks passed: zero primary-feature changes, zero uncertainty
changes and zero derived assignments into non-skin features compared with V3.
The eight before/after cases were visually inspected. All 20 V4 records remain
unapproved pending user review of `aois/test/detailed-v4/index.html`.

Visual comparisons show fuller forehead, cheek and chin coverage and eyelid-
following under-eye bands. These remain approximate AOIs. Some contour joins
are piecewise linear, missing primary features are still possible, and no
independent manual reference masks are available. Consequently no anatomical
IoU, Dice accuracy or validated fixation-assignment accuracy is claimed.
Inspect the complete V4 gallery and resolve remaining errors before approval
or dissertation analysis. Participant gaze error still requires separate
measurement; small, adjacent AOIs can exceed the available gaze precision.

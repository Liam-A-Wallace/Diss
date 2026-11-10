# Workspace cleanup

The retired old-fake workflow was removed from Diss: the archived fake images,
DFR60K manifest, dataset-specific builders/curator/smoke test, and the duplicate
stimulus-curation directory. Tests solely for those removed builders were removed.

Current collection uses trials.csv with real photos and OpenAI fakes. The demo
uses the same sources without making image copies. Placeholder cartoons are
isolated under stimuli/placeholders.

Do not delete outside this workspace. The real image files referenced through
../Dataset remain necessary. Preserve data/, AOI masks/reviews, processed results,
models, current images, and historical real-image provenance. Existing recordings
may reference retired image paths and require those original images to reanalyse.
AOI implementation is outside the scope of this cleanup.

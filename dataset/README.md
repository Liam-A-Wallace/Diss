# Dataset records and optional tools

The experiment uses `../trials.csv`: 15 real images from `../stimuli/real/`
and 15 generated images (10 shared-style, 3 high, 2 extra-high). Each pair
contributes one A or B image. High uses pairs 001-003; extra-high uses 004-005.
The demo selects three real images and two shared-style fakes from this list.
No curation/export command is needed for ordinary collection or analysis.

`outputs/picked.csv`, `outputs/review_log.csv` and metadata retain historical
selection records, including the provenance of the real images still in use.
DFR60K mentions in those records describe provenance; they do not activate old
fake trials. Other source manifests and generic tools are retained for reference.

The optional `scripts/review.py` requires an explicit `--csv PATH`.
`scripts/export_to_diss.py` is a historical exporter that can replace trials.csv;
do not run it for the current OpenAI workflow.

# Dataset records and optional tools

The experiment uses `../trials.csv`. Its fake images are OpenAI images in
`../stimuli/openai_shared_style/`. The demo selects from the same trial list.
No curation/export command is needed for ordinary collection or analysis.

`outputs/picked.csv`, `outputs/review_log.csv` and metadata retain historical
selection records, including the provenance of the real images still in use.
DFR60K mentions in those records describe provenance; they do not activate old
fake trials. Other source manifests and generic tools are retained for reference.

The optional `scripts/review.py` requires an explicit `--csv PATH`.
`scripts/export_to_diss.py` is a historical exporter that can replace trials.csv;
do not run it for the current OpenAI workflow.

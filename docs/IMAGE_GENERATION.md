# Running the image generator

## Local GPU models

The same `generate_images.py` command accepts `--model sd3.5-large`,
`--model stylegan2`, or `--model stylegan3`. Each model gets its own default
folder under `stimuli/`; `--out` can override it. `--count` remains the total
number of images, `--limit` caps new images this run, and rerunning skips saved
images. Both local paths require a CUDA GPU and separate dependencies from the
OpenAI API environment. They do not use `OPENAI_API_KEY` or API credits.

For SD3.5 Large, install a CUDA-enabled PyTorch build appropriate to your GPU
using the [PyTorch installer](https://pytorch.org/get-started/locally/) in a
separate Python environment, then run
`python -m pip install -r requirements-sd35.txt`. Accept access to
[Stability AI's gated model](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
and run `hf auth login` for that environment. SD3.5 uses the same A/B prompt
templates and gives each pair the same random seed. The seed and full prompts
are saved in `prompts.json`, along with the fixed 28 steps and guidance 4.5.
For one test pair of a planned 50-image batch, then the rest:

```powershell
python generate_images.py --model sd3.5-large --count 50 --limit 2
python generate_images.py --model sd3.5-large --count 50
```

For StyleGAN2 or StyleGAN3, use a separate environment following the
[NVlabs StyleGAN3 requirements](https://github.com/NVlabs/stylegan3#requirements)
and clone `NVlabs/stylegan3` into the folder next to this project, or pass
`--stylegan-repo` with the checkout's path. The script uses NVIDIA's FFHQ
1024 × 1024 pretrained checkpoints. StyleGAN2 uses the original StyleGAN2 FFHQ
checkpoint distributed with StyleGAN2-ADA; StyleGAN3 uses the T variant.
The network downloads on first use and is cached by NVIDIA's code. For example:

```powershell
python generate_images.py --model stylegan3 --count 50 --limit 2
python generate_images.py --model stylegan3 --count 50
```

StyleGAN uses two independent random seeds per A/B-labelled pair. Its
`prompts.json` explicitly records `prompt: null` and `prompt_applied: false`:
the letters are filenames only and **do not** represent the A/B realism-prompt
comparison. Text-guided StyleGAN is a separate future method; no text extension
is applied here. Keep StyleGAN stimuli as their own model group in analysis.
Both local models use square PNGs named `pair_001_A.png`, `pair_001_B.png`, etc.
Use `--dry-run` to inspect the planned prompts or seeds without loading a model.
Do not run two generators into the same output folder concurrently.


## Generate two test images

Open PowerShell in the repository root (the folder containing
`generate_images.py`) and run:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
.venv-imagegen\Scripts\python.exe generate_images.py --count 2 --out stimuli/openai_test_two
```

Replace `your-api-key` with your OpenAI API key. The key applies to this terminal
session and is not saved by the script. If the image-generation environment
does not exist yet, create it once before running the command above:

```powershell
python -m venv .venv-imagegen
.venv-imagegen\Scripts\python.exe -m pip install openai
```

This generates one A/B pair with identical subject characteristics. Both prompts
specify the same framing, setting, lighting and everyday basic-camera photographic
style. Prompt A additionally requests realistic facial features, skin texture,
age-appropriate lines, skin-tone variation, asymmetry and believable details.
Prompt B leaves those features to the model, with no additional realism guidance.
Both use Sunburst, medium quality and 1024 × 1024 PNG output. Neither asks for
deliberate artefacts; difficulty instructions are disabled.

The comparison tests whether explicit facial-realism guidance changes the output
within a shared photographic style. It does not assume that B will be less
realistic: both prompts already request a photograph.

Images are generated in pair-number order, A then B. Older logs may list B before
A, but generation and dry-run previews use A then B and skip images already saved.

The output folder `stimuli/openai_test_two` will contain:

- `pair_001_A.png`
- `pair_001_B.png`
- `prompts.json` — the completed prompts, seed, settings and progress.

## Preview without generating

This prints the two prompts without using the API or requiring a key:

```powershell
.venv-imagegen\Scripts\python.exe generate_images.py --count 2 --dry-run
```

## Generate the 20-image batch

Generate the first four for a style check:

```powershell
.venv-imagegen\Scripts\python.exe generate_images.py --limit 4
```

Then generate the remaining 16:

```powershell
.venv-imagegen\Scripts\python.exe generate_images.py
```

These images go into `stimuli/generated_openai`. The batch contains 10 A/B pairs
with no deliberate artefacts. `--count` sets the target total in the folder; `--limit` caps
how many new images are generated in the current run.

## Add more images in the same folder

Increase `--count` and keep the same `--out` folder. For example, after generating
the two test images, extend that folder to 10 images:

```powershell
.venv-imagegen\Scripts\python.exe generate_images.py --count 10 --out stimuli/openai_test_two
```

This keeps `pair_001_A.png` and `pair_001_B.png`, adds eight images named
`pair_002_A.png` through `pair_005_B.png`, and extends `prompts.json`.
`--count 10` means 10 images total, not 10 additional images. Rerun the same
command to resume; increase the count again to add more pairs. A target smaller
than the existing plan is rejected. You can still use `--limit` to cap new images
per run, and `--dry-run` previews an extension without saving or calling the API.

Existing prompts and saved images are preserved, including logs from earlier
versions of the script. New pairs use the same templates and settings, with no
difficulty instructions. Characteristics are balanced within each added batch;
extending in several steps can produce different prompts from planning the whole
total in one step. The log records those steps so subsequent runs can resume.

## Change the prompts or settings

Edit `prompt_a.txt` and `prompt_b.txt` at the project root. The characteristic
lists are near the top of `generate_images.py`.
See [prompt_options.txt](../prompt_options.txt) for every current bracket option,
including the allowed setting/lighting combinations and the blank difficulty instruction.

Optional arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `--count` | `20` | Target total in the folder; increase to add pairs; must be even |
| `--limit` | No limit | Maximum new images this run |
| `--seed` | Random for new folders; saved seed on resume | Set explicitly to reproduce characteristic selections |
| `--model` | `gpt-image-2.6-sunburst` | Image model |
| `--quality` | `medium` | Fixed quality for both groups |
| `--out` | `stimuli/generated_openai` | Output folder |
| `--dry-run` | Off | Print prompts without API calls |

Use a new output folder after changing templates, settings or seed. Increasing
the total count can use the same folder.
The seed reproduces the prompt selections, not the generated image pixels.

The shared-style revision changes both templates. Images and logs made with the
earlier templates retain their original prompts. Start the revised comparison in
one new folder, then increase its total normally:

```powershell
.venv-imagegen\Scripts\python.exe generate_images.py --count 40 --out stimuli/openai_shared_style
```

A process already running continues with the prompts it loaded at startup.

## Resume a run

Run the same command again to skip images already marked as saved. Requests
rejected with HTTP 429 remain pending, with the error recorded in `prompts.json`.
For a "no credits remaining" error, add API credits in your OpenAI billing settings
before rerunning the same command. The script does not automatically retry.

For other failures or interrupted requests, inspect the entry in `prompts.json`
before retrying. The script stops rather than repeating a request that may already
have been charged. Avoid running two generators into the same folder at once.

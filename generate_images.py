"""Generate A/B portrait pairs. See docs/IMAGE_GENERATION.md for usage."""
import argparse
import base64
import json
import os
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent

# Options shared by both groups. The age is sampled within the selected band.
AGE_BANDS = [(18, 29), (30, 44), (45, 59), (60, 80)]
GENDERS = ["person with a masculine presentation", "person with a feminine presentation"]
ETHNICITIES = ["White", "East Asian", "South Asian", "Black", "Latin American", "Middle Eastern", "mixed ethnicity"]
EXPRESSIONS = ["a neutral expression", "a relaxed expression", "a closed-mouth smile", "a smile showing teeth"]
HAIR = ["short straight black hair", "short wavy brown hair", "short tightly curled black hair",
        "shoulder-length curly brown hair", "long straight brown hair tucked behind the ears",
        "shoulder-length wavy blonde hair", "short auburn hair", "neat black braids pulled back",
        "long black hair tied back loosely"]
# Seven of every ten accessory slots are 'none'.
ACCESSORIES = ["no visible accessories"] * 7 + [
    "clear-lens glasses that leave the eyes visible", "small stud earrings",
    "a casual hat worn back so the forehead and eyes remain visible"]
# Only pair indoor lighting with indoor settings, and daylight with outdoor ones.
SETTINGS = {
    "a garden": ["overcast daylight", "direct sunlight"],
    "a residential street": ["overcast daylight", "direct sunlight"],
    "a living room": ["indoor ambient light", "indoor camera flash"],
    "an indoor gathering": ["indoor ambient light", "indoor camera flash"],
    "a café": ["indoor ambient light", "indoor camera flash"],
    "an office": ["indoor ambient light", "indoor camera flash"],
    "an outdoor area beside a brick wall": ["overcast daylight", "direct sunlight"],
}


def balanced(options, count, rng):
    """Deal shuffled copies of the options until count slots are filled."""
    result = []
    while len(result) < count:
        cycle = list(options)
        rng.shuffle(cycle)
        result.extend(cycle[:count - len(result)])
    rng.shuffle(result)
    return result


def make_prompts(count, seed, start_pair=0):
    """Choose characteristics once per pair, then fill both templates."""
    rng = random.Random(seed + start_pair)
    pairs = count // 2
    templates = {g: (ROOT / f"prompt_{g.lower()}.txt").read_text(encoding="utf-8") for g in "AB"}
    ages, genders, ethnicities, expressions, hair, accessories, settings = [
        balanced(options, pairs, rng) for options in
        (AGE_BANDS, GENDERS, ETHNICITIES, EXPRESSIONS, HAIR, ACCESSORIES, list(SETTINGS))]
    lighting = [None] * pairs
    # Spread the available lighting choices across repeats of each setting.
    for setting, options in SETTINGS.items():
        indices = [i for i, s in enumerate(settings) if s == setting]
        for i, light in zip(indices, balanced(options, len(indices), rng)):
            lighting[i] = light
    rows = []
    for i in range(pairs):
        values = {"AGE": str(rng.randint(*ages[i])), "GENDER PRESENTATION": genders[i],
                  "ETHNICITY": ethnicities[i], "EXPRESSION": expressions[i],
                  "HAIR DESCRIPTION": hair[i], "ACCESSORIES": accessories[i],
                  "SETTING": settings[i], "LIGHTING CONDITION": lighting[i],
                  "OPTIONAL DIFFICULTY INSTRUCTION": ""}
        # No difficulty block for this batch. A and B receive the same values.
        groups = list("AB")
        rng.shuffle(groups)
        for group in groups:
            def substitute(match):
                if match[1] not in values:
                    raise ValueError(f"Unknown placeholder: {match[0]}")
                return values[match[1]]
            prompt = re.sub(r"\[([^]]+)\]", substitute, templates[group]).strip()
            rows.append(dict(id=f"pair_{start_pair + i + 1:03d}_{group}", prompt=prompt))
    return rows


def plan_log(path, count, settings):
    """Resume the saved plan, appending prompts when the target increases."""
    if not path.exists():
        return dict(settings=settings, prompt_batches=[count], images=[
            dict(r, status="pending") for r in make_prompts(count, settings["seed"])])

    log = json.loads(path.read_text(encoding="utf-8"))
    if log["settings"] != settings:
        raise ValueError("Settings changed. Use the original settings or a new --out folder.")
    # Older logs contain a single batch and need no manual conversion.
    batches = log.get("prompt_batches", [len(log["images"])])
    if (not batches or any(type(n) is not int or n < 2 or n % 2 for n in batches)
            or sum(batches) != len(log["images"])):
        raise ValueError("Invalid prompt batch history in prompts.json.")
    expected = []
    for batch_count in batches:
        expected.extend(make_prompts(batch_count, settings["seed"], len(expected) // 2))
    if [{"id": r["id"], "prompt": r["prompt"]} for r in log["images"]] != expected:
        raise ValueError("Prompts changed. Restore the original templates or use a new --out folder.")
    previous_count = len(log["images"])
    if count < previous_count:
        raise ValueError(f"This folder already plans {previous_count} images. "
                         f"Use --count {previous_count} to resume, or a larger total to add pairs.")
    if count > previous_count:
        added_count = count - previous_count
        new_rows = make_prompts(added_count, settings["seed"], previous_count // 2)
        log["images"].extend(dict(r, status="pending") for r in new_rows)
        batches = [*batches, added_count]
    log["prompt_batches"] = batches
    return log


def save_log(path, log):
    # Replace the log only after the new copy has been written in full.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=20,
                        help="Target total images in the folder; increase to add pairs (default: 20)")
    parser.add_argument("--limit", type=int, help="Maximum new images this run; use 4 for the pilot")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--model", default="gpt-image-2.6-sunburst",
                        help="OpenAI model, sd3.5-large, stylegan2, or stylegan3")
    parser.add_argument("--quality", choices=["low", "medium", "high", "xhigh", "max"], default="medium")
    parser.add_argument("--out", type=Path,
                        help="Output folder (defaults to a separate folder for each model)")
    parser.add_argument("--stylegan-repo", type=Path, default=ROOT.parent / "stylegan3",
                        help="Local NVlabs/stylegan3 checkout used by StyleGAN models")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.count < 2 or args.count % 2 or (args.limit is not None and args.limit < 1):
        parser.error("--count must be even and at least 2; --limit must be positive")
    try:
        if args.model in ("sd3.5-large", "stylegan2", "stylegan3"):
            from local_generation import run_local
            if args.out is None:
                args.out = ROOT / "stimuli" / args.model.replace(".", "_")
            return run_local(args)
        if args.out is None:
            args.out = ROOT / "stimuli/generated_openai"
        settings = dict(seed=args.seed, model=args.model, quality=args.quality, size="1024x1024")
        path = args.out / "prompts.json"
        log = plan_log(path, args.count, settings)
        # Keep the original prompt plan reproducible, but always request A then B.
        ordered_rows = sorted(log["images"], key=lambda row: (
            int(row["id"].split("_")[1]), row["id"].rsplit("_", 1)[1]))
        if args.dry_run:
            print(f"{args.count} images; {settings}; no deliberate artefacts")
            for row in ordered_rows:
                print(f"\n{row['id']} ({row['status']})\n{row['prompt']}")
            return 0
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY in this terminal first.")
        from openai import OpenAI, RateLimitError
        args.out.mkdir(parents=True, exist_ok=True)
        for row in log["images"]:
            image = args.out / f"{row['id']}.png"
            if row["status"] == "saved":
                if not image.is_file() or image.stat().st_size == 0:
                    raise ValueError(f"Saved image missing: {image}")
            elif image.exists() or row["status"] != "pending":
                raise ValueError(f"Inspect {row['id']} in prompts.json before retrying; the previous request may have been charged.")
        save_log(path, log)
        generated = 0
        # Disable automatic retries: a timed-out request may already be charged.
        with OpenAI(timeout=600, max_retries=0) as client:
            for row in ordered_rows:
                if row["status"] == "saved":
                    continue
                if args.limit is not None and generated >= args.limit:
                    break
                row["status"] = "started"
                # Leave interrupted requests marked so a rerun won't repeat them.
                save_log(path, log)
                print(f"Generating {row['id']}...", flush=True)
                try:
                    response = client.images.generate(model=args.model, prompt=row["prompt"],
                        size="1024x1024", quality=args.quality, output_format="png", n=1)
                except RateLimitError as exc:
                    # A 429 rejects the request; it is safe to retry on a later run.
                    # Timeouts and other ambiguous failures must remain "started".
                    row["status"] = "pending"
                    row["last_error"] = dict(status_code=429, code=exc.code, message=str(exc))
                    save_log(path, log)
                    raise
                image_bytes = base64.b64decode(response.data[0].b64_json, validate=True)
                if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("API returned no valid PNG image.")
                image = args.out / f"{row['id']}.png"
                with image.open("xb") as stream:
                    stream.write(image_bytes)
                row["status"] = "saved"
                save_log(path, log)
                generated += 1
        print(f"Saved {generated} new images to {args.out}")
        return 0
    except KeyboardInterrupt:
        print("Stopped. Completed images are saved; check the log before retrying an interrupted request.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

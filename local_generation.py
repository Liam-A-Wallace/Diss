"""Local GPU backends for generate_images.py; imports heavy libraries only on use."""
import io
import json
from pathlib import Path
import sys

STYLEGAN_NETWORKS = {
    "stylegan2": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
    "stylegan3": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl",
}
SD_MODEL = "stabilityai/stable-diffusion-3.5-large"
PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def stylegan_rows(count, seed):
    """A/B are two independent seed samples, not prompt treatments."""
    return [dict(id=f"pair_{i // 2 + 1:03d}_{'AB'[i % 2]}",
                 seed=seed + i, prompt=None, prompt_applied=False)
            for i in range(count)]


def plan_stylegan(path, count, settings):
    rows = stylegan_rows(count, settings["seed"])
    if not path.exists():
        return dict(settings=settings, images=[dict(r, status="pending") for r in rows])
    log = json.loads(path.read_text(encoding="utf-8"))
    if log["settings"] != settings:
        raise ValueError("StyleGAN settings changed. Use a new --out folder.")
    prior = log["images"]
    if len(prior) > count:
        raise ValueError(f"This folder already plans {len(prior)} images; use a larger --count.")
    if [{k: r[k] for k in ("id", "seed", "prompt", "prompt_applied")} for r in prior] != rows[:len(prior)]:
        raise ValueError("StyleGAN seed plan changed. Use a new --out folder.")
    log["images"].extend(dict(r, status="pending") for r in rows[len(prior):])
    return log


class SD35Generator:
    def __init__(self):
        try:
            import torch
            from diffusers import StableDiffusion3Pipeline
        except ImportError as exc:
            raise RuntimeError("Install torch, diffusers, transformers and accelerate in a separate GPU environment.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("SD3.5 Large needs an available CUDA GPU for this workflow.")
        try:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(SD_MODEL, torch_dtype=torch.bfloat16)
            self.pipe.enable_model_cpu_offload()
        except OSError as exc:
            raise RuntimeError("Could not load gated SD3.5 Large weights. Accept access on Hugging Face and log in with `hf auth login`.") from exc
        self.torch = torch

    def generate(self, row):
        generator = self.torch.Generator(device="cpu").manual_seed(row["seed"])
        image = self.pipe(prompt=row["prompt"], generator=generator, height=1024,
                          width=1024, num_inference_steps=28, guidance_scale=4.5).images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


class StyleGANGenerator:
    def __init__(self, model, repo):
        repo = Path(repo).resolve()
        if not (repo / "legacy.py").is_file() or not (repo / "dnnlib").is_dir():
            raise RuntimeError(f"Clone NVlabs/stylegan3 at {repo} before generating StyleGAN images.")
        try:
            import torch
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install torch, numpy and Pillow in a separate StyleGAN GPU environment.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("StyleGAN needs an available CUDA GPU for this workflow.")
        sys.path.insert(0, str(repo))
        import dnnlib
        import legacy
        with dnnlib.util.open_url(STYLEGAN_NETWORKS[model]) as stream:
            self.net = legacy.load_network_pkl(stream)["G_ema"].to("cuda")
        self.torch, self.np, self.Image = torch, np, Image

    def generate(self, row):
        torch, np = self.torch, self.np
        z = torch.from_numpy(np.random.RandomState(row["seed"]).randn(1, self.net.z_dim)).to("cuda")
        label = torch.zeros([1, self.net.c_dim], device="cuda")
        with torch.no_grad():
            image = self.net(z, label, truncation_psi=1, noise_mode="const")
        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        buffer = io.BytesIO()
        self.Image.fromarray(image[0].cpu().numpy(), "RGB").save(buffer, format="PNG")
        return buffer.getvalue()


def run_local(args):
    from generate_images import make_prompts, plan_log, save_log

    is_sd = args.model == "sd3.5-large"
    if is_sd:
        settings = dict(seed=args.seed, model=args.model, checkpoint=SD_MODEL,
                        size="1024x1024", steps=28, guidance_scale=4.5)
    else:
        settings = dict(seed=args.seed, model=args.model,
                        checkpoint=STYLEGAN_NETWORKS[args.model], size="1024x1024",
                        truncation=1, noise_mode="const", prompt_applied=False)
    path = args.out / "prompts.json"
    if is_sd:
        # Preserve existing prompt generation, including A/B pairing and extensions.
        # plan_log expects the same settings shape it writes to its own log.
        log = plan_log(path, args.count, settings)
        for row in log["images"]:
            row.setdefault("seed", args.seed + 2 * (int(row["id"].split("_")[1]) - 1))
            row.setdefault("prompt_applied", True)
    else:
        log = plan_stylegan(path, args.count, settings)
    ordered = sorted(log["images"], key=lambda row: row["id"])
    if args.dry_run:
        print(f"{args.count} images; {settings}")
        for row in ordered:
            print(f"\n{row['id']} ({row['status']}); seed={row['seed']}; prompt_applied={row['prompt_applied']}")
            if is_sd:
                print(row["prompt"])
        return 0
    if args.quality != "medium":
        raise ValueError("--quality applies only to the OpenAI model; omit it for local generation.")
    for row in ordered:
        image = args.out / f"{row['id']}.png"
        if row["status"] == "saved":
            if not image.is_file() or image.stat().st_size == 0:
                raise ValueError(f"Saved image missing: {image}")
        elif image.exists() or row["status"] != "pending":
            raise ValueError(f"Inspect {row['id']} in prompts.json before retrying.")
    if all(r["status"] == "saved" for r in ordered):
        print(f"Saved 0 new images to {args.out}")
        return 0
    engine = SD35Generator() if is_sd else StyleGANGenerator(args.model, args.stylegan_repo)
    args.out.mkdir(parents=True, exist_ok=True)
    save_log(path, log)
    generated = 0
    for row in ordered:
        if row["status"] == "saved":
            continue
        if args.limit is not None and generated >= args.limit:
            break
        row["status"] = "started"
        save_log(path, log)
        print(f"Generating {row['id']}...", flush=True)
        try:
            image_bytes = engine.generate(row)
            if not image_bytes.startswith(PNG_HEADER):
                raise ValueError("Generator did not return a PNG image.")
            with (args.out / f"{row['id']}.png").open("xb") as stream:
                stream.write(image_bytes)
        except BaseException:
            # Local generation can be safely retried unless a file was written.
            if not (args.out / f"{row['id']}.png").exists():
                row["status"] = "pending"
                save_log(path, log)
            raise
        row["status"] = "saved"
        save_log(path, log)
        generated += 1
    print(f"Saved {generated} new images to {args.out}")
    return 0

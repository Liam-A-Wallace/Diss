"""Compatibility command for the shared dataset manifest builder."""
import importlib.util
from pathlib import Path


def _load_builder():
    folder = Path(__file__).resolve().parent
    source = folder.parent / "dataset/scripts/build_60k_manifest.py"
    spec = importlib.util.spec_from_file_location("diss_manifest_builder", source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def find_dataset_root():
    """Keep the discovery entry point used by enrich's no-CSV fallback."""
    folder = Path(__file__).resolve().parent
    return _load_builder().find_dataset_root(preferred_roots=(folder,))


def main(argv=None):
    folder = Path(__file__).resolve().parent
    builder = _load_builder()
    return builder.main(argv, default_out=folder / "DFR60K_manifest.csv", default_sample=0,
                        preferred_roots=(folder,))


if __name__ == "__main__":
    raise SystemExit(main())

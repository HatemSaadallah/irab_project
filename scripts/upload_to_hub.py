"""Upload a trained model to the Hugging Face Hub.

Usage:
    export HF_TOKEN=hf_XXX
    python scripts/upload_to_hub.py --checkpoint runs/medium/best.pt --repo your-user/irab-tashkeel-model
"""

import argparse
import json
import os
from pathlib import Path

import torch

from irab_tashkeel.models.full_model import FullModel


README_TEMPLATE = """---
library_name: pytorch
tags:
  - arabic
  - diacritization
  - tashkeel
  - irab
  - nlp
language: ar
license: mit
---

# I'rab + Tashkeel Model

Multi-task model for Arabic:
- Diacritization (tashkīl)
- I'rab role tagging (per-word)
- Orthographic error detection

## Usage

```python
from irab_tashkeel.inference.predictor import Predictor
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download("{repo}", "model.pt")
predictor = Predictor.from_checkpoint(ckpt)
result = predictor.predict("ذهب الطالب إلى المدرسة")
print(result.diacritized)
for word in result.words:
    print(f"{{word['surface']:<15}} {{word['role_en']}}")
```

## Model details

- Parameters: {n_params:.1f}M
- Architecture: character-level Transformer encoder + 3 task heads
- Training data: QAC, Tashkeela (Sadeed cleaned), I3rab, synthetic errors
- See [GitHub repo]({repo_url}) for training details.

## Limitations

- I'rab labels are coarse (11 classes) — fine-grained roles like fāʿil vs mubtadaʾ are collapsed to N_marfu
- Trained predominantly on Classical Arabic (Quran); MSA performance may lag
- Does not handle kāna/inna sisters, relative clauses, or complex coordination
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--repo", type=str, required=True, help="e.g. your-user/irab-tashkeel-model")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", type=str, default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--staging-dir", type=str, default="/tmp/hf_upload")
    args = parser.parse_args()

    from huggingface_hub import HfApi, login

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("No HF token. Set $HF_TOKEN or pass --token.")
    login(token=token)

    staging = Path(args.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    # Load to verify + get config
    print(f"Loading {args.checkpoint} …")
    model = FullModel.load(args.checkpoint, map_location="cpu")
    n_params = model.n_params() / 1e6
    print(f"  Params: {n_params:.1f}M")

    # Save model to staging (stable filename)
    model_path = staging / "model.pt"
    model.save(model_path)

    # Standalone config.json for easy inspection
    config_path = staging / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model.config.to_dict(), f, indent=2)

    # README
    readme_path = staging / "README.md"
    readme = README_TEMPLATE.format(
        repo=args.repo,
        repo_url=f"https://huggingface.co/{args.repo}",
        n_params=n_params,
    )
    readme_path.write_text(readme, encoding="utf-8")

    # Push
    api = HfApi()
    api.create_repo(args.repo, exist_ok=True, private=args.private)
    print(f"Uploading {len(list(staging.iterdir()))} files to {args.repo} …")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"✓ https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()

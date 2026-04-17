# Deployment

## Three deployment targets

### Local (development)
```bash
pip install -e .
export MODEL_CKPT=runs/medium/best.pt
streamlit run app/app.py
```
Open http://localhost:8501.

### Hugging Face Spaces (public demo)

**Step 1: Upload model to HF Hub**
```bash
python scripts/upload_to_hub.py \
    --checkpoint runs/medium/best.pt \
    --repo your-handle/irab-tashkeel-model \
    --private false
```

Produces:
```
your-handle/irab-tashkeel-model/
  config.json
  model.pt
  README.md
```

**Step 2: Create the Space**

On https://huggingface.co/new-space:
- Name: `irab-demo`
- SDK: Streamlit (Docker-backed)
- Hardware: CPU basic (free) or T4 small ($0.40/hr)

**Step 3: Push the app**
```bash
cd app/
git init
git remote add space https://huggingface.co/spaces/your-handle/irab-demo
git add app.py requirements.txt README.md
git commit -m "Initial deploy"

# Set HF token (get from https://huggingface.co/settings/tokens)
git push https://your-handle:hf_XXX@huggingface.co/spaces/your-handle/irab-demo main
```

Also set the env var `MODEL_REPO=your-handle/irab-tashkeel-model` in the Space settings.

**Build time**: ~5 minutes. Space will be live at https://huggingface.co/spaces/your-handle/irab-demo.

### Docker (self-hosted)

For institutional deployment where HF Spaces isn't an option:

```bash
docker build -t irab-tashkeel -f app/Dockerfile .
docker run -p 8501:8501 -e MODEL_CKPT=/app/model/best.pt irab-tashkeel
```

Dockerfile is in `app/Dockerfile`. ~2GB image (PyTorch + deps + model).

## HF Space configuration

The `app/README.md` YAML frontmatter tells HF how to run:
```yaml
---
title: I'rāb Tashkīl Demo
emoji: 📖
colorFrom: purple
colorTo: teal
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: mit
---
```

## Performance tuning for CPU inference

On HF Spaces free tier (CPU only), a 60M-param model takes ~2-3s per sentence. Improvements:

1. **Use `torch.compile`** (PyTorch 2.0+): ~1.5-2× speedup:
   ```python
   model = torch.compile(model, mode="reduce-overhead")
   ```

2. **ONNX export**:
   ```bash
   python scripts/export_onnx.py --checkpoint runs/medium/best.pt --output model.onnx
   ```
   Then load via `onnxruntime` in the app. ~2-3× speedup on CPU.

3. **Quantization** (int8):
   ```python
   from torch.quantization import quantize_dynamic
   model_q = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```
   ~2× speedup, ~0.3-0.5% DER degradation.

4. **Smaller model**: train a 20M-param variant specifically for deployment. The DER goes up ~1-2% but inference is 3× faster.

## Monitoring

HF Spaces gives you:
- Build logs (useful when deployment fails)
- Runtime logs (Streamlit stderr)
- Usage stats (total visitors, sessions)

For custom monitoring, instrument `app.py`:
```python
import time
t0 = time.time()
result = predictor.predict(text)
latency_ms = (time.time() - t0) * 1000
# log to file or send to a telemetry backend
```

## Auth / rate limiting

HF Spaces default: public, no auth, no rate limit. For private deployment:
- HF Spaces Pro: password-protect the Space
- Own Docker deployment: put nginx in front with basic auth + rate limit

## Model versioning

When you retrain:
1. Upload new weights: `python scripts/upload_to_hub.py --repo your-handle/irab-tashkeel-model --tag v2`
2. Space auto-restarts when you push a new commit to the Space repo
3. Users see the new version on next request

Keep old versions accessible via git tags on the Hub model repo.

## Limitations of the deployed model

For the paper, make these clear on the deployed Space (add a section to `app/README.md`):

> **What this demo is**:
> - A research prototype for explainable Arabic diacritization
> - Trained on Classical Arabic (Quran, Shamela) + limited MSA
> - Provides approximate i'rāb labels, not ground-truth parsing
>
> **What this demo is NOT**:
> - Not a replacement for a trained linguist's annotation
> - Not tuned for dialectal Arabic (Egyptian, Levantine, Gulf)
> - Not guaranteed to produce 100% correct diacritization — especially for fine-grained case distinctions in complex sentences

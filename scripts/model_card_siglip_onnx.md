---
license: apache-2.0
base_model: timm/ViT-SO400M-14-SigLIP-384
tags:
  - onnx
  - int8
  - ppe
  - safety
  - zero-shot-classification
  - construction
language:
  - en
pipeline_tag: zero-shot-image-classification
---

# SigLIP SO400M — SentinelVision ONNX INT8

Dynamic INT8 quantization of the [SigLIP SO400M](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384) image encoder, optimized for CPU inference in the [SentinelVision](https://github.com/nhatminh-115/SentinelVision) pipeline.

Only the **image tower** is exported. Text embeddings are pre-computed at startup from the original PyTorch weights and cached in memory.

## Model details

| Property | Value |
|---|---|
| Base model | `ViT-SO400M-14-SigLIP` (`webli` pretrained, open_clip) |
| Input size | 224 × 224 |
| Embedding dim | 1152 |
| Quantization | Dynamic INT8 (MatMul/Gemm), opset 17 |
| FP32 size | 1632 MB |
| INT8 size | 411 MB |
| Cosine similarity vs FP32 | 0.994 (random image) |
| Decision flip rate | 5.3% (19 real PPE crops) |

## Intended use

Zero-shot PPE classification in the [SentinelVision](https://github.com/nhatminh-115/SentinelVision) system:

- **Hardhat detection**: head region crop → cosine similarity against positive/negative text prompts
- **Safety vest detection**: torso region crop → cosine similarity against positive/negative text prompts

Not intended as a standalone model — requires the text embeddings and `logit_scale` from the full pipeline.

## Usage

```python
import onnxruntime as ort
import numpy as np
from huggingface_hub import hf_hub_download

path = hf_hub_download("Nhatminh1234/siglip-so400m-ppe-int8", "siglip_image_encoder.onnx")
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

# Input: preprocessed image batch (B, 3, 224, 224) float32, normalized per open_clip preprocess
pixel_values = np.random.randn(1, 3, 224, 224).astype(np.float32)
image_features = session.run(None, {"pixel_values": pixel_values})[0]  # (B, 1152)

# Normalize before cosine similarity
image_features /= np.linalg.norm(image_features, axis=-1, keepdims=True)
```

## Performance notes

Dynamic INT8 quantizes only MatMul/Gemm layers (~80% of ViT compute). LayerNorm, softmax, and attention scores remain FP32. Expected cosine similarity vs FP32: 0.990–0.998 for large ViT models.

The `logit_scale` for this model is **111.54** (i.e., `exp(4.714)`).

## Regenerating

```bash
git clone https://github.com/nhatminh-115/SentinelVision
cd SentinelVision
python scripts/export_siglip_onnx.py --upload
```

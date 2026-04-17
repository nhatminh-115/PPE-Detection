"""
Export SigLIP SO400M image encoder to ONNX and apply dynamic INT8 PTQ.

Usage:
    python scripts/export_siglip_onnx.py
    python scripts/export_siglip_onnx.py --fp32-only
    python scripts/export_siglip_onnx.py --output model_cache/siglip_image_encoder.onnx
"""

import argparse
import os
import sys
from pathlib import Path

# Must be set before any torch/open_clip import to prevent Windows CUDA init crash.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import open_clip
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import SIGLIP_MODEL_NAME, SIGLIP_PRETRAINED, SIGLIP_ONNX_PATH, SIGLIP_ONNX_HF_REPO


class _SigLIPVisualEncoder(torch.nn.Module):
    """Thin wrapper that exports only the image tower (no logit_scale, no text)."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.visual = model.visual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def _get_image_size(model) -> int:
    size = getattr(model.visual, "image_size", 224)
    if isinstance(size, (tuple, list)):
        return int(size[0])
    return int(size)


def export_fp32(model: torch.nn.Module, image_size: int, output_path: str) -> None:
    encoder = _SigLIPVisualEncoder(model).eval().cpu()
    dummy = torch.zeros(1, 3, image_size, image_size)

    torch.onnx.export(
        encoder,
        dummy,
        output_path,
        dynamo=False,
        opset_version=17,
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_features": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  FP32 ONNX -> {output_path} ({size_mb:.0f} MB)")


def quantize_int8(fp32_path: str, int8_path: str) -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        fp32_path,
        int8_path,
        weight_type=QuantType.QInt8,
    )
    size_mb = os.path.getsize(int8_path) / 1024 / 1024
    print(f"  INT8 ONNX -> {int8_path} ({size_mb:.0f} MB)")


def verify_parity(
    model: torch.nn.Module,
    ort_session,
    preprocess,
    image_size: int,
) -> float:
    rng = np.random.RandomState(42)
    dummy_np = rng.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    pil = Image.fromarray(dummy_np)
    tensor = preprocess(pil).unsqueeze(0)

    with torch.no_grad():
        pt_feat = model.visual(tensor).cpu().numpy()

    input_name = ort_session.get_inputs()[0].name
    ort_feat = ort_session.run(None, {input_name: tensor.numpy().astype(np.float32)})[0]

    pt_norm = pt_feat / np.linalg.norm(pt_feat, axis=-1, keepdims=True)
    ort_norm = ort_feat / np.linalg.norm(ort_feat, axis=-1, keepdims=True)
    cosine_sim = float((pt_norm * ort_norm).sum(axis=-1).mean())

    # Dynamic INT8 typically yields 0.990-0.998; FP32/FP16 should be >0.9999.
    threshold = 0.990
    status = "OK" if cosine_sim >= threshold else "WARNING"
    print(f"  [{status}] cosine similarity PyTorch vs ONNX: {cosine_sim:.6f}")
    return cosine_sim


def upload_to_hf(onnx_path: str, repo_id: str) -> None:
    from huggingface_hub import HfApi
    token = os.getenv("HF_TOKEN")
    if not token:
        print("  WARNING: HF_TOKEN not set. Upload may fail for private repos.")
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    api.upload_file(
        path_or_fileobj=onnx_path,
        path_in_repo="siglip_image_encoder.onnx",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"  Uploaded ONNX -> https://huggingface.co/{repo_id}")

    card_path = Path(__file__).parent / "model_card_siglip_onnx.md"
    if card_path.exists():
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        print(f"  Uploaded model card")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-only", action="store_true", help="Export FP32 only, skip INT8")
    parser.add_argument("--output", default=None, help="INT8 output path (default: from config)")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--upload", action="store_true", help="Upload INT8 ONNX to HF Hub after export")
    parser.add_argument("--hf-repo", default=None, help=f"HF repo ID (default: {SIGLIP_ONNX_HF_REPO})")
    args = parser.parse_args()

    int8_path = args.output or SIGLIP_ONNX_PATH
    output_dir = os.path.dirname(int8_path) or "model_cache"
    os.makedirs(output_dir, exist_ok=True)
    repo_id = args.hf_repo or SIGLIP_ONNX_HF_REPO

    # Skip model loading entirely if the ONNX file already exists.
    if os.path.exists(int8_path):
        print(f"ONNX file already exists: {int8_path}")
        if args.upload:
            print(f"Uploading to HF Hub ({repo_id})...")
            upload_to_hf(int8_path, repo_id)
        print("\nDone.")
        return

    fp32_path = os.path.join(output_dir, "siglip_image_encoder_fp32.onnx")

    print(f"Loading {SIGLIP_MODEL_NAME} ({SIGLIP_PRETRAINED})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        SIGLIP_MODEL_NAME, pretrained=SIGLIP_PRETRAINED,
    )
    model.eval()

    image_size = _get_image_size(model)
    logit_scale = float(model.logit_scale.exp().item())
    print(f"  image_size: {image_size}x{image_size}, logit_scale: {logit_scale:.4f}")

    print("Exporting FP32 ONNX...")
    export_fp32(model, image_size, fp32_path)

    target_path = fp32_path
    if not args.fp32_only:
        print("Quantizing to dynamic INT8...")
        quantize_int8(fp32_path, int8_path)
        target_path = int8_path
        print(f"  Removing FP32 intermediate: {fp32_path}")
        os.remove(fp32_path)

    if not args.skip_parity:
        import onnxruntime as ort
        print("Running parity check...")
        session = ort.InferenceSession(target_path, providers=["CPUExecutionProvider"])
        verify_parity(model, session, preprocess, image_size)

    if args.upload and not args.fp32_only:
        print(f"Uploading to HF Hub ({repo_id})...")
        upload_to_hf(target_path, repo_id)

    print("\nDone.")
    print(f"Set in .env to enable:")
    print(f"  SIGLIP_ONNX_ENABLED=1")
    print(f"  SIGLIP_ONNX_PATH={target_path}")


if __name__ == "__main__":
    main()

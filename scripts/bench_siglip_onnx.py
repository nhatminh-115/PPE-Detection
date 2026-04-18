"""
Benchmark SigLIP image encoder latency/throughput on CPU.

Default mode benchmarks ONNX INT8. Optional compare mode benchmarks both
PyTorch (original) and ONNX INT8 on the same input batches.

Examples:
    python scripts/bench_siglip_onnx.py
    python scripts/bench_siglip_onnx.py --batch-sizes 1 2 4 8 --runs 120
    python scripts/bench_siglip_onnx.py --use-crops --crops-dir violation_crops
    python scripts/bench_siglip_onnx.py --compare-pytorch --batch-sizes 1 2 4
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Prevent Windows CUDA init issues when importing open_clip/torch for CPU benchmark.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2
import numpy as np
import onnxruntime as ort


def _build_session(onnx_path: str, intra_threads: int, inter_threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_threads > 0:
        so.intra_op_num_threads = intra_threads
    if inter_threads > 0:
        so.inter_op_num_threads = inter_threads
    return ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])


def _load_crop_bank(crops_dir: str, image_size: int, max_images: int) -> np.ndarray:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [
        p for p in Path(crops_dir).iterdir()
        if p.suffix.lower() in image_exts
    ]
    if not paths:
        raise ValueError(f"No images found in crops dir: {crops_dir}")

    bank: list[np.ndarray] = []
    for path in sorted(paths)[:max_images]:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        bank.append(arr)

    if not bank:
        raise ValueError(f"Could not decode images from: {crops_dir}")
    return np.stack(bank, axis=0).astype(np.float32)


def _make_batch(batch_size: int, image_size: int, crop_bank: np.ndarray | None, seed: int) -> np.ndarray:
    if crop_bank is not None:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(crop_bank), size=batch_size)
        return crop_bank[idx]
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(batch_size, 3, image_size, image_size), dtype=np.float32)


def _percentile(data: list[float], q: float) -> float:
    return float(np.percentile(np.array(data, dtype=np.float64), q))


def _bench_onnx_batch(
    session: ort.InferenceSession,
    input_name: str,
    batch_size: int,
    image_size: int,
    warmup: int,
    runs: int,
    crop_bank: np.ndarray | None,
    seed: int,
) -> dict[str, float]:
    batch = _make_batch(batch_size=batch_size, image_size=image_size, crop_bank=crop_bank, seed=seed)

    for _ in range(warmup):
        session.run(None, {input_name: batch})

    lat_ms: list[float] = []
    total_images = 0
    t0 = time.perf_counter()
    for _ in range(runs):
        t1 = time.perf_counter()
        session.run(None, {input_name: batch})
        t2 = time.perf_counter()
        lat_ms.append((t2 - t1) * 1000.0)
        total_images += batch_size
    t_total = time.perf_counter() - t0

    return {
        "batch": float(batch_size),
        "lat_mean_ms": float(np.mean(lat_ms)),
        "lat_p50_ms": _percentile(lat_ms, 50),
        "lat_p95_ms": _percentile(lat_ms, 95),
        "lat_p99_ms": _percentile(lat_ms, 99),
        "throughput_img_s": float(total_images / max(t_total, 1e-9)),
        "per_image_ms": float(np.mean(lat_ms) / max(batch_size, 1)),
    }


def _load_pytorch_visual(siglip_model: str, siglip_pretrained: str):
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(siglip_model, pretrained=siglip_pretrained)
    model = model.eval().cpu()
    return model.visual


def _bench_torch_batch(
    visual_model,
    batch_size: int,
    image_size: int,
    warmup: int,
    runs: int,
    crop_bank: np.ndarray | None,
    seed: int,
) -> dict[str, float]:
    import torch

    batch_np = _make_batch(batch_size=batch_size, image_size=image_size, crop_bank=crop_bank, seed=seed)
    batch = torch.from_numpy(batch_np).to("cpu")

    with torch.no_grad():
        for _ in range(warmup):
            _ = visual_model(batch)

    lat_ms: list[float] = []
    total_images = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            t1 = time.perf_counter()
            _ = visual_model(batch)
            t2 = time.perf_counter()
            lat_ms.append((t2 - t1) * 1000.0)
            total_images += batch_size
    t_total = time.perf_counter() - t0

    return {
        "batch": float(batch_size),
        "lat_mean_ms": float(np.mean(lat_ms)),
        "lat_p50_ms": _percentile(lat_ms, 50),
        "lat_p95_ms": _percentile(lat_ms, 95),
        "lat_p99_ms": _percentile(lat_ms, 99),
        "throughput_img_s": float(total_images / max(t_total, 1e-9)),
        "per_image_ms": float(np.mean(lat_ms) / max(batch_size, 1)),
    }


def _print_table(title: str, rows: list[dict[str, float]]) -> None:
    print(f"\n{title}")
    print(
        f"{'batch':>6}  {'lat_mean':>9}  {'p50':>9}  {'p95':>9}  {'p99':>9}  "
        f"{'per_img':>9}  {'throughput':>11}"
    )
    for r in rows:
        print(
            f"{int(r['batch']):>6}  "
            f"{r['lat_mean_ms']:>9.2f}  "
            f"{r['lat_p50_ms']:>9.2f}  "
            f"{r['lat_p95_ms']:>9.2f}  "
            f"{r['lat_p99_ms']:>9.2f}  "
            f"{r['per_image_ms']:>9.2f}  "
            f"{r['throughput_img_s']:>11.2f}"
        )


def _print_compare(torch_rows: list[dict[str, float]], onnx_rows: list[dict[str, float]]) -> None:
    torch_map = {int(r["batch"]): r for r in torch_rows}
    onnx_map = {int(r["batch"]): r for r in onnx_rows}

    print("\nPyTorch vs ONNX INT8")
    print(
        f"{'batch':>6}  {'pt_lat(ms)':>10}  {'onnx_lat(ms)':>12}  "
        f"{'lat_speedup':>11}  {'pt_tp':>9}  {'onnx_tp':>9}  {'tp_speedup':>10}"
    )
    for bsz in sorted(set(torch_map.keys()) & set(onnx_map.keys())):
        pt = torch_map[bsz]
        ox = onnx_map[bsz]
        lat_speedup = pt["lat_mean_ms"] / max(ox["lat_mean_ms"], 1e-9)
        tp_speedup = ox["throughput_img_s"] / max(pt["throughput_img_s"], 1e-9)
        print(
            f"{bsz:>6}  "
            f"{pt['lat_mean_ms']:>10.2f}  "
            f"{ox['lat_mean_ms']:>12.2f}  "
            f"{lat_speedup:>11.2f}x  "
            f"{pt['throughput_img_s']:>9.2f}  "
            f"{ox['throughput_img_s']:>9.2f}  "
            f"{tp_speedup:>10.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="model_cache/siglip_image_encoder.onnx")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--intra-threads", type=int, default=0)
    parser.add_argument("--inter-threads", type=int, default=0)
    parser.add_argument("--use-crops", action="store_true")
    parser.add_argument("--crops-dir", default="violation_crops")
    parser.add_argument("--max-crop-images", type=int, default=128)
    parser.add_argument("--compare-pytorch", action="store_true")
    parser.add_argument("--siglip-model", default="ViT-SO400M-14-SigLIP")
    parser.add_argument("--siglip-pretrained", default="webli")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

    if args.intra_threads > 0:
        try:
            import torch

            torch.set_num_threads(args.intra_threads)
        except Exception:
            pass

    session = _build_session(
        onnx_path=args.onnx,
        intra_threads=args.intra_threads,
        inter_threads=args.inter_threads,
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    crop_bank = None
    if args.use_crops:
        crop_bank = _load_crop_bank(
            crops_dir=args.crops_dir,
            image_size=args.image_size,
            max_images=args.max_crop_images,
        )

    print("=" * 78)
    print("SigLIP ONNX Benchmark (CPUExecutionProvider)")
    print(f"onnx_path        : {args.onnx}")
    print(f"input_name       : {input_name}")
    print(f"output_name      : {output_name}")
    print(f"image_size       : {args.image_size}")
    print(f"warmup / runs    : {args.warmup} / {args.runs}")
    print(f"batch_sizes      : {args.batch_sizes}")
    print(f"input_source     : {'violation_crops' if args.use_crops else 'synthetic'}")
    print(f"compare_pytorch  : {args.compare_pytorch}")
    print("=" * 78)

    onnx_results: list[dict[str, float]] = []
    for i, bsz in enumerate(args.batch_sizes):
        row = _bench_onnx_batch(
            session=session,
            input_name=input_name,
            batch_size=bsz,
            image_size=args.image_size,
            warmup=args.warmup,
            runs=args.runs,
            crop_bank=crop_bank,
            seed=args.seed + i,
        )
        onnx_results.append(row)

    _print_table("ONNX INT8", onnx_results)

    if args.compare_pytorch:
        visual_model = _load_pytorch_visual(args.siglip_model, args.siglip_pretrained)
        torch_results: list[dict[str, float]] = []
        for i, bsz in enumerate(args.batch_sizes):
            row = _bench_torch_batch(
                visual_model=visual_model,
                batch_size=bsz,
                image_size=args.image_size,
                warmup=args.warmup,
                runs=args.runs,
                crop_bank=crop_bank,
                seed=args.seed + i,
            )
            torch_results.append(row)

        _print_table("PyTorch FP32", torch_results)
        _print_compare(torch_results, onnx_results)

    print("\nUnits: latency/per_img in ms, throughput in images/sec")


if __name__ == "__main__":
    main()

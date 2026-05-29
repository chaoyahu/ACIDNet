import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acidnet.models.acidnet import ACIDNet
from acidnet.paths import DATASET_ROOT, OUTPUT_ROOT, PROJECT_ROOT


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".PNG", ".JPEG", ".BMP"}


@dataclass
class EvalSpec:
    name: str
    model: str
    weight: str
    input_dir: str
    gt_dir: Optional[str] = None
    output_dir: Optional[str] = None
    mode: Optional[str] = None
    alpha: Optional[float] = None
    use_gt_mean: bool = False
    recursive: bool = False


PAIRED_SPECS = [
    EvalSpec("LOLv1", "acidnet", "experiments/weights/lolv1/lolv1_28.37_final.pth", "LOLdataset/eval15/low", "LOLdataset/eval15/high"),
    EvalSpec("LOLv2_real", "acidnet", "experiments/weights/lolv2/ours_2r_24.26_final.pth", "LOLv2/Real_captured/Test/Low", "LOLv2/Real_captured/Test/Normal", alpha=0.81),
    EvalSpec("LOLv2_syn", "acidnet", "experiments/weights/lolv2_s/ours_lolv2s_26.04_final.pth", "LOLv2/Synthetic/Test/Low", "LOLv2/Synthetic/Test/Normal"),
    EvalSpec("SICE_mix", "acidnet", "experiments/weights/SCIE/SCIE.pth", "SICE/SICE_Mix", "SICE/SICE_Reshape"),
    EvalSpec("SICE_grad", "acidnet", "experiments/weights/SCIE/SCIE.pth", "SICE/SICE_Grad", "SICE/SICE_Reshape"),
]


UNPAIRED_SPECS = [
    EvalSpec("DICM", "acidnet", "experiments/weights/SCIE/SCIE.pth", "DICM"),
    EvalSpec("LIME", "acidnet", "experiments/weights/SCIE/SCIE.pth", "LIME"),
    EvalSpec("MEF", "acidnet", "experiments/weights/SCIE/SCIE.pth", "MEF"),
    EvalSpec("NPE", "acidnet", "experiments/weights/SCIE/SCIE.pth", "NPE"),
    EvalSpec("VV", "acidnet", "experiments/weights/SCIE/SCIE.pth", "VV"),
]


def image_files(path: Path, recursive: bool = False) -> list[Path]:
    iterator = path.rglob("*") if recursive else path.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix in IMAGE_SUFFIXES)


def read_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pad_to_factor(tensor: torch.Tensor, factor: int = 8) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = tensor.shape
    out_h = ((h + factor) // factor) * factor
    out_w = ((w + factor) // factor) * factor
    pad_h = out_h - h if h % factor != 0 else 0
    pad_w = out_w - w if w % factor != 0 else 0
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), "reflect")
    return tensor, h, w


def build_model(spec: EvalSpec, device: torch.device) -> torch.nn.Module:
    model = ACIDNet()
    model.to(device).eval()
    return model


def load_weight(model: torch.nn.Module, weight_path: Path, device: torch.device) -> None:
    from acidnet.checkpoints import load_model_weights
    load_model_weights(model, weight_path, map_location=device, strict=True)


def save_outputs(spec: EvalSpec, repo: Path, dataset_root: Path, output_root: Path, device: torch.device, limit: Optional[int]) -> tuple[bool, str, int]:
    weight_path = repo / spec.weight
    input_dir = dataset_root / spec.input_dir
    if not weight_path.exists():
        return False, f"missing weight: {weight_path}", 0
    if not input_dir.exists():
        return False, f"missing input dir: {input_dir}", 0

    files = image_files(input_dir, spec.recursive)
    if limit is not None:
        files = files[:limit]
    if not files:
        return False, f"no input images: {input_dir}", 0

    out_dir = output_root / (spec.output_dir or spec.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(spec, device)
    load_weight(model, weight_path, device)
    if spec.alpha is not None:
        model.trans.gated2 = True
        model.trans.alpha = spec.alpha

    pil_to_tensor = transforms.ToTensor()
    tensor_to_pil = transforms.ToPILImage()
    with torch.no_grad():
        for path in tqdm(files, desc=f"infer {spec.name}"):
            image = read_image(path)
            x = pil_to_tensor(image).unsqueeze(0).to(device)
            x, h, w = pad_to_factor(x)
            y = model(x)
            y = torch.clamp(y[:, :, :h, :w], 0, 1)
            rel_name = path.name
            tensor_to_pil(y.squeeze(0).detach().cpu()).save(out_dir / rel_name)
    return True, "ok", len(files)


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())


def calculate_ssim(target: np.ndarray, ref: np.ndarray) -> float:
    if target.shape != ref.shape:
        raise ValueError("Input images must have the same dimensions.")
    if target.ndim == 2:
        return ssim(target, ref)
    return float(np.array([ssim(target[:, :, i], ref[:, :, i]) for i in range(target.shape[2])]).mean())


def calculate_psnr(target: np.ndarray, ref: np.ndarray) -> float:
    diff = target.astype(np.float32) - ref.astype(np.float32)
    return float(10.0 * np.log10(255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8)))


def find_gt(gt_dir: Path, name: str) -> Optional[Path]:
    candidates = [
        gt_dir / name,
        gt_dir / Path(name).with_suffix(".png").name,
        gt_dir / Path(name).with_suffix(".jpg").name,
        gt_dir / Path(name).with_suffix(".JPG").name,
        gt_dir / Path(name).with_suffix(".PNG").name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def paired_metrics(spec: EvalSpec, dataset_root: Path, output_root: Path, device: torch.device, limit: Optional[int]) -> dict:
    out_dir = output_root / (spec.output_dir or spec.name)
    gt_dir = dataset_root / spec.gt_dir
    files = image_files(out_dir)
    if limit is not None:
        files = files[:limit]

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    scores = {"psnr": [], "ssim": [], "lpips": []}
    missing = 0
    with torch.no_grad():
        for out_path in tqdm(files, desc=f"metric {spec.name}"):
            gt_path = find_gt(gt_dir, out_path.name)
            if gt_path is None:
                missing += 1
                continue
            pred_img = read_image(out_path)
            gt_img = read_image(gt_path)
            if pred_img.size != gt_img.size:
                pred_img = pred_img.resize(gt_img.size)
            pred = np.array(pred_img)
            gt = np.array(gt_img)
            if spec.use_gt_mean:
                mean_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY).mean()
                mean_gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).mean()
                if mean_pred > 0:
                    pred = np.clip(pred * (mean_gt / mean_pred), 0, 255)
            scores["psnr"].append(calculate_psnr(pred, gt))
            scores["ssim"].append(calculate_ssim(pred, gt))
            pred_t = lpips.im2tensor(pred).to(device)
            gt_t = lpips.im2tensor(gt).to(device)
            scores["lpips"].append(float(loss_fn.forward(gt_t, pred_t).item()))

    n = len(scores["psnr"])
    return {
        "dataset": spec.name,
        "weight": spec.weight,
        "status": "ok" if n else "no_metrics",
        "images": n,
        "missing_gt": missing,
        "psnr": float(np.mean(scores["psnr"])) if n else None,
        "ssim": float(np.mean(scores["ssim"])) if n else None,
        "lpips": float(np.mean(scores["lpips"])) if n else None,
    }


def write_results(rows: list[dict], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = ["dataset", "weight", "status", "images", "missing_gt", "psnr", "ssim", "lpips", "message"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ACIDNet checkpoints on benchmark datasets.")
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for smoke tests.")
    parser.add_argument("--paired-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    specs = PAIRED_SPECS if args.paired_only else PAIRED_SPECS + UNPAIRED_SPECS
    rows = []
    for spec in specs:
        ok, message, count = save_outputs(spec, args.repo, args.dataset_root, args.output_root, device, args.limit)
        if not ok:
            rows.append({"dataset": spec.name, "weight": spec.weight, "status": "failed", "images": 0, "message": message})
            continue
        if spec.gt_dir:
            row = paired_metrics(spec, args.dataset_root, args.output_root, device, args.limit)
            row["message"] = message
            rows.append(row)
        else:
            rows.append({"dataset": spec.name, "weight": spec.weight, "status": "inference_ok", "images": count, "message": message})

    write_results(rows, args.repo / "outputs/benchmark_metrics.json", args.repo / "outputs/benchmark_metrics.csv")
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

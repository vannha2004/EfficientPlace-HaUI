from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "Bigblue1_EfficientPlace_SafeHPWL_Stage1.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    markdown_cell(
        r'''
# Stage 1 — Safe HPWL refinement từ EfficientPlace trên `bigblue1`

Notebook đầu tiên này chỉ trả lời **một câu hỏi thực nghiệm**:

> Có thể giảm pin-aware HPWL của placement EfficientPlace mà luôn giữ `overlap = 0` và không vượt boundary hay không?

Protocol:

1. đọc trực tiếp `best_X_eq6.csv` từ dataset checkpoint, không rollout lại `actor_net.pth` cuối run;
2. audit ba định nghĩa HPWL: center-only Eq.(5), pin-aware liên tục và proxy lượng tử hóa của EfficientPlace;
3. chạy local refinement trên GPU với **hard feasibility filter**;
4. chỉ chấp nhận move nếu pin-aware HPWL giảm; nghiệm cuối luôn có fallback về checkpoint gốc;
5. xuất đầy đủ CSV/JSON/NPY/Bookshelf `.pl` để tái lập.

Notebook không tối ưu RUDY/congestion. Đó là Stage 2, chỉ thực hiện sau khi Stage 1 vượt audit.

## Dữ liệu Kaggle cần gắn

- `/kaggle/input/datasets/tranvannha/ckpt-efficientplace-bigblue1`
- dataset ISPD2005 có các file `bigblue1.nodes`, `bigblue1.nets`, `bigblue1.pl`, `bigblue1.aux`

Mặc định `RUN_MODE="smoke"`. Hãy chạy nguyên notebook một lần; sau khi audit thành công mới đổi thành `"full"`.
'''
    ),
    code_cell(
        r'''
# Cell 1 — Imports, cấu hình tái lập và chế độ chạy T4
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class ExperimentConfig:
    benchmark: str = "bigblue1"
    checkpoint_root: str = "/kaggle/input/datasets/tranvannha/ckpt-efficientplace-bigblue1"
    benchmark_roots: Tuple[str, ...] = (
        "/kaggle/input/datasets/tranvannha/ispd2005",
        "/kaggle/input/ispd2005",
        "/kaggle/input/ISPD2005",
    )
    output_root: str = "/kaggle/working/safe_hpwl_bigblue1_stage1"
    run_mode: str = "smoke"  # đổi thành "full" sau khi smoke audit thành công
    grid_fallback: int = 512
    rank_mode: int = 1
    hpwl_tolerance: float = 1e-6
    geometry_tolerance: float = 1e-7
    smoke_seeds: Tuple[int, ...] = (0,)
    full_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    smoke_radii_cells: Tuple[int, ...] = (16, 8, 4, 2, 1)
    full_radii_cells: Tuple[int, ...] = (64, 32, 16, 8, 4, 2, 1)
    smoke_candidates_per_macro: int = 64
    full_candidates_per_macro: int = 128
    smoke_max_macros_per_sweep: int = 192
    sweeps_per_radius_smoke: int = 1
    sweeps_per_radius_full: int = 2


CFG = ExperimentConfig()
if CFG.run_mode not in {"smoke", "full"}:
    raise ValueError("run_mode must be 'smoke' or 'full'")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_ROOT = Path(CFG.output_root)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
else:
    print("WARNING: GPU is unavailable; Kaggle Accelerator should be set to GPU T4 x1.")
print("Run mode:", CFG.run_mode)
'''
    ),
    code_cell(
        r'''
# Cell 2 — Khám phá checkpoint và benchmark; không đoán thứ tự macro
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_checkpoint_run(root: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    candidates = sorted(root.rglob("eq6_best.json"))
    valid = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        shape = payload.get("X_shape", [])
        run_root = path.parent.parent
        csv_path = run_root / "sol" / "best_X_eq6.csv"
        if len(shape) == 2 and int(shape[1]) == 2 and csv_path.exists():
            valid.append((path, run_root, payload))
    if len(valid) != 1:
        details = [str(item[0]) for item in valid]
        raise RuntimeError(
            "Expected exactly one run containing eq6_best.json + best_X_eq6.csv; "
            f"found {len(valid)}: {details}"
        )
    return valid[0]


def find_benchmark_base(benchmark: str, roots: Sequence[Path]) -> Path:
    matches: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for aux_path in root.rglob(f"{benchmark}.aux"):
            base = aux_path.with_suffix("")
            if all(base.with_suffix(suffix).exists() for suffix in (".nodes", ".nets", ".pl")):
                matches.append(base)
    unique = sorted({path.resolve() for path in matches})
    if not unique:
        raise FileNotFoundError(
            f"Cannot find {benchmark}.aux/.nodes/.nets/.pl. Attach the ISPD2005 dataset, "
            f"then rerun this cell. Searched roots: {[str(root) for root in roots]}"
        )
    preferred = [path for path in unique if "ISPD2005_clean" in str(path)]
    chosen = preferred[0] if preferred else unique[0]
    if len(unique) > 1:
        print("Multiple benchmark copies found; selected:", chosen)
    return chosen


def read_grid(checkpoint_root: Path, benchmark: str, fallback: int) -> Tuple[int, str]:
    yaml_paths = sorted(checkpoint_root.rglob(f"{benchmark}.yaml"))
    for path in yaml_paths:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if "grid" in payload:
                return int(payload["grid"]), str(path)
        except Exception:
            pass
    return int(fallback), "fallback"


CHECKPOINT_ROOT = Path(CFG.checkpoint_root)
if not CHECKPOINT_ROOT.exists():
    raise FileNotFoundError(f"Checkpoint dataset not attached: {CHECKPOINT_ROOT}")

EQ6_JSON_PATH, CHECKPOINT_RUN_ROOT, EQ6_METADATA = select_checkpoint_run(CHECKPOINT_ROOT)
SOLUTION_DIR = CHECKPOINT_RUN_ROOT / "sol"
X_CSV_PATH = SOLUTION_DIR / "best_X_eq6.csv"
X_NPY_PATH = SOLUTION_DIR / "best_X_eq6.npy"
HPWL_HISTORY_PATH = CHECKPOINT_RUN_ROOT / "model" / "hpwl_history.csv"

search_roots = [CHECKPOINT_ROOT, *(Path(value) for value in CFG.benchmark_roots)]
BENCHMARK_BASE = find_benchmark_base(CFG.benchmark, search_roots)
GRID, GRID_SOURCE = read_grid(CHECKPOINT_ROOT, CFG.benchmark, CFG.grid_fallback)

print("Checkpoint run :", CHECKPOINT_RUN_ROOT)
print("Eq.(6) metadata:", EQ6_JSON_PATH)
print("Placement CSV  :", X_CSV_PATH)
print("Benchmark base :", BENCHMARK_BASE)
print("Grid           :", GRID, "from", GRID_SOURCE)
print("Checkpoint SHA :", sha256_file(X_CSV_PATH))
'''
    ),
    code_cell(
        r'''
# Cell 3 — Parser Bookshelf tối thiểu và loader placement theo tên macro
NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def parse_nodes(path: Path) -> Dict[str, Tuple[float, float]]:
    macros: Dict[str, Tuple[float, float]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if len(fields) < 4 or not fields[-1].lower().startswith("terminal"):
            continue
        if not (NUMBER_PATTERN.match(fields[1]) and NUMBER_PATTERN.match(fields[2])):
            continue
        macros[fields[0]] = (float(fields[1]), float(fields[2]))
    if not macros:
        raise RuntimeError(f"No terminal macros parsed from {path}")
    return macros


def parse_original_pl(path: Path, names: set[str]) -> Dict[str, Tuple[float, float]]:
    positions: Dict[str, Tuple[float, float]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if len(fields) >= 3 and fields[0] in names:
            if NUMBER_PATTERN.match(fields[1]) and NUMBER_PATTERN.match(fields[2]):
                positions[fields[0]] = (float(fields[1]), float(fields[2]))
    return positions


def parse_macro_nets(
    path: Path,
    name_to_id: Dict[str, int],
) -> List[Dict[str, np.ndarray]]:
    raw_nets: List[List[Tuple[int, float, float]]] = []
    current: List[Tuple[int, float, float]] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("NetDegree"):
            if current is not None and len(current) >= 2:
                raw_nets.append(current)
            current = []
            continue
        if current is None or not raw[:1].isspace():
            continue
        fields = stripped.split()
        if len(fields) < 3 or fields[0] not in name_to_id:
            continue
        numeric = [float(token) for token in fields[1:] if NUMBER_PATTERN.match(token)]
        if len(numeric) < 2:
            continue
        current.append((name_to_id[fields[0]], numeric[-2], numeric[-1]))
    if current is not None and len(current) >= 2:
        raw_nets.append(current)

    nets: List[Dict[str, np.ndarray]] = []
    for pins in raw_nets:
        deduplicated: Dict[int, Tuple[float, float]] = {}
        for macro_id, dx, dy in pins:
            deduplicated.setdefault(macro_id, (dx, dy))
        if len(deduplicated) < 2:
            continue
        ids = np.asarray(list(deduplicated), dtype=np.int64)
        offsets = np.asarray([deduplicated[idx] for idx in ids], dtype=np.float64)
        nets.append({"ids": ids, "offsets": offsets})
    if not nets:
        raise RuntimeError(f"No macro-to-macro nets parsed from {path}")
    return nets


NODE_DIMENSIONS = parse_nodes(BENCHMARK_BASE.with_suffix(".nodes"))
checkpoint_frame = pd.read_csv(X_CSV_PATH)
required_columns = {"macro", "x_center", "y_center", "width", "height"}
if not required_columns.issubset(checkpoint_frame.columns):
    raise RuntimeError(f"Missing columns in {X_CSV_PATH}: {required_columns - set(checkpoint_frame.columns)}")
if checkpoint_frame["macro"].duplicated().any():
    raise RuntimeError("Duplicate macro names in best_X_eq6.csv")

MACRO_NAMES = checkpoint_frame["macro"].astype(str).tolist()
missing_nodes = sorted(set(MACRO_NAMES) - set(NODE_DIMENSIONS))
extra_nodes = sorted(set(NODE_DIMENSIONS) - set(MACRO_NAMES))
if missing_nodes or extra_nodes:
    raise RuntimeError(
        f"Checkpoint/benchmark macro mismatch: missing={missing_nodes[:10]}, extra={extra_nodes[:10]}"
    )

NAME_TO_ID = {name: idx for idx, name in enumerate(MACRO_NAMES)}
CENTERS_0 = checkpoint_frame[["x_center", "y_center"]].to_numpy(dtype=np.float64)
SIZES = checkpoint_frame[["width", "height"]].to_numpy(dtype=np.float64)
BOOKSHELF_SIZES = np.asarray([NODE_DIMENSIONS[name] for name in MACRO_NAMES], dtype=np.float64)
if not np.allclose(SIZES, BOOKSHELF_SIZES, rtol=0.0, atol=1e-6):
    raise RuntimeError("Macro dimensions in checkpoint CSV do not match the Bookshelf .nodes file")
if X_NPY_PATH.exists():
    npy_centers = np.load(X_NPY_PATH)
    if npy_centers.shape != CENTERS_0.shape or not np.allclose(npy_centers, CENTERS_0, atol=1e-7):
        raise RuntimeError("best_X_eq6.npy and best_X_eq6.csv disagree")

ORIGINAL_POSITIONS = parse_original_pl(BENCHMARK_BASE.with_suffix(".pl"), set(MACRO_NAMES))
if len(ORIGINAL_POSITIONS) != len(MACRO_NAMES):
    raise RuntimeError(f"Parsed {len(ORIGINAL_POSITIONS)}/{len(MACRO_NAMES)} macro positions from .pl")
max_x = max(ORIGINAL_POSITIONS[name][0] + NODE_DIMENSIONS[name][0] for name in MACRO_NAMES)
max_y = max(ORIGINAL_POSITIONS[name][1] + NODE_DIMENSIONS[name][1] for name in MACRO_NAMES)
# Khớp PlaceDB của EfficientPlace: canvas vuông dùng min(max_x, max_y).
CANVAS_EXTENT = float(min(max_x, max_y))
GRID_PITCH = CANVAS_EXTENT / GRID

NETS = parse_macro_nets(BENCHMARK_BASE.with_suffix(".nets"), NAME_TO_ID)
MACRO_TO_NETS: List[List[Tuple[int, int]]] = [[] for _ in MACRO_NAMES]
for net_id, net in enumerate(NETS):
    for local_pin, macro_id in enumerate(net["ids"]):
        MACRO_TO_NETS[int(macro_id)].append((net_id, local_pin))

print("Macros        :", len(MACRO_NAMES))
print("Macro nets    :", len(NETS))
print("Canvas extent :", CANVAS_EXTENT)
print("Grid pitch    :", GRID_PITCH)
print("First macros  :", MACRO_NAMES[:5])
'''
    ),
    code_cell(
        r'''
# Cell 4 — Unified evaluator và audit checkpoint
def hpwl_center(centers: np.ndarray) -> float:
    total = 0.0
    for net in NETS:
        points = centers[net["ids"]]
        total += float(np.ptp(points[:, 0]) + np.ptp(points[:, 1]))
    return total


def hpwl_pin(centers: np.ndarray) -> float:
    total = 0.0
    for net in NETS:
        pins = centers[net["ids"]] + net["offsets"]
        total += float(np.ptp(pins[:, 0]) + np.ptp(pins[:, 1]))
    return total


def hpwl_efficientplace_proxy(centers: np.ndarray) -> float:
    total_grid = 0.0
    for net in NETS:
        pins = centers[net["ids"]] + net["offsets"]
        quantized = np.rint(pins / GRID_PITCH)
        total_grid += float(np.ptp(quantized[:, 0]) + np.ptp(quantized[:, 1]))
    return total_grid * GRID_PITCH


def legality_metrics(centers: np.ndarray) -> Dict[str, Any]:
    lower = centers - SIZES / 2.0
    upper = centers + SIZES / 2.0
    tol = CFG.geometry_tolerance
    boundary_mask = np.any(lower < -tol, axis=1) | np.any(upper > CANVAS_EXTENT + tol, axis=1)
    max_boundary_violation = float(max(
        np.maximum(-lower, 0.0).max(initial=0.0),
        np.maximum(upper - CANVAS_EXTENT, 0.0).max(initial=0.0),
    ))
    overlap_pairs = 0
    overlap_area = 0.0
    for idx in range(len(centers) - 1):
        overlap_x = np.minimum(upper[idx, 0], upper[idx + 1:, 0]) - np.maximum(
            lower[idx, 0], lower[idx + 1:, 0]
        )
        overlap_y = np.minimum(upper[idx, 1], upper[idx + 1:, 1]) - np.maximum(
            lower[idx, 1], lower[idx + 1:, 1]
        )
        positive = (overlap_x > tol) & (overlap_y > tol)
        overlap_pairs += int(positive.sum())
        if np.any(positive):
            overlap_area += float((overlap_x[positive] * overlap_y[positive]).sum())
    return {
        "boundary_violations": int(boundary_mask.sum()),
        "max_boundary_violation": max_boundary_violation,
        "overlap_pairs": overlap_pairs,
        "pairwise_overlap_area": overlap_area,
        "legal": bool(boundary_mask.sum() == 0 and overlap_pairs == 0),
    }


def evaluate(centers: np.ndarray) -> Dict[str, Any]:
    metrics = {
        "hpwl_center": hpwl_center(centers),
        "hpwl_pin": hpwl_pin(centers),
        "hpwl_efficientplace_proxy": hpwl_efficientplace_proxy(centers),
    }
    metrics.update(legality_metrics(centers))
    metrics["hpwl_center_x1e5"] = metrics["hpwl_center"] / 1e5
    metrics["hpwl_pin_x1e5"] = metrics["hpwl_pin"] / 1e5
    metrics["hpwl_efficientplace_proxy_x1e5"] = metrics["hpwl_efficientplace_proxy"] / 1e5
    return metrics


BASELINE_METRICS = evaluate(CENTERS_0)
checkpoint_center_hpwl = float(EQ6_METADATA["hpwl_eq5"])
if not math.isclose(BASELINE_METRICS["hpwl_center"], checkpoint_center_hpwl, rel_tol=0.0, abs_tol=1e-5):
    raise AssertionError(
        f"Center HPWL mismatch: evaluator={BASELINE_METRICS['hpwl_center']}, "
        f"checkpoint={checkpoint_center_hpwl}"
    )

checkpoint_episode = int(EQ6_METADATA["global_episode"])
checkpoint_proxy_x1e5 = None
if HPWL_HISTORY_PATH.exists():
    history_frame = pd.read_csv(HPWL_HISTORY_PATH)
    selected = history_frame.loc[history_frame["global_episode"] == checkpoint_episode]
    if len(selected) == 1:
        checkpoint_proxy_x1e5 = float(selected.iloc[0]["hpwl"])

audit_table = pd.DataFrame([
    {"metric": "Checkpoint Eq.(5), center-only", "value_x1e5": checkpoint_center_hpwl / 1e5},
    {"metric": "Unified center-only", "value_x1e5": BASELINE_METRICS["hpwl_center_x1e5"]},
    {"metric": "Unified pin-aware (PRIMARY)", "value_x1e5": BASELINE_METRICS["hpwl_pin_x1e5"]},
    {"metric": "Unified EfficientPlace proxy", "value_x1e5": BASELINE_METRICS["hpwl_efficientplace_proxy_x1e5"]},
    {"metric": "Checkpoint history proxy", "value_x1e5": checkpoint_proxy_x1e5},
])
display(audit_table)
display(pd.DataFrame([BASELINE_METRICS]))

if not BASELINE_METRICS["legal"]:
    raise AssertionError(f"Checkpoint placement is not legal: {BASELINE_METRICS}")
print("AUDIT PASSED: checkpoint is legal and center HPWL reproduces Eq.(5).")
'''
    ),
    code_cell(
        r'''
# Cell 5 — Trực quan hóa placement ban đầu
def draw_placement(axis, centers: np.ndarray, title: str) -> None:
    lower = centers - SIZES / 2.0
    for idx in range(len(centers)):
        axis.add_patch(Rectangle(
            lower[idx], SIZES[idx, 0], SIZES[idx, 1],
            fill=False, linewidth=0.28, alpha=0.65,
        ))
    axis.set_xlim(0, CANVAS_EXTENT)
    axis.set_ylim(0, CANVAS_EXTENT)
    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title)


figure, axis = plt.subplots(figsize=(7, 7))
draw_placement(
    axis,
    CENTERS_0,
    f"EfficientPlace bigblue1\npin-HPWL={BASELINE_METRICS['hpwl_pin_x1e5']:.6f}×1e5",
)
figure.tight_layout()
plt.show()
'''
    ),
    markdown_cell(
        r'''
## Thuật toán Stage 1

Mỗi move thay đổi đúng một macro trên grid gốc của EfficientPlace. Với một macro đang xét:

1. tạo ứng viên quanh vị trí hiện tại và quanh median của các pin lân cận;
2. GPU loại mọi ứng viên vượt boundary hoặc overlap với bất kỳ macro nào;
3. GPU tính chính xác thay đổi pin-aware HPWL chỉ trên các net liên quan;
4. chỉ chấp nhận ứng viên có `ΔHPWL < 0`;
5. sau mỗi sweep, full evaluator kiểm tra lại legality và monotonicity.

Đây là hard safety gate: nếu không tìm được cải thiện, output bằng đúng checkpoint gốc.
'''
    ),
    code_cell(
        r'''
# Cell 6 — GPU legality-preserving monotone local refiner
class SafeHPWLRefiner:
    def __init__(
        self,
        initial_centers: np.ndarray,
        seed: int,
        candidates_per_macro: int,
        max_macros_per_sweep: int | None,
    ) -> None:
        self.centers = np.asarray(initial_centers, dtype=np.float64).copy()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.candidates_per_macro = int(candidates_per_macro)
        self.max_macros_per_sweep = max_macros_per_sweep
        self.net_values = np.asarray([self._net_hpwl(net_id) for net_id in range(len(NETS))])
        self.total_hpwl = float(self.net_values.sum())
        self.initial_hpwl = self.total_hpwl
        self.history: List[Dict[str, Any]] = []
        self.accepted_moves = 0
        self.start_time = time.perf_counter()

    def _net_hpwl(self, net_id: int) -> float:
        net = NETS[net_id]
        pins = self.centers[net["ids"]] + net["offsets"]
        return float(np.ptp(pins[:, 0]) + np.ptp(pins[:, 1]))

    def _target_center(self, macro_id: int) -> np.ndarray:
        targets = []
        for net_id, local_pin in MACRO_TO_NETS[macro_id]:
            net = NETS[net_id]
            mask = np.arange(len(net["ids"])) != local_pin
            if not np.any(mask):
                continue
            other_pins = self.centers[net["ids"][mask]] + net["offsets"][mask]
            desired_pin = np.median(other_pins, axis=0)
            targets.append(desired_pin - net["offsets"][local_pin])
        if not targets:
            return self.centers[macro_id].copy()
        return np.median(np.asarray(targets), axis=0)

    def _candidate_lower(self, macro_id: int, radius_cells: int) -> np.ndarray:
        size = SIZES[macro_id]
        current_lower = self.centers[macro_id] - size / 2.0
        target_lower = self._target_center(macro_id) - size / 2.0
        radius = int(radius_cells)
        directions = np.asarray([
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2),
        ], dtype=np.float64)
        scale = GRID_PITCH * max(radius, 1)
        candidates = [current_lower[None, :], target_lower[None, :]]
        candidates.append(current_lower[None, :] + directions * scale)
        candidates.append(target_lower[None, :] + directions * scale)

        random_count = max(self.candidates_per_macro - sum(len(item) for item in candidates), 0)
        if random_count:
            around_current = random_count // 2
            offsets_a = self.rng.integers(-radius, radius + 1, size=(around_current, 2))
            offsets_b = self.rng.integers(-radius, radius + 1, size=(random_count - around_current, 2))
            candidates.append(current_lower[None, :] + offsets_a * GRID_PITCH)
            candidates.append(target_lower[None, :] + offsets_b * GRID_PITCH)

        candidate_lower = np.vstack(candidates)
        max_lower = CANVAS_EXTENT - size
        candidate_lower = np.clip(candidate_lower, 0.0, max_lower)
        candidate_lower = np.rint(candidate_lower / GRID_PITCH) * GRID_PITCH
        candidate_lower = np.clip(candidate_lower, 0.0, max_lower)
        candidate_lower = np.unique(np.round(candidate_lower, 8), axis=0)
        return candidate_lower

    def _feasible_mask(self, macro_id: int, candidate_lower: np.ndarray) -> torch.Tensor:
        candidate = torch.as_tensor(candidate_lower, dtype=torch.float64, device=DEVICE)
        size = torch.as_tensor(SIZES[macro_id], dtype=torch.float64, device=DEVICE)
        upper = candidate + size

        current_lower = self.centers - SIZES / 2.0
        other_mask = np.arange(len(self.centers)) != macro_id
        other_lower = torch.as_tensor(current_lower[other_mask], dtype=torch.float64, device=DEVICE)
        other_upper = torch.as_tensor(
            current_lower[other_mask] + SIZES[other_mask], dtype=torch.float64, device=DEVICE
        )
        tol = CFG.geometry_tolerance
        separated = (
            (upper[:, None, 0] <= other_lower[None, :, 0] + tol)
            | (candidate[:, None, 0] >= other_upper[None, :, 0] - tol)
            | (upper[:, None, 1] <= other_lower[None, :, 1] + tol)
            | (candidate[:, None, 1] >= other_upper[None, :, 1] - tol)
        )
        no_overlap = separated.all(dim=1)
        in_bounds = (candidate >= -tol).all(dim=1) & (upper <= CANVAS_EXTENT + tol).all(dim=1)
        return no_overlap & in_bounds

    def _candidate_deltas(self, macro_id: int, candidate_centers: np.ndarray) -> torch.Tensor:
        candidate = torch.as_tensor(candidate_centers, dtype=torch.float64, device=DEVICE)
        deltas = torch.zeros(len(candidate), dtype=torch.float64, device=DEVICE)
        for net_id, local_pin in MACRO_TO_NETS[macro_id]:
            net = NETS[net_id]
            mask = np.arange(len(net["ids"])) != local_pin
            other_pins = self.centers[net["ids"][mask]] + net["offsets"][mask]
            min_other = torch.as_tensor(other_pins.min(axis=0), dtype=torch.float64, device=DEVICE)
            max_other = torch.as_tensor(other_pins.max(axis=0), dtype=torch.float64, device=DEVICE)
            offset = torch.as_tensor(net["offsets"][local_pin], dtype=torch.float64, device=DEVICE)
            pin = candidate + offset
            new_span = (
                torch.maximum(max_other[0], pin[:, 0]) - torch.minimum(min_other[0], pin[:, 0])
                + torch.maximum(max_other[1], pin[:, 1]) - torch.minimum(min_other[1], pin[:, 1])
            )
            deltas += new_span - float(self.net_values[net_id])
        return deltas

    def try_move(self, macro_id: int, radius_cells: int) -> bool:
        candidate_lower = self._candidate_lower(macro_id, radius_cells)
        feasible = self._feasible_mask(macro_id, candidate_lower)
        if not bool(feasible.any()):
            return False
        candidate_centers = candidate_lower + SIZES[macro_id] / 2.0
        deltas = self._candidate_deltas(macro_id, candidate_centers)
        deltas = torch.where(feasible, deltas, torch.full_like(deltas, float("inf")))
        best_index = int(torch.argmin(deltas).item())
        best_delta = float(deltas[best_index].item())
        if not math.isfinite(best_delta) or best_delta >= -CFG.hpwl_tolerance:
            return False

        old_center = self.centers[macro_id].copy()
        old_total = self.total_hpwl
        incident_net_ids = [net_id for net_id, _ in MACRO_TO_NETS[macro_id]]
        old_values = self.net_values[incident_net_ids].copy()
        self.centers[macro_id] = candidate_centers[best_index]
        for net_id in incident_net_ids:
            self.net_values[net_id] = self._net_hpwl(net_id)
        self.total_hpwl = float(self.net_values.sum())

        if self.total_hpwl >= old_total - CFG.hpwl_tolerance:
            self.centers[macro_id] = old_center
            self.net_values[incident_net_ids] = old_values
            self.total_hpwl = old_total
            return False
        self.accepted_moves += 1
        return True

    def _macro_order(self) -> np.ndarray:
        criticality = np.asarray([
            sum(self.net_values[net_id] for net_id, _ in MACRO_TO_NETS[macro_id])
            for macro_id in range(len(self.centers))
        ])
        jitter = self.rng.uniform(0.0, 1e-9, size=len(criticality))
        order = np.argsort(-(criticality + jitter))
        if self.max_macros_per_sweep is not None:
            order = order[: self.max_macros_per_sweep]
        return order

    def run(self, radii_cells: Sequence[int], sweeps_per_radius: int) -> Tuple[np.ndarray, pd.DataFrame]:
        previous_full_hpwl = self.total_hpwl
        for radius in radii_cells:
            for sweep in range(int(sweeps_per_radius)):
                accepted_before = self.accepted_moves
                for macro_id in self._macro_order():
                    self.try_move(int(macro_id), int(radius))

                full_metrics = evaluate(self.centers)
                if not full_metrics["legal"]:
                    raise AssertionError(f"Legality violated after radius={radius}, sweep={sweep}")
                if full_metrics["hpwl_pin"] > previous_full_hpwl + CFG.hpwl_tolerance:
                    raise AssertionError("Full pin-HPWL increased despite the monotone acceptance rule")
                previous_full_hpwl = float(full_metrics["hpwl_pin"])
                row = {
                    "seed": self.seed,
                    "radius_cells": int(radius),
                    "sweep": int(sweep),
                    "accepted_this_sweep": self.accepted_moves - accepted_before,
                    "accepted_total": self.accepted_moves,
                    "hpwl_pin": full_metrics["hpwl_pin"],
                    "hpwl_pin_x1e5": full_metrics["hpwl_pin_x1e5"],
                    "improvement_pct": 100.0 * (1.0 - full_metrics["hpwl_pin"] / self.initial_hpwl),
                    "runtime_s": time.perf_counter() - self.start_time,
                }
                self.history.append(row)
                print(
                    f"seed={self.seed} radius={radius:>3} sweep={sweep} "
                    f"accepted={row['accepted_this_sweep']:>3} "
                    f"HPWL={row['hpwl_pin_x1e5']:.6f}×1e5 "
                    f"improvement={row['improvement_pct']:.4f}%"
                )
        return self.centers.copy(), pd.DataFrame(self.history)
'''
    ),
    code_cell(
        r'''
# Cell 7 — Chạy smoke/full experiment và lưu kết quả từng seed
if CFG.run_mode == "smoke":
    RUN_SEEDS = CFG.smoke_seeds
    RADII_CELLS = CFG.smoke_radii_cells
    CANDIDATES_PER_MACRO = CFG.smoke_candidates_per_macro
    MAX_MACROS_PER_SWEEP = CFG.smoke_max_macros_per_sweep
    SWEEPS_PER_RADIUS = CFG.sweeps_per_radius_smoke
else:
    RUN_SEEDS = CFG.full_seeds
    RADII_CELLS = CFG.full_radii_cells
    CANDIDATES_PER_MACRO = CFG.full_candidates_per_macro
    MAX_MACROS_PER_SWEEP = None
    SWEEPS_PER_RADIUS = CFG.sweeps_per_radius_full

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = OUTPUT_ROOT / f"{CFG.run_mode}_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=False)

seed_records = []
seed_centers: Dict[int, np.ndarray] = {}
all_history = []
for seed in RUN_SEEDS:
    print(f"\n===== Safe HPWL refinement: seed {seed} =====")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    refiner = SafeHPWLRefiner(
        CENTERS_0,
        seed=seed,
        candidates_per_macro=CANDIDATES_PER_MACRO,
        max_macros_per_sweep=MAX_MACROS_PER_SWEEP,
    )
    refined, history = refiner.run(RADII_CELLS, SWEEPS_PER_RADIUS)
    metrics = evaluate(refined)
    if not metrics["legal"]:
        raise AssertionError(f"Seed {seed} produced an illegal final placement")
    if metrics["hpwl_pin"] > BASELINE_METRICS["hpwl_pin"] + CFG.hpwl_tolerance:
        raise AssertionError(f"Seed {seed} violated safe HPWL fallback")

    seed_dir = RUN_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    np.save(seed_dir / "refined_centers.npy", refined)
    pd.DataFrame({
        "macro": MACRO_NAMES,
        "x_center": refined[:, 0],
        "y_center": refined[:, 1],
        "width": SIZES[:, 0],
        "height": SIZES[:, 1],
    }).to_csv(seed_dir / "refined_centers.csv", index=False)
    history.to_csv(seed_dir / "history.csv", index=False)

    record = {
        "seed": int(seed),
        "run_mode": CFG.run_mode,
        "accepted_moves": int(refiner.accepted_moves),
        "baseline_hpwl_pin": float(BASELINE_METRICS["hpwl_pin"]),
        "final_hpwl_pin": float(metrics["hpwl_pin"]),
        "final_hpwl_pin_x1e5": float(metrics["hpwl_pin_x1e5"]),
        "improvement_pct": float(100.0 * (1.0 - metrics["hpwl_pin"] / BASELINE_METRICS["hpwl_pin"])),
        "overlap_pairs": int(metrics["overlap_pairs"]),
        "boundary_violations": int(metrics["boundary_violations"]),
        "legal": bool(metrics["legal"]),
        "runtime_s": float(history["runtime_s"].iloc[-1]) if len(history) else 0.0,
    }
    seed_records.append(record)
    seed_centers[int(seed)] = refined
    all_history.append(history)

RESULTS = pd.DataFrame(seed_records).sort_values("final_hpwl_pin").reset_index(drop=True)
RESULTS.to_csv(RUN_DIR / "per_seed_results.csv", index=False)
if all_history:
    pd.concat(all_history, ignore_index=True).to_csv(RUN_DIR / "all_history.csv", index=False)
display(RESULTS)

BEST_SEED = int(RESULTS.iloc[0]["seed"])
BEST_CENTERS = seed_centers[BEST_SEED]
BEST_METRICS = evaluate(BEST_CENTERS)
print("Best seed:", BEST_SEED)
print("Best metrics:", json.dumps(BEST_METRICS, indent=2))
'''
    ),
    code_cell(
        r'''
# Cell 8 — Xuất Bookshelf .pl, manifest và archive
def write_bookshelf_pl(source_path: Path, destination_path: Path, centers: np.ndarray) -> None:
    lower = centers - SIZES / 2.0
    replacement = {name: lower[idx] for idx, name in enumerate(MACRO_NAMES)}
    output_lines = []
    for raw in source_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
        fields = raw.split()
        if fields and fields[0] in replacement:
            suffix = raw[raw.find(":"):].strip() if ":" in raw else ": N /FIXED"
            x, y = replacement[fields[0]]
            output_lines.append(f"{fields[0]}\t{x:.8f}\t{y:.8f} {suffix}\n")
        else:
            output_lines.append(raw)
    destination_path.write_text("".join(output_lines), encoding="utf-8")


np.save(RUN_DIR / "best_centers.npy", BEST_CENTERS)
best_frame = pd.DataFrame({
    "macro": MACRO_NAMES,
    "x_center": BEST_CENTERS[:, 0],
    "y_center": BEST_CENTERS[:, 1],
    "width": SIZES[:, 0],
    "height": SIZES[:, 1],
})
best_frame.to_csv(RUN_DIR / "best_centers.csv", index=False)
BEST_PL_PATH = RUN_DIR / f"{CFG.benchmark}.efficientplace_safe_hpwl.pl"
write_bookshelf_pl(BENCHMARK_BASE.with_suffix(".pl"), BEST_PL_PATH, BEST_CENTERS)

manifest = {
    "experiment": "stage1_safe_hpwl_refinement",
    "config": asdict(CFG),
    "device": str(DEVICE),
    "checkpoint_run_root": str(CHECKPOINT_RUN_ROOT),
    "checkpoint_eq6_json": str(EQ6_JSON_PATH),
    "checkpoint_x_csv": str(X_CSV_PATH),
    "checkpoint_x_csv_sha256": sha256_file(X_CSV_PATH),
    "benchmark_base": str(BENCHMARK_BASE),
    "grid": int(GRID),
    "grid_pitch": float(GRID_PITCH),
    "macro_count": len(MACRO_NAMES),
    "macro_net_count": len(NETS),
    "primary_metric": "continuous pin-aware macro-only HPWL",
    "baseline_metrics": BASELINE_METRICS,
    "best_seed": BEST_SEED,
    "best_metrics": BEST_METRICS,
    "safe_gate_passed": bool(
        BEST_METRICS["legal"]
        and BEST_METRICS["hpwl_pin"] <= BASELINE_METRICS["hpwl_pin"] + CFG.hpwl_tolerance
    ),
}
(RUN_DIR / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, default=float), encoding="utf-8"
)

archive_path = shutil.make_archive(str(RUN_DIR), "zip", root_dir=RUN_DIR)
print("Bookshelf placement:", BEST_PL_PATH)
print("Manifest           :", RUN_DIR / "run_manifest.json")
print("Archive            :", archive_path)
'''
    ),
    code_cell(
        r'''
# Cell 9 — Đồ thị hội tụ và before/after
history_frame = pd.read_csv(RUN_DIR / "all_history.csv")
figure, axes = plt.subplots(1, 3, figsize=(19, 6))

draw_placement(
    axes[0], CENTERS_0,
    f"EfficientPlace input\nHPWL={BASELINE_METRICS['hpwl_pin_x1e5']:.6f}×1e5",
)
draw_placement(
    axes[1], BEST_CENTERS,
    f"Safe refined, seed={BEST_SEED}\nHPWL={BEST_METRICS['hpwl_pin_x1e5']:.6f}×1e5",
)
for seed, group in history_frame.groupby("seed"):
    axes[2].plot(
        np.arange(1, len(group) + 1), group["hpwl_pin_x1e5"],
        marker="o", linewidth=1.5, label=f"seed {seed}",
    )
axes[2].axhline(BASELINE_METRICS["hpwl_pin_x1e5"], color="black", linestyle="--", label="baseline")
axes[2].set_xlabel("completed sweep")
axes[2].set_ylabel("pin-aware HPWL (×1e5)")
axes[2].set_title("Monotone convergence")
axes[2].grid(alpha=0.25)
axes[2].legend()

figure.tight_layout()
plot_path = RUN_DIR / "safe_hpwl_before_after.png"
figure.savefig(plot_path, dpi=180, bbox_inches="tight")
plt.show()
print("Plot:", plot_path)
'''
    ),
    code_cell(
        r'''
# Cell 10 — Final safety assertions và thông tin cần gửi lại
assert BASELINE_METRICS["legal"], "Baseline must be legal"
assert BEST_METRICS["legal"], "Final placement must be legal"
assert BEST_METRICS["overlap_pairs"] == 0
assert BEST_METRICS["boundary_violations"] == 0
assert BEST_METRICS["hpwl_pin"] <= BASELINE_METRICS["hpwl_pin"] + CFG.hpwl_tolerance

summary = {
    "run_mode": CFG.run_mode,
    "baseline_hpwl_pin_x1e5": BASELINE_METRICS["hpwl_pin_x1e5"],
    "best_hpwl_pin_x1e5": BEST_METRICS["hpwl_pin_x1e5"],
    "improvement_pct": 100.0 * (
        1.0 - BEST_METRICS["hpwl_pin"] / BASELINE_METRICS["hpwl_pin"]
    ),
    "best_seed": BEST_SEED,
    "overlap_pairs": BEST_METRICS["overlap_pairs"],
    "boundary_violations": BEST_METRICS["boundary_violations"],
    "run_directory": str(RUN_DIR),
    "archive": str(archive_path),
}
print(json.dumps(summary, indent=2))
print("\nNEXT ACTION:")
if CFG.run_mode == "smoke":
    print("1) Tải archive .zip và gửi run_manifest.json/per_seed_results.csv để audit.")
    print("2) Nếu audit sạch, đổi CFG.run_mode thành 'full' và Run All.")
else:
    print("Gửi run_manifest.json, per_seed_results.csv và biểu đồ before/after để thiết kế Stage 2 Pareto.")
'''
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

DESTINATION.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DESTINATION}")

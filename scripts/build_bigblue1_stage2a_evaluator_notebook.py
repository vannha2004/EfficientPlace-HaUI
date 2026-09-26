from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "Bigblue1_EfficientPlace_Stage2A_Pareto_Evaluator.ipynb"


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
# Stage 2A — Pareto evaluator cho EfficientPlace + SafeHPWL trên `bigblue1`

Notebook này **chỉ đánh giá**, không tối ưu thêm. Mục tiêu là trả lời hai câu hỏi trước khi xây thuật toán congestion-aware:

1. Cải thiện pin-aware HPWL của Stage 1 có ổn định trên 5 seed không?
2. Placement tốt hơn về HPWL có làm proxy congestion xấu đi không?

Protocol:

- tự tìm baseline `best_X_eq6.csv` và output `full` của Stage 1;
- xác minh hash checkpoint, thứ tự macro, kích thước macro và toàn bộ legality;
- đánh giá center-only, pin-aware và EfficientPlace-proxy HPWL;
- tính macro occupancy và RUDY ở ba độ phân giải `32×32`, `64×64`, `128×128`;
- tạo Pareto front giữa pin-aware HPWL và capacity-adjusted RUDY;
- xuất CSV, JSON, báo cáo Markdown, heatmap và archive tái lập.

## Input cần gắn vào Kaggle

1. Dataset checkpoint EfficientPlace:
   `/kaggle/input/datasets/tranvannha/ckpt-efficientplace-bigblue1`
2. Dataset ISPD2005 chứa `bigblue1.nodes/.nets/.pl/.aux`.
3. **Notebook Output** của Stage 1 full run — phiên bản có `run_manifest.json`, `per_seed_results.csv` và các thư mục `seed_0` ... `seed_4`.

> Lưu ý khoa học: evaluator này chỉ dùng 409 macro-to-macro nets. RUDY là proxy, không phải routed congestion và kết quả không được gọi là full-chip HPWL.
'''
    ),
    code_cell(
        r'''
# Cell 1 — Imports và cấu hình evaluator cố định
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class EvaluationConfig:
    benchmark: str = "bigblue1"
    input_root: str = "/kaggle/input"
    checkpoint_root: str = "/kaggle/input/datasets/tranvannha/ckpt-efficientplace-bigblue1"
    benchmark_roots: Tuple[str, ...] = (
        "/kaggle/input/datasets/tranvannha/ispd2005",
        "/kaggle/input/ispd2005",
        "/kaggle/input/ISPD2005",
    )
    output_root: str = "/kaggle/working/bigblue1_stage2a_evaluator"
    expected_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    rudy_resolutions: Tuple[int, ...] = (32, 64, 128)
    primary_resolution: int = 64
    blockage_weight: float = 0.80
    capacity_floor: float = 0.10
    grid_fallback: int = 512
    geometry_tolerance: float = 1e-7
    metric_tolerance: float = 1e-5


CFG = EvaluationConfig()
if CFG.primary_resolution not in CFG.rudy_resolutions:
    raise ValueError("primary_resolution must be included in rudy_resolutions")
if not (0.0 <= CFG.blockage_weight <= 1.0):
    raise ValueError("blockage_weight must be in [0, 1]")
if not (0.0 < CFG.capacity_floor <= 1.0):
    raise ValueError("capacity_floor must be in (0, 1]")

INPUT_ROOT = Path(CFG.input_root)
OUTPUT_ROOT = Path(CFG.output_root)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = OUTPUT_ROOT / f"eval_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=False)

print("Evaluation mode : CPU deterministic")
print("RUDY resolutions:", CFG.rudy_resolutions)
print("Primary grid    :", CFG.primary_resolution)
print("Output          :", RUN_DIR)
'''
    ),
    code_cell(
        r'''
# Cell 2 — Tự tìm checkpoint, benchmark và output full của Stage 1
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_benchmark_base(benchmark: str, roots: Sequence[Path]) -> Path:
    matches: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for aux_path in root.rglob(f"{benchmark}.aux"):
            base = aux_path.with_suffix("")
            required = (".nodes", ".nets", ".pl")
            if all(base.with_suffix(suffix).exists() for suffix in required):
                matches.append(base)
    unique = sorted({path.resolve() for path in matches})
    if not unique:
        raise FileNotFoundError(
            f"Cannot find {benchmark}.aux/.nodes/.nets/.pl. Attach ISPD2005. "
            f"Searched: {[str(root) for root in roots]}"
        )
    preferred = [path for path in unique if "ISPD2005_clean" in str(path)]
    return preferred[0] if preferred else unique[0]


def select_checkpoint_run(search_root: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    valid = []
    for metadata_path in sorted(search_root.rglob("eq6_best.json")):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_root = metadata_path.parent.parent
        csv_path = run_root / "sol" / "best_X_eq6.csv"
        shape = payload.get("X_shape", [])
        if csv_path.exists() and len(shape) == 2 and int(shape[1]) == 2:
            valid.append((metadata_path, run_root, payload))
    if len(valid) != 1:
        raise RuntimeError(
            "Expected exactly one EfficientPlace run with eq6_best.json and best_X_eq6.csv; "
            f"found {len(valid)}: {[str(item[0]) for item in valid]}"
        )
    return valid[0]


def safe_extract_zip(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe path in archive {source}: {member.filename}")
        archive.extractall(destination)


def valid_stage1_manifests(root: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    records = []
    for manifest_path in root.rglob("run_manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = payload.get("config", {})
        if payload.get("experiment") != "stage1_safe_hpwl_refinement":
            continue
        if config.get("benchmark") != CFG.benchmark or config.get("run_mode") != "full":
            continue
        if not (manifest_path.parent / "per_seed_results.csv").exists():
            continue
        records.append((manifest_path, payload))
    return records


def select_stage1_run(input_root: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    records = valid_stage1_manifests(input_root)
    if not records:
        extraction_root = RUN_DIR / "attached_stage1_archive"
        for archive_path in sorted(input_root.rglob("full_*.zip")):
            target = extraction_root / archive_path.stem
            safe_extract_zip(archive_path, target)
        records = valid_stage1_manifests(extraction_root)
    if not records:
        raise FileNotFoundError(
            "Cannot find a Stage 1 full output. In Kaggle choose Add Input → Notebook Output → "
            "notebook7e94e56833 (the full-run version), then Run All."
        )

    scored = []
    for manifest_path, payload in records:
        stage1_root = manifest_path.parent
        result_frame = pd.read_csv(stage1_root / "per_seed_results.csv")
        seeds = sorted(result_frame["seed"].astype(int).tolist()) if "seed" in result_frame else []
        complete = seeds == sorted(CFG.expected_seeds)
        scored.append((int(complete), stage1_root.name, str(stage1_root), manifest_path, payload))
    scored.sort()
    best = scored[-1]
    if best[0] != 1:
        raise RuntimeError(
            f"Stage 1 output does not contain exactly seeds {CFG.expected_seeds}: {best[2]}"
        )
    if len(scored) > 1:
        print("Multiple Stage 1 full runs found; selected:", best[2])
    return Path(best[2]), best[3], best[4]


checkpoint_search_root = Path(CFG.checkpoint_root)
if not checkpoint_search_root.exists():
    checkpoint_search_root = INPUT_ROOT
EQ6_JSON_PATH, CHECKPOINT_RUN_ROOT, EQ6_METADATA = select_checkpoint_run(checkpoint_search_root)
X_CSV_PATH = CHECKPOINT_RUN_ROOT / "sol" / "best_X_eq6.csv"

BENCHMARK_BASE = find_benchmark_base(
    CFG.benchmark,
    [checkpoint_search_root, *(Path(value) for value in CFG.benchmark_roots), INPUT_ROOT],
)
STAGE1_ROOT, STAGE1_MANIFEST_PATH, STAGE1_MANIFEST = select_stage1_run(INPUT_ROOT)
STAGE1_RESULTS_PATH = STAGE1_ROOT / "per_seed_results.csv"

expected_checkpoint_hash = STAGE1_MANIFEST.get("checkpoint_x_csv_sha256")
actual_checkpoint_hash = sha256_file(X_CSV_PATH)
if expected_checkpoint_hash and expected_checkpoint_hash != actual_checkpoint_hash:
    raise AssertionError(
        "Checkpoint hash differs from the one used by Stage 1: "
        f"expected={expected_checkpoint_hash}, actual={actual_checkpoint_hash}"
    )

print("Checkpoint CSV :", X_CSV_PATH)
print("Checkpoint SHA :", actual_checkpoint_hash)
print("Benchmark base :", BENCHMARK_BASE)
print("Stage 1 root   :", STAGE1_ROOT)
print("Stage 1 SHA    :", sha256_file(STAGE1_MANIFEST_PATH))
'''
    ),
    code_cell(
        r'''
# Cell 3 — Parser Bookshelf và audit phạm vi metric
NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def parse_header_value(path: Path, key: str) -> int | None:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(\d+)", re.IGNORECASE)
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(raw)
        if match:
            return int(match.group(1))
    return None


def parse_nodes(path: Path) -> Dict[str, Tuple[float, float]]:
    macros: Dict[str, Tuple[float, float]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if len(fields) < 4 or not fields[-1].lower().startswith("terminal"):
            continue
        if NUMBER_PATTERN.match(fields[1]) and NUMBER_PATTERN.match(fields[2]):
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
        if len(numeric) >= 2:
            current.append((name_to_id[fields[0]], numeric[-2], numeric[-1]))
    if current is not None and len(current) >= 2:
        raw_nets.append(current)

    nets: List[Dict[str, np.ndarray]] = []
    for pins in raw_nets:
        unique: Dict[int, Tuple[float, float]] = {}
        for macro_id, dx, dy in pins:
            unique.setdefault(macro_id, (dx, dy))
        if len(unique) < 2:
            continue
        ids = np.asarray(list(unique), dtype=np.int64)
        offsets = np.asarray([unique[idx] for idx in ids], dtype=np.float64)
        nets.append({"ids": ids, "offsets": offsets})
    if not nets:
        raise RuntimeError(f"No macro-to-macro nets parsed from {path}")
    return nets


def read_grid(checkpoint_root: Path, benchmark: str, fallback: int) -> Tuple[int, str]:
    for path in sorted(checkpoint_root.rglob(f"{benchmark}.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if "grid" in payload:
                return int(payload["grid"]), str(path)
        except Exception:
            pass
    return int(fallback), "fallback"


NODE_DIMENSIONS = parse_nodes(BENCHMARK_BASE.with_suffix(".nodes"))
baseline_frame = pd.read_csv(X_CSV_PATH)
required_columns = {"macro", "x_center", "y_center", "width", "height"}
if not required_columns.issubset(baseline_frame.columns):
    raise RuntimeError(f"Missing checkpoint columns: {required_columns - set(baseline_frame.columns)}")
if baseline_frame["macro"].duplicated().any():
    raise RuntimeError("Duplicate macro names in checkpoint placement")

MACRO_NAMES = baseline_frame["macro"].astype(str).tolist()
NAME_TO_ID = {name: idx for idx, name in enumerate(MACRO_NAMES)}
if set(MACRO_NAMES) != set(NODE_DIMENSIONS):
    raise RuntimeError("Checkpoint and Bookshelf macro sets differ")

CENTERS_0 = baseline_frame[["x_center", "y_center"]].to_numpy(dtype=np.float64)
SIZES = baseline_frame[["width", "height"]].to_numpy(dtype=np.float64)
BOOKSHELF_SIZES = np.asarray([NODE_DIMENSIONS[name] for name in MACRO_NAMES], dtype=np.float64)
if not np.allclose(SIZES, BOOKSHELF_SIZES, rtol=0.0, atol=1e-6):
    raise RuntimeError("Checkpoint macro dimensions differ from Bookshelf .nodes")

ORIGINAL_POSITIONS = parse_original_pl(BENCHMARK_BASE.with_suffix(".pl"), set(MACRO_NAMES))
if len(ORIGINAL_POSITIONS) != len(MACRO_NAMES):
    raise RuntimeError(f"Parsed {len(ORIGINAL_POSITIONS)}/{len(MACRO_NAMES)} macro positions")
max_x = max(ORIGINAL_POSITIONS[name][0] + NODE_DIMENSIONS[name][0] for name in MACRO_NAMES)
max_y = max(ORIGINAL_POSITIONS[name][1] + NODE_DIMENSIONS[name][1] for name in MACRO_NAMES)
CANVAS_EXTENT = float(min(max_x, max_y))

NETS = parse_macro_nets(BENCHMARK_BASE.with_suffix(".nets"), NAME_TO_ID)
GRID, GRID_SOURCE = read_grid(checkpoint_search_root, CFG.benchmark, CFG.grid_fallback)
GRID_PITCH = CANVAS_EXTENT / GRID

TOTAL_NODES = parse_header_value(BENCHMARK_BASE.with_suffix(".nodes"), "NumNodes")
TOTAL_TERMINALS = parse_header_value(BENCHMARK_BASE.with_suffix(".nodes"), "NumTerminals")
TOTAL_NETS = parse_header_value(BENCHMARK_BASE.with_suffix(".nets"), "NumNets")
TOTAL_PINS = parse_header_value(BENCHMARK_BASE.with_suffix(".nets"), "NumPins")
METRIC_SCOPE = "macro-only proxy: macro-to-macro nets with >=2 macros"
FULL_CHIP_COMPARABLE = False

scope_table = pd.DataFrame([{
    "total_nodes": TOTAL_NODES,
    "terminal_macros": TOTAL_TERMINALS,
    "total_nets": TOTAL_NETS,
    "total_pins": TOTAL_PINS,
    "evaluated_macros": len(MACRO_NAMES),
    "evaluated_macro_nets": len(NETS),
    "full_chip_comparable": FULL_CHIP_COMPARABLE,
    "metric_scope": METRIC_SCOPE,
}])
display(scope_table)
print("Canvas extent:", CANVAS_EXTENT)
print("EfficientPlace grid:", GRID, "from", GRID_SOURCE)
print("SCIENTIFIC GUARD: these metrics are not full-chip placement/routing results.")
'''
    ),
    markdown_cell(
        r'''
## Định nghĩa congestion proxy

Với mỗi macro-net, các pin tạo thành bounding box. Nhu cầu dây được giả sử phân bố đều:

\[
D_n = \frac{HPWL_n}{\max(w_n, b)\max(h_n, b)}
\]

trong đó `b` là kích thước một bin. Tích phân demand trên bounding box bằng HPWL của net. Bề rộng/cao tối thiểu một bin tránh singularity.

Macro occupancy `u` được dùng để tạo capacity-adjusted proxy:

\[
D_{blocked} = \frac{D}{\max(c_{min}, 1 - \alpha u)}
\]

Đây là proxy nhất quán để so sánh các placement, **không phải global-routing overflow thực tế**.
'''
    ),
    code_cell(
        r'''
# Cell 4 — Unified HPWL, legality, occupancy và multi-resolution RUDY
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
        "overlap_pairs": overlap_pairs,
        "pairwise_overlap_area": overlap_area,
        "legal": bool(boundary_mask.sum() == 0 and overlap_pairs == 0),
    }


def add_rectangle_average(
    grid: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    density: float,
) -> None:
    resolution = grid.shape[0]
    pitch = CANVAS_EXTENT / resolution
    bin_area = pitch * pitch
    x0 = float(np.clip(x0, 0.0, CANVAS_EXTENT))
    y0 = float(np.clip(y0, 0.0, CANVAS_EXTENT))
    x1 = float(np.clip(x1, 0.0, CANVAS_EXTENT))
    y1 = float(np.clip(y1, 0.0, CANVAS_EXTENT))
    if x1 <= x0 or y1 <= y0:
        return
    ix0 = max(0, min(resolution - 1, int(math.floor(x0 / pitch))))
    iy0 = max(0, min(resolution - 1, int(math.floor(y0 / pitch))))
    ix1 = max(0, min(resolution - 1, int(math.ceil(x1 / pitch) - 1)))
    iy1 = max(0, min(resolution - 1, int(math.ceil(y1 / pitch) - 1)))
    for iy in range(iy0, iy1 + 1):
        by0, by1 = iy * pitch, (iy + 1) * pitch
        overlap_y = max(0.0, min(y1, by1) - max(y0, by0))
        if overlap_y == 0.0:
            continue
        for ix in range(ix0, ix1 + 1):
            bx0, bx1 = ix * pitch, (ix + 1) * pitch
            overlap_x = max(0.0, min(x1, bx1) - max(x0, bx0))
            if overlap_x:
                grid[iy, ix] += density * overlap_x * overlap_y / bin_area


def congestion_maps(centers: np.ndarray, resolution: int) -> Dict[str, np.ndarray]:
    pitch = CANVAS_EXTENT / resolution
    occupancy = np.zeros((resolution, resolution), dtype=np.float64)
    lower = centers - SIZES / 2.0
    upper = centers + SIZES / 2.0
    for idx in range(len(centers)):
        add_rectangle_average(
            occupancy,
            lower[idx, 0], lower[idx, 1], upper[idx, 0], upper[idx, 1], 1.0,
        )

    rudy = np.zeros_like(occupancy)
    for net in NETS:
        pins = centers[net["ids"]] + net["offsets"]
        xmin, ymin = pins.min(axis=0)
        xmax, ymax = pins.max(axis=0)
        hpwl = float((xmax - xmin) + (ymax - ymin))
        if hpwl <= 0.0:
            continue
        width = max(float(xmax - xmin), pitch)
        height = max(float(ymax - ymin), pitch)
        cx, cy = float((xmin + xmax) / 2.0), float((ymin + ymax) / 2.0)
        x0 = float(np.clip(cx - width / 2.0, 0.0, CANVAS_EXTENT - width))
        y0 = float(np.clip(cy - height / 2.0, 0.0, CANVAS_EXTENT - height))
        x1, y1 = x0 + width, y0 + height
        demand_density = hpwl / (width * height)
        add_rectangle_average(rudy, x0, y0, x1, y1, demand_density)

    available_capacity = np.clip(
        1.0 - CFG.blockage_weight * occupancy,
        CFG.capacity_floor,
        1.0,
    )
    blocked_rudy = rudy / available_capacity
    return {
        "macro_occupancy": occupancy,
        "rudy": rudy,
        "available_capacity": available_capacity,
        "blocked_rudy": blocked_rudy,
    }


def top_fraction_mean(values: np.ndarray, fraction: float) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    count = max(1, int(math.ceil(len(flat) * fraction)))
    return float(np.partition(flat, len(flat) - count)[-count:].mean())


def summarize_maps(maps: Dict[str, np.ndarray], resolution: int, pin_hpwl: float) -> Dict[str, float]:
    occupancy = maps["macro_occupancy"]
    rudy = maps["rudy"]
    blocked = maps["blocked_rudy"]
    bin_area = (CANVAS_EXTENT / resolution) ** 2
    integrated_rudy = float(rudy.sum() * bin_area)
    return {
        "macro_util_mean": float(occupancy.mean()),
        "macro_util_p95": float(np.quantile(occupancy, 0.95)),
        "macro_util_peak": float(occupancy.max()),
        "rudy_p95": float(np.quantile(rudy, 0.95)),
        "rudy_peak": float(rudy.max()),
        "rudy_top1pct_mean": top_fraction_mean(rudy, 0.01),
        "blocked_rudy_p95": float(np.quantile(blocked, 0.95)),
        "blocked_rudy_peak": float(blocked.max()),
        "blocked_rudy_top1pct_mean": top_fraction_mean(blocked, 0.01),
        "rudy_integral": integrated_rudy,
        "rudy_integral_error_pct": 100.0 * (integrated_rudy / pin_hpwl - 1.0),
    }


def evaluate_placement(
    centers: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, np.ndarray]], List[Dict[str, Any]]]:
    pin_hpwl = hpwl_pin(centers)
    base_metrics: Dict[str, Any] = {
        "hpwl_center": hpwl_center(centers),
        "hpwl_pin": pin_hpwl,
        "hpwl_efficientplace_proxy": hpwl_efficientplace_proxy(centers),
    }
    base_metrics.update(legality_metrics(centers))
    maps_by_resolution: Dict[int, Dict[str, np.ndarray]] = {}
    resolution_rows: List[Dict[str, Any]] = []
    for resolution in CFG.rudy_resolutions:
        maps = congestion_maps(centers, int(resolution))
        maps_by_resolution[int(resolution)] = maps
        resolution_rows.append({
            "resolution": int(resolution),
            **summarize_maps(maps, int(resolution), pin_hpwl),
        })
    primary = next(row for row in resolution_rows if row["resolution"] == CFG.primary_resolution)
    base_metrics.update({key: value for key, value in primary.items() if key != "resolution"})
    return base_metrics, maps_by_resolution, resolution_rows
'''
    ),
    code_cell(
        r'''
# Cell 5 — Load baseline + 5 seed, audit Stage 1 và chạy evaluator
def load_named_placement(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if not required_columns.issubset(frame.columns):
        raise RuntimeError(f"Missing placement columns in {path}")
    if frame["macro"].duplicated().any():
        raise RuntimeError(f"Duplicate macro names in {path}")
    indexed = frame.set_index(frame["macro"].astype(str))
    if set(indexed.index) != set(MACRO_NAMES):
        raise RuntimeError(f"Macro set mismatch in {path}")
    aligned = indexed.loc[MACRO_NAMES]
    sizes = aligned[["width", "height"]].to_numpy(dtype=np.float64)
    if not np.allclose(sizes, SIZES, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"Macro dimension mismatch in {path}")
    return aligned[["x_center", "y_center"]].to_numpy(dtype=np.float64)


stage1_results = pd.read_csv(STAGE1_RESULTS_PATH)
stage1_results["seed"] = stage1_results["seed"].astype(int)
if sorted(stage1_results["seed"].tolist()) != sorted(CFG.expected_seeds):
    raise AssertionError("Stage 1 per_seed_results.csv does not contain the expected seeds")

placements: Dict[str, Dict[str, Any]] = {
    "baseline": {"kind": "baseline", "seed": None, "centers": CENTERS_0, "source": X_CSV_PATH}
}
for seed in CFG.expected_seeds:
    path = STAGE1_ROOT / f"seed_{seed}" / "refined_centers.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 1 placement: {path}")
    placements[f"seed_{seed}"] = {
        "kind": "refined",
        "seed": int(seed),
        "centers": load_named_placement(path),
        "source": path,
    }

metric_rows = []
resolution_rows = []
MAPS: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
for placement_id, payload in placements.items():
    metrics, maps_by_resolution, per_resolution = evaluate_placement(payload["centers"])
    if not metrics["legal"]:
        raise AssertionError(f"Illegal placement detected: {placement_id}: {metrics}")
    MAPS[placement_id] = maps_by_resolution
    metric_rows.append({
        "placement_id": placement_id,
        "kind": payload["kind"],
        "seed": payload["seed"],
        "source": str(payload["source"]),
        **metrics,
    })
    for row in per_resolution:
        resolution_rows.append({
            "placement_id": placement_id,
            "kind": payload["kind"],
            "seed": payload["seed"],
            **row,
        })

METRICS = pd.DataFrame(metric_rows)
RESOLUTION_METRICS = pd.DataFrame(resolution_rows)

baseline_metrics = METRICS.loc[METRICS["placement_id"] == "baseline"].iloc[0]
manifest_baseline = STAGE1_MANIFEST.get("baseline_metrics", {})
if manifest_baseline:
    if not math.isclose(
        float(baseline_metrics["hpwl_pin"]),
        float(manifest_baseline["hpwl_pin"]),
        rel_tol=0.0,
        abs_tol=CFG.metric_tolerance,
    ):
        raise AssertionError("Baseline pin-HPWL does not reproduce Stage 1 manifest")

for _, stage1_row in stage1_results.iterrows():
    placement_id = f"seed_{int(stage1_row['seed'])}"
    evaluated = METRICS.loc[METRICS["placement_id"] == placement_id, "hpwl_pin"].iloc[0]
    if not math.isclose(
        float(evaluated), float(stage1_row["final_hpwl_pin"]),
        rel_tol=0.0, abs_tol=CFG.metric_tolerance,
    ):
        raise AssertionError(f"HPWL mismatch for {placement_id}")

if float(RESOLUTION_METRICS["rudy_integral_error_pct"].abs().max()) > 1e-8:
    raise AssertionError("RUDY integral sanity check failed")

METRICS.to_csv(RUN_DIR / "placement_metrics.csv", index=False)
RESOLUTION_METRICS.to_csv(RUN_DIR / "resolution_metrics.csv", index=False)
display(METRICS[[
    "placement_id", "hpwl_pin", "hpwl_center", "hpwl_efficientplace_proxy",
    "blocked_rudy_top1pct_mean", "blocked_rudy_peak",
    "overlap_pairs", "boundary_violations", "legal",
]].sort_values("hpwl_pin"))
print("AUDIT PASSED: all six placements reproduce Stage 1 and remain legal.")
'''
    ),
    code_cell(
        r'''
# Cell 6 — Delta, thống kê 5 seed, Pareto front và robustness theo resolution
OBJECTIVE_X = "hpwl_pin"
OBJECTIVE_Y = "blocked_rudy_top1pct_mean"


def pareto_mask(frame: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray:
    values = frame[[x_col, y_col]].to_numpy(dtype=np.float64)
    keep = np.ones(len(values), dtype=bool)
    tolerance = 1e-15
    for idx, point in enumerate(values):
        weakly_better = np.all(values <= point + tolerance, axis=1)
        strictly_better = np.any(values < point - tolerance, axis=1)
        dominated = weakly_better & strictly_better
        dominated[idx] = False
        keep[idx] = not np.any(dominated)
    return keep


baseline = METRICS.loc[METRICS["placement_id"] == "baseline"].iloc[0]
SUMMARY = METRICS.copy()
SUMMARY["hpwl_pin_improvement_pct"] = 100.0 * (1.0 - SUMMARY["hpwl_pin"] / baseline["hpwl_pin"])
SUMMARY["hpwl_center_improvement_pct"] = 100.0 * (
    1.0 - SUMMARY["hpwl_center"] / baseline["hpwl_center"]
)
SUMMARY["proxy_improvement_pct"] = 100.0 * (
    1.0 - SUMMARY["hpwl_efficientplace_proxy"] / baseline["hpwl_efficientplace_proxy"]
)
for metric in (
    "rudy_p95", "rudy_peak", "rudy_top1pct_mean",
    "blocked_rudy_p95", "blocked_rudy_peak", "blocked_rudy_top1pct_mean",
    "macro_util_p95", "macro_util_peak",
):
    SUMMARY[f"{metric}_delta_pct"] = 100.0 * (SUMMARY[metric] / baseline[metric] - 1.0)

SUMMARY["pareto_primary"] = pareto_mask(SUMMARY, OBJECTIVE_X, OBJECTIVE_Y)
SUMMARY["dominates_baseline"] = (
    (SUMMARY[OBJECTIVE_X] <= baseline[OBJECTIVE_X] + CFG.metric_tolerance)
    & (SUMMARY[OBJECTIVE_Y] <= baseline[OBJECTIVE_Y] + 1e-15)
    & (
        (SUMMARY[OBJECTIVE_X] < baseline[OBJECTIVE_X] - CFG.metric_tolerance)
        | (SUMMARY[OBJECTIVE_Y] < baseline[OBJECTIVE_Y] - 1e-15)
    )
)

refined = SUMMARY.loc[SUMMARY["kind"] == "refined"].copy()
aggregate_rows = []
for metric in (
    "hpwl_pin_improvement_pct",
    "hpwl_center_improvement_pct",
    "proxy_improvement_pct",
    "blocked_rudy_top1pct_mean_delta_pct",
    "blocked_rudy_peak_delta_pct",
):
    values = refined[metric].to_numpy(dtype=np.float64)
    aggregate_rows.append({
        "metric": metric,
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    })
AGGREGATE = pd.DataFrame(aggregate_rows)

sensitivity = RESOLUTION_METRICS.pivot(
    index="placement_id", columns="resolution", values="blocked_rudy_top1pct_mean"
)
SPEARMAN_RESOLUTION = sensitivity.corr(method="spearman")
CORRELATIONS = refined[[
    "hpwl_pin", "hpwl_center", "hpwl_efficientplace_proxy",
    "rudy_top1pct_mean", "blocked_rudy_top1pct_mean", "blocked_rudy_peak",
]].corr(method="spearman")

SUMMARY.to_csv(RUN_DIR / "placement_summary.csv", index=False)
AGGREGATE.to_csv(RUN_DIR / "aggregate_seed_statistics.csv", index=False)
SPEARMAN_RESOLUTION.to_csv(RUN_DIR / "resolution_spearman.csv")
CORRELATIONS.to_csv(RUN_DIR / "metric_spearman.csv")
SUMMARY.loc[SUMMARY["pareto_primary"]].to_csv(RUN_DIR / "pareto_front.csv", index=False)

display(SUMMARY[[
    "placement_id", "hpwl_pin_improvement_pct",
    "blocked_rudy_top1pct_mean_delta_pct", "blocked_rudy_peak_delta_pct",
    "pareto_primary", "dominates_baseline",
]].sort_values("hpwl_pin_improvement_pct", ascending=False))
display(AGGREGATE)
display(SPEARMAN_RESOLUTION)

BEST_HPWL_ID = str(SUMMARY.loc[SUMMARY["hpwl_pin"].idxmin(), "placement_id"])
BEST_CONGESTION_ID = str(SUMMARY.loc[SUMMARY[OBJECTIVE_Y].idxmin(), "placement_id"])
PARETO_IDS = SUMMARY.loc[SUMMARY["pareto_primary"], "placement_id"].tolist()
print("Best HPWL       :", BEST_HPWL_ID)
print("Best congestion :", BEST_CONGESTION_ID)
print("Pareto front    :", PARETO_IDS)
'''
    ),
    code_cell(
        r'''
# Cell 7 — Pareto plot, metric deltas, heatmaps và resolution sensitivity
figure, axes = plt.subplots(2, 2, figsize=(16, 13))

# Pareto plane: lower-left is better.
axis = axes[0, 0]
for _, row in SUMMARY.iterrows():
    color = "tab:red" if row["placement_id"] == "baseline" else (
        "tab:green" if row["pareto_primary"] else "tab:blue"
    )
    axis.scatter(row[OBJECTIVE_X], row[OBJECTIVE_Y], s=75, color=color)
    axis.annotate(row["placement_id"], (row[OBJECTIVE_X], row[OBJECTIVE_Y]), xytext=(5, 5),
                  textcoords="offset points", fontsize=8)
axis.set_xlabel("Pin-aware macro-only HPWL ↓")
axis.set_ylabel("Capacity-adjusted RUDY top-1% mean ↓")
axis.set_title(f"Primary Pareto plane ({CFG.primary_resolution}×{CFG.primary_resolution})")
axis.grid(alpha=0.25)

# Relative deltas against EfficientPlace baseline.
axis = axes[0, 1]
ordered = SUMMARY.loc[SUMMARY["kind"] == "refined"].sort_values("seed")
x = np.arange(len(ordered))
width = 0.36
axis.bar(x - width / 2, ordered["hpwl_pin_improvement_pct"], width, label="HPWL improvement (+ better)")
axis.bar(
    x + width / 2,
    -ordered["blocked_rudy_top1pct_mean_delta_pct"],
    width,
    label="Congestion improvement (+ better)",
)
axis.axhline(0.0, color="black", linewidth=0.8)
axis.set_xticks(x, ordered["placement_id"])
axis.set_ylabel("Improvement versus baseline (%)")
axis.set_title("HPWL–congestion trade-off by seed")
axis.legend()
axis.grid(axis="y", alpha=0.25)

# Resolution sensitivity.
axis = axes[1, 0]
for placement_id in sensitivity.index:
    values = sensitivity.loc[placement_id]
    normalized = values / sensitivity.loc["baseline"]
    axis.plot(values.index, normalized, marker="o", label=placement_id)
axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
axis.set_xlabel("RUDY grid resolution")
axis.set_ylabel("Blocked RUDY top-1% / baseline")
axis.set_title("Resolution robustness")
axis.grid(alpha=0.25)
axis.legend(fontsize=8)

# Mean result with min/max range.
axis = axes[1, 1]
hpwl_values = ordered["hpwl_pin_improvement_pct"].to_numpy(dtype=float)
congestion_values = -ordered["blocked_rudy_top1pct_mean_delta_pct"].to_numpy(dtype=float)
means = [hpwl_values.mean(), congestion_values.mean()]
lower = [means[0] - hpwl_values.min(), means[1] - congestion_values.min()]
upper = [hpwl_values.max() - means[0], congestion_values.max() - means[1]]
axis.bar([0, 1], means, color=["tab:blue", "tab:orange"], alpha=0.8)
axis.errorbar([0, 1], means, yerr=[lower, upper], fmt="none", color="black", capsize=5)
axis.axhline(0.0, color="black", linewidth=0.8)
axis.set_xticks([0, 1], ["Pin HPWL", "Blocked RUDY top-1%"])
axis.set_ylabel("Mean improvement versus baseline (%)")
axis.set_title("Five-seed mean with min–max")
axis.grid(axis="y", alpha=0.25)

figure.tight_layout()
summary_plot_path = RUN_DIR / "pareto_and_robustness.png"
figure.savefig(summary_plot_path, dpi=180, bbox_inches="tight")
plt.show()


heatmap_ids = list(dict.fromkeys(["baseline", BEST_HPWL_ID, BEST_CONGESTION_ID]))
figure, heat_axes = plt.subplots(1, len(heatmap_ids), figsize=(6 * len(heatmap_ids), 5), squeeze=False)
all_heatmaps = [MAPS[item][CFG.primary_resolution]["blocked_rudy"] for item in heatmap_ids]
vmax = float(np.quantile(np.concatenate([item.ravel() for item in all_heatmaps]), 0.995))
for axis, placement_id, heatmap in zip(heat_axes[0], heatmap_ids, all_heatmaps):
    image = axis.imshow(heatmap, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
    axis.set_title(placement_id)
    axis.set_xlabel("bin x")
    axis.set_ylabel("bin y")
figure.colorbar(image, ax=heat_axes.ravel().tolist(), shrink=0.78, label="capacity-adjusted RUDY")
figure.suptitle(f"Primary blocked-RUDY maps ({CFG.primary_resolution}×{CFG.primary_resolution})")
heatmap_path = RUN_DIR / "blocked_rudy_heatmaps.png"
figure.savefig(heatmap_path, dpi=180, bbox_inches="tight")
plt.show()

print("Summary plot:", summary_plot_path)
print("Heatmaps    :", heatmap_path)
'''
    ),
    code_cell(
        r'''
# Cell 8 — Xuất manifest, báo cáo và archive tái lập
def format_number(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


hpwl_stats = AGGREGATE.loc[AGGREGATE["metric"] == "hpwl_pin_improvement_pct"].iloc[0]
congestion_stats = AGGREGATE.loc[
    AGGREGATE["metric"] == "blocked_rudy_top1pct_mean_delta_pct"
].iloc[0]
dominators = SUMMARY.loc[SUMMARY["dominates_baseline"], "placement_id"].tolist()

report_lines = [
    "# Stage 2A evaluation report — bigblue1",
    "",
    f"- Metric scope: `{METRIC_SCOPE}`",
    f"- Full-chip comparable: `{FULL_CHIP_COMPARABLE}`",
    f"- Macro count: `{len(MACRO_NAMES)}`",
    f"- Evaluated macro-net count: `{len(NETS)}`",
    f"- Primary RUDY grid: `{CFG.primary_resolution}×{CFG.primary_resolution}`",
    "",
    "## Five-seed Stage 1 statistics",
    "",
    f"- Mean pin-HPWL improvement: `{format_number(hpwl_stats['mean'])}%`",
    f"- Sample standard deviation: `{format_number(hpwl_stats['sample_std'])}%`",
    f"- Range: `{format_number(hpwl_stats['min'])}%` to `{format_number(hpwl_stats['max'])}%`",
    f"- Mean blocked-RUDY top-1% delta: `{format_number(congestion_stats['mean'])}%` "
    "(negative means improvement)",
    f"- Best HPWL placement: `{BEST_HPWL_ID}`",
    f"- Best congestion placement: `{BEST_CONGESTION_ID}`",
    f"- Primary Pareto front: `{', '.join(PARETO_IDS)}`",
    f"- Placements dominating baseline on both objectives: `{', '.join(dominators) if dominators else 'none'}`",
    "",
    "## Interpretation guard",
    "",
    "These results evaluate macro-to-macro HPWL and a deterministic RUDY proxy. "
    "They are not routed congestion, timing, or downstream full-chip HPWL.",
]
report_path = RUN_DIR / "stage2a_report.md"
report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

manifest = {
    "experiment": "stage2a_pareto_evaluator",
    "config": asdict(CFG),
    "metric_scope": METRIC_SCOPE,
    "full_chip_comparable": FULL_CHIP_COMPARABLE,
    "checkpoint_x_csv": str(X_CSV_PATH),
    "checkpoint_x_csv_sha256": actual_checkpoint_hash,
    "benchmark_base": str(BENCHMARK_BASE),
    "benchmark_hashes": {
        suffix: sha256_file(BENCHMARK_BASE.with_suffix(suffix))
        for suffix in (".nodes", ".nets", ".pl", ".aux")
    },
    "stage1_root": str(STAGE1_ROOT),
    "stage1_manifest": str(STAGE1_MANIFEST_PATH),
    "stage1_manifest_sha256": sha256_file(STAGE1_MANIFEST_PATH),
    "macro_count": len(MACRO_NAMES),
    "evaluated_macro_net_count": len(NETS),
    "best_hpwl_placement": BEST_HPWL_ID,
    "best_congestion_placement": BEST_CONGESTION_ID,
    "pareto_placements": PARETO_IDS,
    "dominates_baseline": dominators,
    "all_placements_legal": bool(SUMMARY["legal"].all()),
    "rudy_integral_sanity_max_abs_error_pct": float(
        RESOLUTION_METRICS["rudy_integral_error_pct"].abs().max()
    ),
}
manifest_path = RUN_DIR / "run_manifest_stage2a.json"
manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")

archive_path = shutil.make_archive(str(RUN_DIR), "zip", root_dir=RUN_DIR)
print(report_path.read_text(encoding="utf-8"))
print("Manifest:", manifest_path)
print("Archive :", archive_path)
'''
    ),
    code_cell(
        r'''
# Cell 9 — Final scientific gate và hành động tiếp theo
assert len(SUMMARY) == 1 + len(CFG.expected_seeds)
assert bool(SUMMARY["legal"].all())
assert int(SUMMARY["overlap_pairs"].sum()) == 0
assert int(SUMMARY["boundary_violations"].sum()) == 0
assert float(RESOLUTION_METRICS["rudy_integral_error_pct"].abs().max()) <= 1e-8
assert not FULL_CHIP_COMPARABLE

resolution_min_correlation = float(
    SPEARMAN_RESOLUTION.to_numpy()[np.triu_indices_from(SPEARMAN_RESOLUTION, k=1)].min()
)
final_summary = {
    "mean_hpwl_improvement_pct": float(hpwl_stats["mean"]),
    "std_hpwl_improvement_pct": float(hpwl_stats["sample_std"]),
    "mean_blocked_rudy_top1pct_delta_pct": float(congestion_stats["mean"]),
    "best_hpwl_placement": BEST_HPWL_ID,
    "best_congestion_placement": BEST_CONGESTION_ID,
    "pareto_placements": PARETO_IDS,
    "placements_dominating_baseline": dominators,
    "minimum_resolution_rank_correlation": resolution_min_correlation,
    "full_chip_comparable": FULL_CHIP_COMPARABLE,
    "archive": str(archive_path),
}
print(json.dumps(final_summary, indent=2))
print("\nNEXT DECISION:")
if dominators:
    print("At least one refined placement improves both HPWL and the congestion proxy.")
    print("Use the Pareto placements as initial states for Stage 2B multi-objective refinement.")
else:
    print("No refined placement dominates the EfficientPlace baseline on both objectives.")
    print("Stage 2B must explicitly optimize HPWL and congestion; do not select by HPWL alone.")
print("In both cases, a downstream standard-cell placer is still required before paper-level comparison.")
print("Send run_manifest_stage2a.json, placement_summary.csv, and the two PNG figures for audit.")
'''
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {DESTINATION}")

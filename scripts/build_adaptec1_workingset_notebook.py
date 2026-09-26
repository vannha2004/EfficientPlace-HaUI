from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cplex_resolver" / "adaptec1_drl_pdca_cplex.ipynb"
DESTINATION = ROOT / "cplex_resolver" / "adaptec1_drl_pdca_cplex_workingset.ipynb"


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
# adaptec1 — DRL + proximal DCA + CPLEX working set

Notebook này sửa các vấn đề chính của bản CPLEX trước:

- không dựng full-pair trong DOcplex; full-pair chỉ được đánh giá bằng NumPy;
- working set được giữ lại và bổ sung bằng separation loop trước khi nhận nghiệm;
- nghiệm làm tăng full objective bị từ chối hoặc backtracking;
- subgradient có deterministic symmetry breaking khi hai macro trùng tâm;
- HPWL dùng pin offsets giống reward của DRL;
- tọa độ local/global được tách riêng khi ghi Bookshelf `.pl`;
- chạy adaptive penalty continuation và dùng relative stopping criterion.

Notebook cũ được giữ nguyên để đối chiếu. Các tham số ngân sách nằm trong Cell 1.
'''
    ),
    code_cell(
        r'''
# Cell 1 - Imports, reproducible configuration, and RAM-safe defaults
from __future__ import annotations

import csv
import json
import math
import os
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Compatibility for older DOcplex releases on NumPy 2.x.
if not hasattr(np, "float_"):
    np.float_ = np.float64

BENCHMARK = "adaptec1"
EXPECTED_MACROS = 543
GRID_SIZE_FALLBACK = 32

# Penalty continuation. Gamma stays fixed inside each DCA stage.
INITIAL_GAMMA_BOUNDARY = 100.0
INITIAL_GAMMA_OVERLAP = 10_000.0
PENALTY_GROWTH = 2.0
MAX_PENALTY_STAGES = 4

RHO = 1e-3
MAX_DCA_ITER_PER_STAGE = 50
RELATIVE_STEP_TOL = 1e-6

# RAM-safe CPLEX working-set controls.
ACTIVE_SET_MARGIN = 1.0
MAX_WORKING_SET_REFINEMENTS = 8
BACKTRACK_STEPS = 12
TRUST_RADIUS_FRACTION = 0.08
CPLEX_THREADS = 4
MONOTONE_RTOL = 1e-8

# Stable nonzero subgradient when two centers coincide.
ZERO_DISPLACEMENT_TOL = 1e-7
BRANCH_TIE_TOL = 1e-12

print("Configuration loaded")
print({
    "benchmark": BENCHMARK,
    "initial_gamma_boundary": INITIAL_GAMMA_BOUNDARY,
    "initial_gamma_overlap": INITIAL_GAMMA_OVERLAP,
    "rho": RHO,
    "max_penalty_stages": MAX_PENALTY_STAGES,
    "max_dca_iter_per_stage": MAX_DCA_ITER_PER_STAGE,
    "active_set_margin": ACTIVE_SET_MARGIN,
    "cplex_threads": CPLEX_THREADS,
})
'''
    ),
    code_cell(
        r'''
# Cell 2 - Resolve repository paths; never silently read a stale external folder

def locate_cplex_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        cwd / "cplex_resolver",
        cwd.parent / "cplex_resolver",
    ]
    for candidate in candidates:
        if (
            (candidate / "ISPD2005" / BENCHMARK / f"{BENCHMARK}.nodes").is_file()
            and (candidate / "results" / BENCHMARK / "placement_grid.npy").is_file()
        ):
            return candidate
    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Cannot locate cplex_resolver from the current working directory. Checked:\n"
        + checked
    )


PROJECT_ROOT = locate_cplex_root()
BENCH_DIR = PROJECT_ROOT / "ISPD2005" / BENCHMARK
DRL_RESULTS_DIR = PROJECT_ROOT / "results" / BENCHMARK
PLACEMENT_GRID_PATH = DRL_RESULTS_DIR / "placement_grid.npy"

timestamp = time.strftime("%Y%m%d_%H%M%S")
EXP_DIR = (
    PROJECT_ROOT
    / "experiments"
    / BENCHMARK
    / "workingset_pin_hpwl"
    / f"run_{timestamp}"
)
CONFIG_DIR = EXP_DIR / "config"
LOGS_DIR = EXP_DIR / "logs"
METRICS_DIR = EXP_DIR / "metrics"
PLACEMENTS_DIR = EXP_DIR / "placements"
PLOTS_DIR = EXP_DIR / "plots"
CPLEX_MODELS_DIR = EXP_DIR / "cplex_models"
for folder in [
    CONFIG_DIR, LOGS_DIR, METRICS_DIR, PLACEMENTS_DIR, PLOTS_DIR, CPLEX_MODELS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOGS_DIR / "run.log"


def log_msg(message: str) -> None:
    formatted = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(formatted)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(formatted + "\n")


try:
    import cplex
    import docplex
    from docplex.mp.model import Model
except Exception as exc:
    raise RuntimeError(f"CPLEX/DOcplex is unavailable: {exc}") from exc

cpx_probe = cplex.Cplex()
CPLEX_VERSION = cpx_probe.get_version()
del cpx_probe

GRID_SIZE = GRID_SIZE_FALLBACK
drl_config_path = DRL_RESULTS_DIR / "config.json"
if drl_config_path.is_file():
    with drl_config_path.open("r", encoding="utf-8") as handle:
        drl_config = json.load(handle)
    GRID_SIZE = int(drl_config.get("grid_size", GRID_SIZE_FALLBACK))
else:
    drl_config = {}

drl_metrics_path = DRL_RESULTS_DIR / "metrics.json"
drl_metrics = {}
if drl_metrics_path.is_file():
    with drl_metrics_path.open("r", encoding="utf-8") as handle:
        drl_metrics = json.load(handle)

experiment_config = {
    "benchmark": BENCHMARK,
    "project_root": str(PROJECT_ROOT),
    "placement_grid": str(PLACEMENT_GRID_PATH),
    "drl_seed": drl_metrics.get("seed"),
    "grid_size": GRID_SIZE,
    "initial_gamma_boundary": INITIAL_GAMMA_BOUNDARY,
    "initial_gamma_overlap": INITIAL_GAMMA_OVERLAP,
    "penalty_growth": PENALTY_GROWTH,
    "max_penalty_stages": MAX_PENALTY_STAGES,
    "rho": RHO,
    "max_dca_iter_per_stage": MAX_DCA_ITER_PER_STAGE,
    "relative_step_tol": RELATIVE_STEP_TOL,
    "active_set_margin": ACTIVE_SET_MARGIN,
    "max_working_set_refinements": MAX_WORKING_SET_REFINEMENTS,
    "trust_radius_fraction": TRUST_RADIUS_FRACTION,
    "cplex_threads": CPLEX_THREADS,
    "cplex_version": CPLEX_VERSION,
    "docplex_version": docplex.__version__,
}
with (CONFIG_DIR / "experiment_config.json").open("w", encoding="utf-8") as handle:
    json.dump(experiment_config, handle, indent=2)

log_msg(f"PROJECT_ROOT={PROJECT_ROOT}")
log_msg(f"DRL seed={drl_metrics.get('seed', '<unknown>')}, GRID_SIZE={GRID_SIZE}")
log_msg(f"CPLEX={CPLEX_VERSION}, DOcplex={docplex.__version__}")
log_msg(f"Experiment output={EXP_DIR}")
'''
    ),
    code_cell(
        r'''
# Cell 3 - Parse Bookshelf and preserve every macro pin offset

@dataclass
class Macro:
    name: str
    width: float
    height: float
    initial_x: float
    initial_y: float
    fixed: bool


@dataclass
class MacroNet:
    macro_ids: np.ndarray
    pin_macro_ids: np.ndarray
    pin_offsets: np.ndarray


@dataclass
class PlacementDB:
    benchmark: str
    macros: List[Macro]
    macro_name_to_id: Dict[str, int]
    nets: List[MacroNet]
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    total_nodes: int
    total_nets: int

    @property
    def num_macros(self) -> int:
        return len(self.macros)

    @property
    def canvas_width(self) -> float:
        return self.x_max - self.x_min

    @property
    def canvas_height(self) -> float:
        return self.y_max - self.y_min


def parse_nodes(path: Path):
    nodes = {}
    total_nodes = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("NumNodes"):
                total_nodes = int(line.split(":", 1)[1])
                continue
            if not line or line.startswith(("#", "UCLA", "NumTerminals")):
                continue
            fields = line.split()
            if len(fields) < 3:
                continue
            try:
                width, height = float(fields[1]), float(fields[2])
            except ValueError:
                continue
            terminal = any(token.lower().startswith("terminal") for token in fields[3:])
            nodes[fields[0]] = (width, height, terminal)
    if total_nodes is not None and len(nodes) != total_nodes:
        raise ValueError(f"{path}: declared {total_nodes} nodes, parsed {len(nodes)}")
    return nodes, int(total_nodes or len(nodes))


def parse_pl(path: Path):
    positions = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith(("#", "UCLA")):
                continue
            fields = line.split()
            if len(fields) < 3:
                continue
            try:
                positions[fields[0]] = (
                    float(fields[1]), float(fields[2]), "/FIXED" in line.upper()
                )
            except ValueError:
                continue
    return positions


def parse_scl(path: Path):
    x_min = y_min = float("inf")
    x_max = y_max = -float("inf")
    coordinate, height, site_width = 0.0, 0.0, 1.0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("Coordinate"):
                coordinate = float(line.split(":", 1)[1])
            elif line.startswith("Height"):
                height = float(line.split(":", 1)[1])
            elif line.startswith("Sitewidth"):
                site_width = float(line.split(":", 1)[1])
            elif line.startswith("SubrowOrigin"):
                values = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?", line)]
                if len(values) >= 2:
                    origin, num_sites = values[:2]
                    x_min = min(x_min, origin)
                    y_min = min(y_min, coordinate)
                    x_max = max(x_max, origin + num_sites * site_width)
                    y_max = max(y_max, coordinate + height)
    bounds = np.asarray([x_min, y_min, x_max, y_max], dtype=np.float64)
    if not np.isfinite(bounds).all() or x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid placement canvas in {path}")
    return x_min, y_min, x_max, y_max


def parse_macro_nets(path: Path, macro_name_to_id: Dict[str, int]):
    nets: List[MacroNet] = []
    total_nets = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        iterator = iter(handle)
        for raw in iterator:
            line = raw.strip()
            if line.startswith("NumNets"):
                total_nets = int(line.split(":", 1)[1])
            if not line.startswith("NetDegree"):
                continue
            match = re.search(r"NetDegree\s*:\s*(\d+)", line)
            if not match:
                continue
            pin_ids, offsets = [], []
            for _ in range(int(match.group(1))):
                try:
                    pin_line = next(iterator).strip()
                except StopIteration as exc:
                    raise ValueError(f"Unexpected EOF in {path}") from exc
                fields = pin_line.split()
                if not fields or fields[0] not in macro_name_to_id:
                    continue
                numbers = re.findall(
                    r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
                    pin_line.split(":", 1)[-1],
                )
                dx, dy = (
                    (float(numbers[-2]), float(numbers[-1]))
                    if len(numbers) >= 2 else (0.0, 0.0)
                )
                pin_ids.append(macro_name_to_id[fields[0]])
                offsets.append((dx, dy))
            unique_ids = list(dict.fromkeys(pin_ids))
            if len(unique_ids) >= 2:
                nets.append(MacroNet(
                    macro_ids=np.asarray(unique_ids, dtype=np.int64),
                    pin_macro_ids=np.asarray(pin_ids, dtype=np.int64),
                    pin_offsets=np.asarray(offsets, dtype=np.float64),
                ))
    return nets, total_nets


def build_db() -> PlacementDB:
    base = BENCH_DIR / BENCHMARK
    nodes, total_nodes = parse_nodes(base.with_suffix(".nodes"))
    positions = parse_pl(base.with_suffix(".pl"))
    macro_names = [name for name, (_, _, terminal) in nodes.items() if terminal]
    macros = []
    for name in macro_names:
        width, height, _ = nodes[name]
        x, y, fixed = positions.get(name, (0.0, 0.0, False))
        macros.append(Macro(name, width, height, x, y, fixed))
    name_to_id = {macro.name: idx for idx, macro in enumerate(macros)}
    nets, total_nets = parse_macro_nets(base.with_suffix(".nets"), name_to_id)
    x_min, y_min, x_max, y_max = parse_scl(base.with_suffix(".scl"))
    return PlacementDB(
        BENCHMARK, macros, name_to_id, nets,
        x_min, y_min, x_max, y_max, total_nodes, total_nets,
    )


DB = build_db()
if DB.num_macros != EXPECTED_MACROS:
    raise ValueError(f"Expected {EXPECTED_MACROS} macros, parsed {DB.num_macros}")
if DB.num_macros > GRID_SIZE ** 2:
    raise ValueError(f"{DB.num_macros} macros do not fit a {GRID_SIZE}x{GRID_SIZE} grid")

WIDTHS = np.asarray([m.width for m in DB.macros], dtype=np.float64)
HEIGHTS = np.asarray([m.height for m in DB.macros], dtype=np.float64)
if np.any(WIDTHS <= 0) or np.any(HEIGHTS <= 0):
    raise ValueError("Every macro must have positive width and height")

log_msg(
    f"DB: macros={DB.num_macros}, macro_nets={len(DB.nets)}, "
    f"total_nets={DB.total_nets}, canvas_local={DB.canvas_width}x{DB.canvas_height}, "
    f"origin=({DB.x_min},{DB.y_min})"
)
'''
    ),
    code_cell(
        r'''
# Cell 4 - Load DRL placement and map it to local paper-center coordinates

def validate_placement_grid(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    if grid.shape != (DB.num_macros, 2):
        raise ValueError(f"Expected {(DB.num_macros, 2)}, got {grid.shape}")
    if not np.isfinite(grid).all():
        raise ValueError("placement_grid contains NaN/Inf")
    rounded = np.rint(grid)
    if not np.allclose(grid, rounded, atol=1e-6):
        raise ValueError("placement_grid must contain integer grid coordinates")
    rounded = rounded.astype(np.int64)
    if np.any(rounded < 0) or np.any(rounded >= GRID_SIZE):
        raise ValueError(f"placement_grid is outside [0,{GRID_SIZE - 1}]")
    return rounded


def grid_to_local_centers(grid: np.ndarray) -> np.ndarray:
    grid = validate_placement_grid(grid)
    lower_left_local = np.empty_like(grid, dtype=np.float64)
    lower_left_local[:, 0] = grid[:, 0] * DB.canvas_width / GRID_SIZE
    lower_left_local[:, 1] = grid[:, 1] * DB.canvas_height / GRID_SIZE
    return lower_left_local + np.column_stack((WIDTHS, HEIGHTS)) / 2.0


RAW_GRID = validate_placement_grid(np.load(PLACEMENT_GRID_PATH))
Z0_PAPER_CENTERS = grid_to_local_centers(RAW_GRID)
np.save(PLACEMENTS_DIR / "z0_grid.npy", RAW_GRID)
np.save(PLACEMENTS_DIR / "z0_paper_centers.npy", Z0_PAPER_CENTERS)

log_msg(
    f"z0 range: x=[{Z0_PAPER_CENTERS[:,0].min():.3f},"
    f"{Z0_PAPER_CENTERS[:,0].max():.3f}], "
    f"y=[{Z0_PAPER_CENTERS[:,1].min():.3f},"
    f"{Z0_PAPER_CENTERS[:,1].max():.3f}]"
)
'''
    ),
    code_cell(
        r'''
# Cell 5 - Full NumPy audit metrics; HPWL uses actual macro pin offsets

PAIR_I, PAIR_J = np.triu_indices(DB.num_macros, k=1)
PAIR_I = PAIR_I.astype(np.int64)
PAIR_J = PAIR_J.astype(np.int64)
PAIR_W = (WIDTHS[PAIR_I] + WIDTHS[PAIR_J]) / 2.0
PAIR_H = (HEIGHTS[PAIR_I] + HEIGHTS[PAIR_J]) / 2.0
PAIR_KEYS = PAIR_I * DB.num_macros + PAIR_J
PAIR_KEY_TO_INDEX = {int(key): idx for idx, key in enumerate(PAIR_KEYS)}


def pin_hpwl(centers_local: np.ndarray) -> float:
    total = 0.0
    for net in DB.nets:
        pins = centers_local[net.pin_macro_ids] + net.pin_offsets
        total += np.ptp(pins[:, 0]) + np.ptp(pins[:, 1])
    return float(total)


def center_hpwl_audit(centers_local: np.ndarray) -> float:
    total = 0.0
    for net in DB.nets:
        points = centers_local[net.macro_ids]
        total += np.ptp(points[:, 0]) + np.ptp(points[:, 1])
    return float(total)


def boundary_penalty(centers_local: np.ndarray) -> np.ndarray:
    x, y = centers_local[:, 0], centers_local[:, 1]
    return (
        np.maximum(WIDTHS / 2.0 - x, 0.0)
        + np.maximum(x + WIDTHS / 2.0 - DB.canvas_width, 0.0)
        + np.maximum(HEIGHTS / 2.0 - y, 0.0)
        + np.maximum(y + HEIGHTS / 2.0 - DB.canvas_height, 0.0)
    )


def pair_values(
    centers_local: np.ndarray,
    pair_i: np.ndarray = PAIR_I,
    pair_j: np.ndarray = PAIR_J,
    pair_w: Optional[np.ndarray] = None,
    pair_h: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    if pair_w is None:
        pair_w = (WIDTHS[pair_i] + WIDTHS[pair_j]) / 2.0
    if pair_h is None:
        pair_h = (HEIGHTS[pair_i] + HEIGHTS[pair_j]) / 2.0
    dx = centers_local[pair_i, 0] - centers_local[pair_j, 0]
    dy = centers_local[pair_i, 1] - centers_local[pair_j, 1]
    nij = np.maximum(np.abs(dx) / pair_w, np.abs(dy) / pair_h) - 1.0
    phi = np.maximum(-nij, 0.0)
    return {"dx": dx, "dy": dy, "Nij": nij, "phi": phi}


def full_metrics(
    centers_local: np.ndarray,
    gamma_boundary: float,
    gamma_overlap: float,
) -> Dict[str, Any]:
    centers_local = np.asarray(centers_local, dtype=np.float64)
    if centers_local.shape != (DB.num_macros, 2):
        raise ValueError(f"Unexpected placement shape {centers_local.shape}")
    pair = pair_values(centers_local)
    boundary = boundary_penalty(centers_local)
    hpwl_pin = pin_hpwl(centers_local)
    hpwl_center = center_hpwl_audit(centers_local)
    overlap_sum = float(pair["phi"].sum())
    boundary_sum = float(boundary.sum())
    return {
        "hpwl_pin": hpwl_pin,
        "hpwl_center_audit": hpwl_center,
        "total_pairs": int(PAIR_I.size),
        "overlap_pairs": int((pair["Nij"] < 0.0).sum()),
        "overlap_hat_sum": overlap_sum,
        "overlap_hat_max": float(pair["phi"].max()) if PAIR_I.size else 0.0,
        "boundary_violating_macros": int((boundary > 0.0).sum()),
        "boundary_penalty_sum": boundary_sum,
        "weighted_overlap_penalty": gamma_overlap * overlap_sum,
        "weighted_boundary_penalty": gamma_boundary * boundary_sum,
        "penalized_objective": (
            hpwl_pin + gamma_overlap * overlap_sum + gamma_boundary * boundary_sum
        ),
        "Nij_min": float(pair["Nij"].min()) if PAIR_I.size else 0.0,
        "Nij_max": float(pair["Nij"].max()) if PAIR_I.size else 0.0,
    }


INITIAL_METRICS_BASE_GAMMA = full_metrics(
    Z0_PAPER_CENTERS, INITIAL_GAMMA_BOUNDARY, INITIAL_GAMMA_OVERLAP
)
log_msg("Initial full-pair metrics: " + json.dumps(INITIAL_METRICS_BASE_GAMMA))
'''
    ),
    markdown_cell(
        r'''
## Vì sao notebook này không đưa full-pairs vào CPLEX?

Với 543 macro, adaptec1 có 147,153 pair. Full model cần một biến `q` và bốn ràng buộc cho mỗi pair, tức gần 589 nghìn ràng buộc overlap trước khi tính HPWL và boundary. DOcplex tạo thêm Python objects nên RAM lớn hơn nhiều so với các NumPy arrays.

Full-pair vẫn được dùng để đánh giá và kiểm tra nghiệm. Chỉ phần mô hình CPLEX dùng persistent working set.
'''
    ),
    code_cell(
        r'''
# Cell 6 - Persistent working set and deterministic nonzero subgradient

@dataclass
class ActiveSetData:
    keys: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_w: np.ndarray
    pair_h: np.ndarray

    @property
    def size(self) -> int:
        return int(self.keys.size)


class PersistentWorkingSet:
    def __init__(self):
        self.keys: Set[int] = set()

    def add_from_mask(self, mask: np.ndarray) -> int:
        new_keys = {int(key) for key in PAIR_KEYS[np.asarray(mask, dtype=bool)]}
        before = len(self.keys)
        self.keys.update(new_keys)
        return len(self.keys) - before

    def add_near(self, z: np.ndarray, margin: float) -> int:
        nij = pair_values(z)["Nij"]
        return self.add_from_mask(nij <= margin)

    def missing_near_mask(self, z: np.ndarray, margin: float) -> np.ndarray:
        nij = pair_values(z)["Nij"]
        near_indices = np.flatnonzero(nij <= margin)
        missing = np.zeros(PAIR_I.size, dtype=bool)
        for idx in near_indices:
            if int(PAIR_KEYS[idx]) not in self.keys:
                missing[idx] = True
        return missing

    def as_active_data(self) -> ActiveSetData:
        keys = np.asarray(sorted(self.keys), dtype=np.int64)
        indices = np.asarray([PAIR_KEY_TO_INDEX[int(key)] for key in keys], dtype=np.int64)
        return ActiveSetData(
            keys=keys,
            pair_i=PAIR_I[indices],
            pair_j=PAIR_J[indices],
            pair_w=PAIR_W[indices],
            pair_h=PAIR_H[indices],
        )


def deterministic_sign(
    values: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    axis: int,
) -> np.ndarray:
    signs = np.sign(values).astype(np.float64)
    zero = np.abs(values) <= ZERO_DISPLACEMENT_TOL
    # Stable ±1 direction derived from pair ids. This is a valid abs subgradient at zero.
    parity = (pair_i * 131 + pair_j * 17 + axis) & 1
    fallback = np.where(parity == 0, 1.0, -1.0)
    signs[zero] = fallback[zero]
    return signs


def subgradient_H(
    z: np.ndarray,
    active: ActiveSetData,
    gamma_overlap: float,
    rho: float,
) -> np.ndarray:
    grad = rho * z.copy()
    if active.size == 0:
        return grad

    dx = z[active.pair_i, 0] - z[active.pair_j, 0]
    dy = z[active.pair_i, 1] - z[active.pair_j, 1]
    ax = np.abs(dx) / active.pair_w
    ay = np.abs(dy) / active.pair_h

    x_dominant = ax > ay + BRANCH_TIE_TOL
    y_dominant = ay > ax + BRANCH_TIE_TOL
    tied = ~(x_dominant | y_dominant)
    choose_x_on_tie = ((active.pair_i + active.pair_j) & 1) == 0
    x_weight = (x_dominant | (tied & choose_x_on_tie)).astype(np.float64)
    y_weight = (y_dominant | (tied & ~choose_x_on_tie)).astype(np.float64)

    sx = deterministic_sign(dx, active.pair_i, active.pair_j, axis=0)
    sy = deterministic_sign(dy, active.pair_i, active.pair_j, axis=1)
    gx = gamma_overlap * x_weight * sx / active.pair_w
    gy = gamma_overlap * y_weight * sy / active.pair_h

    np.add.at(grad[:, 0], active.pair_i, gx)
    np.add.at(grad[:, 0], active.pair_j, -gx)
    np.add.at(grad[:, 1], active.pair_i, gy)
    np.add.at(grad[:, 1], active.pair_j, -gy)
    return grad


WORKING_SET = PersistentWorkingSet()
added = WORKING_SET.add_near(Z0_PAPER_CENTERS, ACTIVE_SET_MARGIN)
log_msg(f"Initial working set: {len(WORKING_SET.keys)} pairs; added={added}")
'''
    ),
    code_cell(
        r'''
# Cell 7 - Build the convex CPLEX QP using pin-level HPWL and a trust region

def build_subproblem_model(
    z_anchor: np.ndarray,
    active: ActiveSetData,
    yk: np.ndarray,
    gamma_boundary: float,
    gamma_overlap: float,
    rho: float,
    trust_radius: float,
):
    model = Model(name="pdca_workingset_eq17")
    model.parameters.threads = CPLEX_THREADS
    infinity = model.infinity

    x = [
        model.continuous_var(
            lb=float(z_anchor[i, 0] - trust_radius),
            ub=float(z_anchor[i, 0] + trust_radius),
            name=f"x_{i}",
        )
        for i in range(DB.num_macros)
    ]
    y = [
        model.continuous_var(
            lb=float(z_anchor[i, 1] - trust_radius),
            ub=float(z_anchor[i, 1] + trust_radius),
            name=f"y_{i}",
        )
        for i in range(DB.num_macros)
    ]

    num_nets = len(DB.nets)
    xmax = model.continuous_var_list(num_nets, lb=-infinity, name="xmax")
    xmin = model.continuous_var_list(num_nets, lb=-infinity, name="xmin")
    ymax = model.continuous_var_list(num_nets, lb=-infinity, name="ymax")
    ymin = model.continuous_var_list(num_nets, lb=-infinity, name="ymin")

    b_left = model.continuous_var_list(DB.num_macros, lb=0, name="b_l")
    b_right = model.continuous_var_list(DB.num_macros, lb=0, name="b_r")
    b_bottom = model.continuous_var_list(DB.num_macros, lb=0, name="b_b")
    b_top = model.continuous_var_list(DB.num_macros, lb=0, name="b_t")
    q = model.continuous_var_list(active.size, lb=0, name="q")

    constraints = []
    # Pin-level HPWL: pin position = macro center + Bookshelf pin offset.
    for net_id, net in enumerate(DB.nets):
        for macro_id, offset in zip(net.pin_macro_ids, net.pin_offsets):
            i = int(macro_id)
            dx, dy = float(offset[0]), float(offset[1])
            constraints.extend([
                xmax[net_id] >= x[i] + dx,
                xmin[net_id] <= x[i] + dx,
                ymax[net_id] >= y[i] + dy,
                ymin[net_id] <= y[i] + dy,
            ])

    for i in range(DB.num_macros):
        constraints.extend([
            b_left[i] + x[i] >= float(WIDTHS[i] / 2.0),
            b_right[i] - x[i] >= float(WIDTHS[i] / 2.0 - DB.canvas_width),
            b_bottom[i] + y[i] >= float(HEIGHTS[i] / 2.0),
            b_top[i] - y[i] >= float(HEIGHTS[i] / 2.0 - DB.canvas_height),
        ])

    for k in range(active.size):
        i, j = int(active.pair_i[k]), int(active.pair_j[k])
        inv_w, inv_h = 1.0 / float(active.pair_w[k]), 1.0 / float(active.pair_h[k])
        constraints.extend([
            q[k] >= inv_w * (x[i] - x[j]) - 1.0,
            q[k] >= inv_w * (x[j] - x[i]) - 1.0,
            q[k] >= inv_h * (y[i] - y[j]) - 1.0,
            q[k] >= inv_h * (y[j] - y[i]) - 1.0,
        ])
    model.add_constraints(constraints)

    hpwl_term = model.sum(
        xmax[e] - xmin[e] + ymax[e] - ymin[e] for e in range(num_nets)
    )
    boundary_term = gamma_boundary * model.sum(
        b_left[i] + b_right[i] + b_bottom[i] + b_top[i]
        for i in range(DB.num_macros)
    )
    overlap_term = gamma_overlap * model.sum(q)
    proximal_term = 0.5 * rho * model.sum(
        x[i] ** 2 + y[i] ** 2 for i in range(DB.num_macros)
    )
    linearized_H = -model.sum(
        float(yk[i, 0]) * x[i] + float(yk[i, 1]) * y[i]
        for i in range(DB.num_macros)
    )
    model.minimize(
        hpwl_term + boundary_term + overlap_term + proximal_term + linearized_H
    )
    return model, x, y


log_msg("CPLEX working-set QP builder ready")
'''
    ),
    code_cell(
        r'''
# Cell 8 - Solve one QP and preserve the real CPLEX status

def solve_subproblem(
    z_anchor: np.ndarray,
    active: ActiveSetData,
    yk: np.ndarray,
    gamma_boundary: float,
    gamma_overlap: float,
    rho: float,
    trust_radius: float,
    export_lp_path: Optional[Path] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    build_start = time.time()
    model, x_vars, y_vars = build_subproblem_model(
        z_anchor=z_anchor,
        active=active,
        yk=yk,
        gamma_boundary=gamma_boundary,
        gamma_overlap=gamma_overlap,
        rho=rho,
        trust_radius=trust_radius,
    )
    build_time = time.time() - build_start

    if export_lp_path is not None:
        model.export_as_lp(str(export_lp_path))

    solve_start = time.time()
    solution = model.solve(log_output=False)
    solve_time = time.time() - solve_start
    status = str(model.solve_details.status if model.solve_details else "unknown")

    info = {
        "cplex_status": status,
        "cplex_build_time_sec": build_time,
        "cplex_solve_time_sec": solve_time,
        "active_pairs": active.size,
    }
    if solution is None or "optimal" not in status.lower():
        model.end()
        return None, info

    z_candidate = np.empty((DB.num_macros, 2), dtype=np.float64)
    z_candidate[:, 0] = [solution.get_value(var) for var in x_vars]
    z_candidate[:, 1] = [solution.get_value(var) for var in y_vars]
    info["surrogate_objective"] = float(solution.objective_value)
    model.end()

    if not np.isfinite(z_candidate).all():
        info["cplex_status"] = "non_finite_solution"
        return None, info
    return z_candidate, info


log_msg("CPLEX solver wrapper ready")
'''
    ),
    code_cell(
        r'''
# Cell 9 - One fixed-penalty proximal-DCA stage with separation and rejection

@dataclass
class StageResult:
    z: np.ndarray
    history: List[Dict[str, Any]]
    status: str
    converged: bool


def line_search_full_objective(
    zk: np.ndarray,
    candidate: np.ndarray,
    current_objective: float,
    gamma_boundary: float,
    gamma_overlap: float,
    working_set: PersistentWorkingSet,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], float, int]:
    tolerance = MONOTONE_RTOL * max(1.0, abs(current_objective))
    for backtrack in range(1, BACKTRACK_STEPS + 1):
        alpha = 0.5 ** backtrack
        trial = zk + alpha * (candidate - zk)
        missing = working_set.missing_near_mask(trial, ACTIVE_SET_MARGIN)
        if missing.any():
            working_set.add_from_mask(missing)
            return None, None, alpha, int(missing.sum())
        metrics = full_metrics(trial, gamma_boundary, gamma_overlap)
        if metrics["penalized_objective"] <= current_objective + tolerance:
            return trial, metrics, alpha, 0
    return None, None, 0.0, 0


def run_fixed_penalty_stage(
    z_start: np.ndarray,
    working_set: PersistentWorkingSet,
    stage: int,
    gamma_boundary: float,
    gamma_overlap: float,
) -> StageResult:
    zk = z_start.copy()
    history: List[Dict[str, Any]] = []
    trust_radius = TRUST_RADIUS_FRACTION * min(DB.canvas_width, DB.canvas_height)
    status = "max_iterations_reached"

    for iteration in range(MAX_DCA_ITER_PER_STAGE):
        iter_start = time.time()
        working_set.add_near(zk, ACTIVE_SET_MARGIN)
        current_metrics = full_metrics(zk, gamma_boundary, gamma_overlap)
        current_objective = current_metrics["penalized_objective"]
        accepted = False
        total_new_pairs = 0
        total_build_time = 0.0
        total_solve_time = 0.0
        candidate_info: Dict[str, Any] = {}
        accepted_alpha = 1.0

        for refinement in range(MAX_WORKING_SET_REFINEMENTS):
            active = working_set.as_active_data()
            yk = subgradient_H(zk, active, gamma_overlap, RHO)
            export_path = None
            if stage == 0 and iteration == 0 and refinement == 0:
                export_path = CPLEX_MODELS_DIR / "stage_00_iter_000_workingset.lp"
            candidate, candidate_info = solve_subproblem(
                z_anchor=zk,
                active=active,
                yk=yk,
                gamma_boundary=gamma_boundary,
                gamma_overlap=gamma_overlap,
                rho=RHO,
                trust_radius=trust_radius,
                export_lp_path=export_path,
            )
            total_build_time += candidate_info.get("cplex_build_time_sec", 0.0)
            total_solve_time += candidate_info.get("cplex_solve_time_sec", 0.0)
            if candidate is None:
                status = "cplex_failed"
                return StageResult(zk, history, status, False)

            # Separation: a candidate is not accepted while it creates a near pair
            # that was absent from the QP.
            missing = working_set.missing_near_mask(candidate, ACTIVE_SET_MARGIN)
            if missing.any():
                added = working_set.add_from_mask(missing)
                total_new_pairs += added
                log_msg(
                    f"stage={stage} iter={iteration} refinement={refinement}: "
                    f"added {added} candidate-near pairs"
                )
                continue

            candidate_metrics = full_metrics(candidate, gamma_boundary, gamma_overlap)
            tolerance = MONOTONE_RTOL * max(1.0, abs(current_objective))
            if candidate_metrics["penalized_objective"] <= current_objective + tolerance:
                z_next = candidate
                next_metrics = candidate_metrics
                accepted = True
                break

            # The incomplete working-set surrogate can still lose descent even if no
            # new pair crosses the margin. Backtracking is accepted only after a full audit.
            trial, trial_metrics, alpha, added = line_search_full_objective(
                zk, candidate, current_objective,
                gamma_boundary, gamma_overlap, working_set,
            )
            total_new_pairs += added
            if added > 0:
                continue
            if trial is not None and trial_metrics is not None:
                z_next = trial
                next_metrics = trial_metrics
                accepted_alpha = alpha
                accepted = True
                break
            status = "monotone_guard_failed"
            break

        if not accepted:
            if status == "max_iterations_reached":
                status = "working_set_refinement_limit"
            log_msg(f"stage={stage} iter={iteration}: stopped with {status}")
            return StageResult(zk, history, status, False)

        step_norm = float(np.linalg.norm(z_next - zk))
        relative_step = step_norm / max(1.0, float(np.linalg.norm(zk)))
        max_coordinate_step = float(np.max(np.abs(z_next - zk)))
        row = {
            "penalty_stage": stage,
            "iteration": iteration,
            "gamma_boundary": gamma_boundary,
            "gamma_overlap": gamma_overlap,
            "F_before": current_objective,
            "F_after": next_metrics["penalized_objective"],
            "hpwl_pin_after": next_metrics["hpwl_pin"],
            "hpwl_center_after": next_metrics["hpwl_center_audit"],
            "overlap_pairs_after": next_metrics["overlap_pairs"],
            "overlap_hat_sum_after": next_metrics["overlap_hat_sum"],
            "overlap_hat_max_after": next_metrics["overlap_hat_max"],
            "boundary_violating_after": next_metrics["boundary_violating_macros"],
            "working_set_pairs": len(working_set.keys),
            "new_pairs_this_iteration": total_new_pairs,
            "step_norm": step_norm,
            "relative_step": relative_step,
            "max_coordinate_step": max_coordinate_step,
            "accepted_alpha": accepted_alpha,
            "cplex_status": candidate_info.get("cplex_status"),
            "cplex_build_time_sec": total_build_time,
            "cplex_solve_time_sec": total_solve_time,
            "iteration_time_sec": time.time() - iter_start,
        }
        history.append(row)
        log_msg(
            f"stage={stage} iter={iteration + 1:02d} "
            f"F={row['F_after']:.6e} pinHPWL={row['hpwl_pin_after']:.2f} "
            f"overlaps={row['overlap_pairs_after']} maxPhi={row['overlap_hat_max_after']:.3e} "
            f"WS={row['working_set_pairs']} relStep={relative_step:.3e} "
            f"alpha={accepted_alpha:.3f}"
        )

        zk = z_next
        if relative_step <= RELATIVE_STEP_TOL:
            status = "converged_relative_step"
            return StageResult(zk, history, status, True)

    return StageResult(zk, history, status, False)


log_msg("Fixed-penalty DCA stage ready")
'''
    ),
    code_cell(
        r'''
# Cell 10 - Adaptive penalty continuation

@dataclass
class AdaptiveResult:
    z_star: np.ndarray
    history: List[Dict[str, Any]]
    stage_history: List[Dict[str, Any]]
    final_gamma_boundary: float
    final_gamma_overlap: float
    final_status: str
    feasible: bool
    total_time_sec: float


def run_adaptive_pdca(z0: np.ndarray) -> AdaptiveResult:
    start = time.time()
    z = z0.copy()
    gamma_boundary = INITIAL_GAMMA_BOUNDARY
    gamma_overlap = INITIAL_GAMMA_OVERLAP
    working_set = PersistentWorkingSet()
    working_set.add_near(z, ACTIVE_SET_MARGIN)
    history: List[Dict[str, Any]] = []
    stage_history: List[Dict[str, Any]] = []
    final_status = "not_started"

    for stage in range(MAX_PENALTY_STAGES):
        before = full_metrics(z, gamma_boundary, gamma_overlap)
        log_msg("=" * 88)
        log_msg(
            f"PENALTY STAGE {stage}: gamma_boundary={gamma_boundary:.3e}, "
            f"gamma_overlap={gamma_overlap:.3e}, WS={len(working_set.keys)}"
        )
        stage_result = run_fixed_penalty_stage(
            z_start=z,
            working_set=working_set,
            stage=stage,
            gamma_boundary=gamma_boundary,
            gamma_overlap=gamma_overlap,
        )
        z = stage_result.z
        history.extend(stage_result.history)
        after = full_metrics(z, gamma_boundary, gamma_overlap)
        feasible = (
            after["overlap_pairs"] == 0
            and after["boundary_violating_macros"] == 0
        )
        stage_history.append({
            "penalty_stage": stage,
            "gamma_boundary": gamma_boundary,
            "gamma_overlap": gamma_overlap,
            "status": stage_result.status,
            "converged": stage_result.converged,
            "F_before": before["penalized_objective"],
            "F_after": after["penalized_objective"],
            "hpwl_pin_after": after["hpwl_pin"],
            "overlap_pairs_after": after["overlap_pairs"],
            "overlap_hat_max_after": after["overlap_hat_max"],
            "boundary_violating_after": after["boundary_violating_macros"],
            "working_set_pairs": len(working_set.keys),
            "feasible": feasible,
        })
        final_status = stage_result.status
        log_msg(
            f"STAGE {stage} END: status={final_status}, pinHPWL={after['hpwl_pin']:.2f}, "
            f"overlaps={after['overlap_pairs']}, boundary={after['boundary_violating_macros']}"
        )

        if feasible:
            final_status = "paper_feasible"
            break
        if stage_result.status in {"cplex_failed", "monotone_guard_failed"}:
            break
        # Do not report a gamma value that was never used to optimize z.
        if stage + 1 < MAX_PENALTY_STAGES:
            if after["overlap_pairs"] > 0:
                gamma_overlap *= PENALTY_GROWTH
            if after["boundary_violating_macros"] > 0:
                gamma_boundary *= PENALTY_GROWTH

    final_metrics = full_metrics(z, gamma_boundary, gamma_overlap)
    return AdaptiveResult(
        z_star=z,
        history=history,
        stage_history=stage_history,
        final_gamma_boundary=gamma_boundary,
        final_gamma_overlap=gamma_overlap,
        final_status=final_status,
        feasible=(
            final_metrics["overlap_pairs"] == 0
            and final_metrics["boundary_violating_macros"] == 0
        ),
        total_time_sec=time.time() - start,
    )


log_msg("Adaptive continuation ready")
'''
    ),
    code_cell(
        r'''
# Cell 11 - Run DRL -> adaptive proximal DCA with CPLEX working set

RESULT = run_adaptive_pdca(Z0_PAPER_CENTERS)
Z_STAR = RESULT.z_star

log_msg("=" * 88)
log_msg(
    f"RUN END: status={RESULT.final_status}, feasible={RESULT.feasible}, "
    f"time={RESULT.total_time_sec:.2f}s, accepted_iterations={len(RESULT.history)}"
)
'''
    ),
    code_cell(
        r'''
# Cell 12 - Correct local/global coordinate conversion and write Bookshelf .pl

def paper_centers_to_local_lower_left(centers_local: np.ndarray) -> np.ndarray:
    return centers_local - np.column_stack((WIDTHS, HEIGHTS)) / 2.0


def paper_centers_to_global_lower_left(centers_local: np.ndarray) -> np.ndarray:
    local_lower_left = paper_centers_to_local_lower_left(centers_local)
    return local_lower_left + np.asarray([DB.x_min, DB.y_min], dtype=np.float64)


def global_lower_left_to_paper_centers(global_lower_left: np.ndarray) -> np.ndarray:
    local_lower_left = global_lower_left - np.asarray([DB.x_min, DB.y_min])
    return local_lower_left + np.column_stack((WIDTHS, HEIGHTS)) / 2.0


def write_bookshelf_pl(centers_local: np.ndarray, destination: Path) -> None:
    global_lower_left = paper_centers_to_global_lower_left(centers_local)
    coordinates = {
        macro.name: global_lower_left[idx]
        for idx, macro in enumerate(DB.macros)
    }
    source = BENCH_DIR / f"{BENCHMARK}.pl"
    output_lines = []
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            fields = raw.strip().split()
            if fields and fields[0] in coordinates and len(fields) >= 3:
                x_value, y_value = coordinates[fields[0]]
                fields[1] = f"{x_value:.6f}"
                fields[2] = f"{y_value:.6f}"
                output_lines.append("\t".join(fields))
            else:
                output_lines.append(raw.rstrip("\r\n"))
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(output_lines) + "\n")


FINAL_PL_PATH = PLACEMENTS_DIR / f"{BENCHMARK}.drl_pdca_cplex_workingset.pl"
GLOBAL_LOWER_LEFT = paper_centers_to_global_lower_left(Z_STAR)
roundtrip = global_lower_left_to_paper_centers(GLOBAL_LOWER_LEFT)
if not np.allclose(roundtrip, Z_STAR, atol=1e-9):
    raise AssertionError("Local/global coordinate round-trip failed")

np.save(PLACEMENTS_DIR / "z_star_paper_centers.npy", Z_STAR)
np.save(PLACEMENTS_DIR / "z_star_lower_left_global.npy", GLOBAL_LOWER_LEFT)
write_bookshelf_pl(Z_STAR, FINAL_PL_PATH)
log_msg(f"Saved corrected global-coordinate Bookshelf placement: {FINAL_PL_PATH}")
'''
    ),
    code_cell(
        r'''
# Cell 13 - Save metrics and auditable histories

FINAL_METRICS = full_metrics(
    Z_STAR, RESULT.final_gamma_boundary, RESULT.final_gamma_overlap
)
INITIAL_METRICS_FINAL_GAMMA = full_metrics(
    Z0_PAPER_CENTERS, RESULT.final_gamma_boundary, RESULT.final_gamma_overlap
)

with (METRICS_DIR / "metrics_before.json").open("w", encoding="utf-8") as handle:
    json.dump(INITIAL_METRICS_FINAL_GAMMA, handle, indent=2)
with (METRICS_DIR / "metrics_after.json").open("w", encoding="utf-8") as handle:
    json.dump(FINAL_METRICS, handle, indent=2)
with (METRICS_DIR / "stage_history.json").open("w", encoding="utf-8") as handle:
    json.dump(RESULT.stage_history, handle, indent=2)

HISTORY_CSV = METRICS_DIR / "dca_history.csv"
if RESULT.history:
    fieldnames = list(RESULT.history[0].keys())
    with HISTORY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(RESULT.history)
else:
    HISTORY_CSV.write_text("", encoding="utf-8")

comparison = [
    ("Pin HPWL", INITIAL_METRICS_FINAL_GAMMA["hpwl_pin"], FINAL_METRICS["hpwl_pin"]),
    ("Center HPWL audit", INITIAL_METRICS_FINAL_GAMMA["hpwl_center_audit"], FINAL_METRICS["hpwl_center_audit"]),
    ("Overlap pairs", INITIAL_METRICS_FINAL_GAMMA["overlap_pairs"], FINAL_METRICS["overlap_pairs"]),
    ("Overlap hat max", INITIAL_METRICS_FINAL_GAMMA["overlap_hat_max"], FINAL_METRICS["overlap_hat_max"]),
    ("Boundary violations", INITIAL_METRICS_FINAL_GAMMA["boundary_violating_macros"], FINAL_METRICS["boundary_violating_macros"]),
    ("Penalized objective", INITIAL_METRICS_FINAL_GAMMA["penalized_objective"], FINAL_METRICS["penalized_objective"]),
]
print(f"{'Metric':<28} {'Before':>18} {'After':>18} {'Delta':>18}")
print("-" * 86)
for name, before, after in comparison:
    print(f"{name:<28} {before:>18.6f} {after:>18.6f} {after-before:>+18.6f}")
'''
    ),
    code_cell(
        r'''
# Cell 14 - Placement and convergence plots

def draw_placement(axis, centers: np.ndarray, title: str) -> None:
    lower_left_local = paper_centers_to_local_lower_left(centers)
    for idx, macro in enumerate(DB.macros):
        rectangle = plt.Rectangle(
            lower_left_local[idx], macro.width, macro.height,
            fill=False, linewidth=0.35, alpha=0.65,
        )
        axis.add_patch(rectangle)
    axis.set_xlim(0, DB.canvas_width)
    axis.set_ylim(0, DB.canvas_height)
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_xlabel("local x")
    axis.set_ylabel("local y")


figure, axes = plt.subplots(1, 2, figsize=(14, 6))
draw_placement(axes[0], Z0_PAPER_CENTERS, "DRL input")
draw_placement(axes[1], Z_STAR, "After CPLEX working-set PDCA")
figure.tight_layout()
PLACEMENT_PLOT = PLOTS_DIR / "checkpoint_vs_workingset_pdca.png"
figure.savefig(PLACEMENT_PLOT, dpi=180, bbox_inches="tight")
plt.show()

if RESULT.history:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    xs = np.arange(1, len(RESULT.history) + 1)
    axes[0].plot(xs, [row["F_after"] for row in RESULT.history])
    axes[0].set_title("Penalized objective")
    axes[1].plot(xs, [row["overlap_pairs_after"] for row in RESULT.history])
    axes[1].set_title("Overlap pairs")
    axes[2].semilogy(xs, np.maximum(
        [row["relative_step"] for row in RESULT.history], 1e-18
    ))
    axes[2].set_title("Relative step")
    for axis in axes:
        axis.set_xlabel("accepted DCA iteration")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    CONVERGENCE_PLOT = PLOTS_DIR / "convergence.png"
    figure.savefig(CONVERGENCE_PLOT, dpi=180, bbox_inches="tight")
    plt.show()
'''
    ),
    code_cell(
        r'''
# Cell 15 - Final validation report

if RESULT.history:
    for row in RESULT.history:
        tolerance = MONOTONE_RTOL * max(1.0, abs(row["F_before"]))
        if row["F_after"] > row["F_before"] + tolerance:
            raise AssertionError(
                f"Accepted non-monotone step at stage={row['penalty_stage']} "
                f"iteration={row['iteration']}"
            )

assert FINAL_PL_PATH.is_file()
assert (METRICS_DIR / "metrics_after.json").is_file()
assert HISTORY_CSV.is_file()

summary = {
    "benchmark": BENCHMARK,
    "drl_seed": drl_metrics.get("seed"),
    "status": RESULT.final_status,
    "paper_feasible": RESULT.feasible,
    "accepted_iterations": len(RESULT.history),
    "penalty_stages": len(RESULT.stage_history),
    "runtime_sec": RESULT.total_time_sec,
    "final_gamma_boundary": RESULT.final_gamma_boundary,
    "final_gamma_overlap": RESULT.final_gamma_overlap,
    "working_set_pairs": (
        RESULT.history[-1]["working_set_pairs"] if RESULT.history else 0
    ),
    "full_pairs": int(PAIR_I.size),
    "final_metrics": FINAL_METRICS,
    "output_pl": str(FINAL_PL_PATH),
}
with (METRICS_DIR / "summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print(json.dumps(summary, indent=2))
print("\nValidation passed: every accepted step is monotone under the full-pair objective.")
'''
    ),
]


metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    },
}

notebook = {
    "cells": cells,
    "metadata": metadata,
    "nbformat": 4,
    "nbformat_minor": 5,
}

DESTINATION.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DESTINATION}")

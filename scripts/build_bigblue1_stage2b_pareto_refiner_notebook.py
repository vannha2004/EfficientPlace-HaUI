from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_NOTEBOOK = ROOT / "Bigblue1_EfficientPlace_Stage2A_Pareto_Evaluator.ipynb"
DESTINATION = ROOT / "Bigblue1_EfficientPlace_Stage2B_ParetoSafe_Refiner.ipynb"


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


if not BASE_NOTEBOOK.exists():
    raise FileNotFoundError(
        f"Build Stage 2A first; missing base notebook: {BASE_NOTEBOOK}"
    )

base = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
# Reuse the self-contained input discovery, parser, evaluator, and Stage 2A audit.
# The generated Stage 2B notebook does not depend on the local Stage 2A file.
audit_cells = copy.deepcopy(base["cells"][1:8])
config_source = "".join(audit_cells[0]["source"])
config_source = config_source.replace(
    'output_root: str = "/kaggle/working/bigblue1_stage2a_evaluator"',
    'output_root: str = "/kaggle/working/bigblue1_stage2b_audit"',
)
audit_cells[0]["source"] = config_source.splitlines(keepends=True)


cells = [
    markdown_cell(
        r'''
# Stage 2B — Pareto-Safe HPWL–Congestion Refinement trên `bigblue1`

Notebook này thử nghiệm đóng góp thuật toán tiếp theo:

> Từ checkpoint EfficientPlace và Pareto endpoints của Stage 1, tìm placement giảm pin-aware HPWL nhưng không làm xấu **worst-resolution capacity-adjusted RUDY**, đồng thời luôn giữ legality.

Các thành phần chính:

1. hai điểm khởi tạo: EfficientPlace baseline và Stage-1 best-HPWL;
2. congestion objective là giá trị xấu nhất trên grid `32×32`, `64×64`, `128×128`;
3. hard boundary/overlap gate cho mọi candidate;
4. exact incremental HPWL và RUDY cho các macro-net bị tác động;
5. single-macro moves kết hợp macro swap;
6. external Pareto archive;
7. strict-dominance target và ε-congestion fallback được cố định **trước khi chạy**.

Mặc định notebook chạy `smoke`. Không chuyển sang `full` trước khi smoke audit thành công.

## Kaggle inputs

- checkpoint `tranvannha/ckpt-efficientplace-bigblue1`;
- dataset ISPD2005;
- Notebook Output của Stage 1 full run, chứa 5 thư mục `seed_0` ... `seed_4`.

> Kết quả vẫn là macro-only proxy. Notebook không tuyên bố full-chip HPWL hoặc routed congestion.
'''
    ),
    *audit_cells,
    markdown_cell(
        r'''
## Stage 2B search protocol

Ký hiệu:

- `H(X)`: pin-aware macro-only HPWL, chuẩn hóa bởi EfficientPlace baseline;
- `C_r(X)`: capacity-adjusted RUDY top-1% trên resolution `r`;
- `C(X) = max_r C_r(X)/C_r(X_baseline)`.

Mục tiêu strict dominance:

\[
H(X) < 1,\qquad C(X) \le 1.
\]

Fallback thực dụng cho quá trình khám phá:

\[
H(X) \le 1,\qquad C(X) \le 1 + \epsilon_C,
\]

với `εC = 0.05%` được đặt trước khi chạy. Pareto archive vẫn lưu và báo cáo strict dominance riêng biệt.
'''
    ),
    code_cell(
        r'''
# Cell 8 — Cấu hình Stage 2B; mặc định smoke
from dataclasses import dataclass as search_dataclass


@search_dataclass(frozen=True)
class SearchConfig:
    run_mode: str = "smoke"  # đổi thành "full" chỉ sau khi smoke audit sạch
    output_root: str = "/kaggle/working/bigblue1_stage2b_pareto_safe"
    smoke_seeds: Tuple[int, ...] = (0,)
    full_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    smoke_radii_cells: Tuple[int, ...] = (8, 4, 2)
    full_radii_cells: Tuple[int, ...] = (32, 16, 8, 4, 2, 1)
    smoke_macros_per_sweep: int = 48
    full_macros_per_sweep: int = 96
    smoke_candidates_per_macro: int = 12
    full_candidates_per_macro: int = 18
    smoke_swap_trials: int = 24
    full_swap_trials: int = 96
    exact_shortlist: int = 3
    congestion_epsilon_pct: float = 0.05
    minimum_preserved_hpwl_improvement_pct: float = 0.25
    objective_tolerance: float = 1e-10
    max_archive_size: int = 64


SEARCH_CFG = SearchConfig()
if SEARCH_CFG.run_mode not in {"smoke", "full"}:
    raise ValueError("run_mode must be 'smoke' or 'full'")
if SEARCH_CFG.congestion_epsilon_pct < 0.0:
    raise ValueError("congestion_epsilon_pct must be non-negative")

EPSILON_RATIO = SEARCH_CFG.congestion_epsilon_pct / 100.0
MIN_HPWL_GAIN_RATIO = SEARCH_CFG.minimum_preserved_hpwl_improvement_pct / 100.0
PRIMARY_RESOLUTION = CFG.primary_resolution
SEARCH_SEEDS = (
    SEARCH_CFG.smoke_seeds if SEARCH_CFG.run_mode == "smoke" else SEARCH_CFG.full_seeds
)
RADII_CELLS = (
    SEARCH_CFG.smoke_radii_cells
    if SEARCH_CFG.run_mode == "smoke"
    else SEARCH_CFG.full_radii_cells
)
MACROS_PER_SWEEP = (
    SEARCH_CFG.smoke_macros_per_sweep
    if SEARCH_CFG.run_mode == "smoke"
    else SEARCH_CFG.full_macros_per_sweep
)
CANDIDATES_PER_MACRO = (
    SEARCH_CFG.smoke_candidates_per_macro
    if SEARCH_CFG.run_mode == "smoke"
    else SEARCH_CFG.full_candidates_per_macro
)
SWAP_TRIALS = (
    SEARCH_CFG.smoke_swap_trials
    if SEARCH_CFG.run_mode == "smoke"
    else SEARCH_CFG.full_swap_trials
)

STAGE2B_ROOT = Path(SEARCH_CFG.output_root)
STAGE2B_ROOT.mkdir(parents=True, exist_ok=True)
STAGE2B_RUN_DIR = STAGE2B_ROOT / f"{SEARCH_CFG.run_mode}_{time.strftime('%Y%m%d-%H%M%S')}"
STAGE2B_RUN_DIR.mkdir(parents=True, exist_ok=False)

BASE_HPWL = float(baseline["hpwl_pin"])
BASE_CONGESTION = {
    int(resolution): float(
        RESOLUTION_METRICS.loc[
            (RESOLUTION_METRICS["placement_id"] == "baseline")
            & (RESOLUTION_METRICS["resolution"] == int(resolution)),
            "blocked_rudy_top1pct_mean",
        ].iloc[0]
    )
    for resolution in CFG.rudy_resolutions
}

print("Run mode                     :", SEARCH_CFG.run_mode)
print("Search seeds                 :", SEARCH_SEEDS)
print("Congestion epsilon           :", SEARCH_CFG.congestion_epsilon_pct, "%")
print("Minimum preserved HPWL gain  :", SEARCH_CFG.minimum_preserved_hpwl_improvement_pct, "%")
print("Search output                :", STAGE2B_RUN_DIR)
'''
    ),
    code_cell(
        r'''
# Cell 9 — Vectorized rectangle patches và exact incremental candidate evaluator
def rectangle_patch(
    resolution: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    density: float,
) -> Tuple[slice, slice, np.ndarray] | None:
    pitch = CANVAS_EXTENT / resolution
    bin_area = pitch * pitch
    x0 = float(np.clip(x0, 0.0, CANVAS_EXTENT))
    y0 = float(np.clip(y0, 0.0, CANVAS_EXTENT))
    x1 = float(np.clip(x1, 0.0, CANVAS_EXTENT))
    y1 = float(np.clip(y1, 0.0, CANVAS_EXTENT))
    if x1 <= x0 or y1 <= y0:
        return None

    ix0 = max(0, min(resolution - 1, int(math.floor(x0 / pitch))))
    iy0 = max(0, min(resolution - 1, int(math.floor(y0 / pitch))))
    ix1 = max(0, min(resolution - 1, int(math.ceil(x1 / pitch) - 1)))
    iy1 = max(0, min(resolution - 1, int(math.ceil(y1 / pitch) - 1)))
    x_indices = np.arange(ix0, ix1 + 1)
    y_indices = np.arange(iy0, iy1 + 1)
    overlap_x = np.maximum(
        0.0,
        np.minimum(x1, (x_indices + 1) * pitch) - np.maximum(x0, x_indices * pitch),
    )
    overlap_y = np.maximum(
        0.0,
        np.minimum(y1, (y_indices + 1) * pitch) - np.maximum(y0, y_indices * pitch),
    )
    patch = density * np.outer(overlap_y, overlap_x) / bin_area
    return slice(iy0, iy1 + 1), slice(ix0, ix1 + 1), patch


def apply_rectangle_patch(target: np.ndarray, patch: Any, sign: float) -> None:
    if patch is None:
        return
    ys, xs, values = patch
    target[ys, xs] += sign * values


def macro_patch(macro_id: int, center: np.ndarray, resolution: int) -> Any:
    lower = center - SIZES[macro_id] / 2.0
    upper = center + SIZES[macro_id] / 2.0
    return rectangle_patch(
        resolution, lower[0], lower[1], upper[0], upper[1], 1.0
    )


def net_patch(net_id: int, centers: np.ndarray, resolution: int) -> Any:
    net = NETS[net_id]
    pins = centers[net["ids"]] + net["offsets"]
    xmin, ymin = pins.min(axis=0)
    xmax, ymax = pins.max(axis=0)
    hpwl = float((xmax - xmin) + (ymax - ymin))
    if hpwl <= 0.0:
        return None
    pitch = CANVAS_EXTENT / resolution
    width = max(float(xmax - xmin), pitch)
    height = max(float(ymax - ymin), pitch)
    cx, cy = float((xmin + xmax) / 2.0), float((ymin + ymax) / 2.0)
    x0 = float(np.clip(cx - width / 2.0, 0.0, CANVAS_EXTENT - width))
    y0 = float(np.clip(cy - height / 2.0, 0.0, CANVAS_EXTENT - height))
    return rectangle_patch(
        resolution, x0, y0, x0 + width, y0 + height, hpwl / (width * height)
    )


def blocked_map(occupancy: np.ndarray, rudy: np.ndarray) -> np.ndarray:
    capacity = np.clip(
        1.0 - CFG.blockage_weight * occupancy,
        CFG.capacity_floor,
        1.0,
    )
    return rudy / capacity


def congestion_value(occupancy: np.ndarray, rudy: np.ndarray) -> float:
    return top_fraction_mean(blocked_map(occupancy, rudy), 0.01)


MACRO_TO_NET_IDS: List[List[int]] = [[] for _ in MACRO_NAMES]
CONNECTED_MACROS: List[set[int]] = [set() for _ in MACRO_NAMES]
for net_id, net in enumerate(NETS):
    ids = [int(value) for value in net["ids"]]
    for macro_id in ids:
        MACRO_TO_NET_IDS[macro_id].append(net_id)
        CONNECTED_MACROS[macro_id].update(other for other in ids if other != macro_id)


class SearchState:
    def __init__(
        self,
        centers: np.ndarray,
        label: str,
        maps: Dict[int, Dict[str, np.ndarray]] | None = None,
    ) -> None:
        self.centers = np.asarray(centers, dtype=np.float64).copy()
        self.label = str(label)
        self.net_values = np.asarray([
            self._net_hpwl(net_id, self.centers) for net_id in range(len(NETS))
        ])
        self.hpwl = float(self.net_values.sum())
        if maps is None:
            self.maps = {
                int(resolution): congestion_maps(self.centers, int(resolution))
                for resolution in CFG.rudy_resolutions
            }
        else:
            self.maps = {
                int(resolution): {
                    key: np.asarray(value, dtype=np.float64).copy()
                    for key, value in maps[int(resolution)].items()
                }
                for resolution in CFG.rudy_resolutions
            }
        self.refresh_objectives()

    @staticmethod
    def _net_hpwl(net_id: int, centers: np.ndarray) -> float:
        net = NETS[net_id]
        pins = centers[net["ids"]] + net["offsets"]
        return float(np.ptp(pins[:, 0]) + np.ptp(pins[:, 1]))

    def refresh_objectives(self) -> None:
        self.hpwl_ratio = self.hpwl / BASE_HPWL
        self.congestion_by_resolution = {
            int(resolution): congestion_value(
                self.maps[int(resolution)]["macro_occupancy"],
                self.maps[int(resolution)]["rudy"],
            )
            for resolution in CFG.rudy_resolutions
        }
        self.congestion_ratio_by_resolution = {
            resolution: value / BASE_CONGESTION[resolution]
            for resolution, value in self.congestion_by_resolution.items()
        }
        self.robust_congestion_ratio = max(self.congestion_ratio_by_resolution.values())

    def snapshot(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "centers": self.centers.copy(),
            "hpwl": self.hpwl,
            "hpwl_ratio": self.hpwl_ratio,
            "robust_congestion_ratio": self.robust_congestion_ratio,
            "congestion_by_resolution": dict(self.congestion_by_resolution),
        }

    def moved_legal(self, candidate_centers: np.ndarray, moved_ids: Sequence[int]) -> bool:
        moved = sorted({int(value) for value in moved_ids})
        lower = candidate_centers - SIZES / 2.0
        upper = candidate_centers + SIZES / 2.0
        tol = CFG.geometry_tolerance
        for macro_id in moved:
            if np.any(lower[macro_id] < -tol) or np.any(upper[macro_id] > CANVAS_EXTENT + tol):
                return False
            overlap_x = np.minimum(upper[macro_id, 0], upper[:, 0]) - np.maximum(
                lower[macro_id, 0], lower[:, 0]
            )
            overlap_y = np.minimum(upper[macro_id, 1], upper[:, 1]) - np.maximum(
                lower[macro_id, 1], lower[:, 1]
            )
            overlapping = (overlap_x > tol) & (overlap_y > tol)
            overlapping[macro_id] = False
            if np.any(overlapping):
                return False
        return True

    def evaluate_moves(
        self,
        moves: Dict[int, np.ndarray],
        resolutions: Sequence[int],
    ) -> Dict[str, Any] | None:
        moved_ids = sorted(int(value) for value in moves)
        candidate_centers = self.centers.copy()
        for macro_id, center in moves.items():
            candidate_centers[int(macro_id)] = np.asarray(center, dtype=np.float64)
        if not self.moved_legal(candidate_centers, moved_ids):
            return None

        affected_nets = sorted({
            net_id for macro_id in moved_ids for net_id in MACRO_TO_NET_IDS[macro_id]
        })
        new_net_values = {
            net_id: self._net_hpwl(net_id, candidate_centers) for net_id in affected_nets
        }
        new_hpwl = self.hpwl + sum(
            new_net_values[net_id] - self.net_values[net_id] for net_id in affected_nets
        )
        hpwl_ratio = new_hpwl / BASE_HPWL

        candidate_maps: Dict[int, Dict[str, np.ndarray]] = {}
        ratios: Dict[int, float] = {}
        for resolution in resolutions:
            resolution = int(resolution)
            occupancy = self.maps[resolution]["macro_occupancy"].copy()
            rudy = self.maps[resolution]["rudy"].copy()
            for macro_id in moved_ids:
                apply_rectangle_patch(
                    occupancy, macro_patch(macro_id, self.centers[macro_id], resolution), -1.0
                )
                apply_rectangle_patch(
                    occupancy, macro_patch(macro_id, candidate_centers[macro_id], resolution), 1.0
                )
            for net_id in affected_nets:
                apply_rectangle_patch(rudy, net_patch(net_id, self.centers, resolution), -1.0)
                apply_rectangle_patch(rudy, net_patch(net_id, candidate_centers, resolution), 1.0)
            occupancy[np.abs(occupancy) < 1e-14] = 0.0
            rudy[np.abs(rudy) < 1e-14] = 0.0
            capacity = np.clip(
                1.0 - CFG.blockage_weight * occupancy,
                CFG.capacity_floor,
                1.0,
            )
            blocked = rudy / capacity
            value = top_fraction_mean(blocked, 0.01)
            ratios[resolution] = value / BASE_CONGESTION[resolution]
            candidate_maps[resolution] = {
                "macro_occupancy": occupancy,
                "rudy": rudy,
                "available_capacity": capacity,
                "blocked_rudy": blocked,
            }

        return {
            "moves": {key: value.copy() for key, value in moves.items()},
            "moved_ids": moved_ids,
            "centers": candidate_centers,
            "affected_nets": affected_nets,
            "new_net_values": new_net_values,
            "hpwl": float(new_hpwl),
            "hpwl_ratio": float(hpwl_ratio),
            "maps": candidate_maps,
            "congestion_ratio_by_resolution": ratios,
            "robust_congestion_ratio": float(max(ratios.values())),
        }

    def commit(self, candidate: Dict[str, Any], move_label: str) -> None:
        self.centers = candidate["centers"]
        self.hpwl = float(candidate["hpwl"])
        for net_id, value in candidate["new_net_values"].items():
            self.net_values[net_id] = value
        for resolution, maps in candidate["maps"].items():
            self.maps[int(resolution)] = maps
        self.label = move_label
        self.refresh_objectives()


baseline_state_for_patch_test = SearchState(CENTERS_0, "patch_test", MAPS["baseline"])
for resolution in CFG.rudy_resolutions:
    rebuilt = congestion_maps(CENTERS_0, int(resolution))
    if not np.allclose(
        baseline_state_for_patch_test.maps[int(resolution)]["rudy"],
        rebuilt["rudy"], rtol=0.0, atol=1e-12,
    ):
        raise AssertionError(f"Incremental map initialization mismatch at resolution {resolution}")
print("PATCH AUDIT PASSED: vectorized and full RUDY maps agree.")
'''
    ),
    code_cell(
        r'''
# Cell 10 — External Pareto archive và Pareto-safe local refiner
class ParetoArchive:
    def __init__(self, max_size: int, tolerance: float) -> None:
        self.max_size = int(max_size)
        self.tolerance = float(tolerance)
        self.entries: List[Dict[str, Any]] = []
        self.counter = 0

    def _dominates(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        tol = self.tolerance
        weak = (
            left["hpwl_ratio"] <= right["hpwl_ratio"] + tol
            and left["robust_congestion_ratio"] <= right["robust_congestion_ratio"] + tol
        )
        strict = (
            left["hpwl_ratio"] < right["hpwl_ratio"] - tol
            or left["robust_congestion_ratio"] < right["robust_congestion_ratio"] - tol
        )
        return bool(weak and strict)

    def add(self, state: SearchState, metadata: Dict[str, Any]) -> bool:
        entry = state.snapshot()
        if entry["hpwl_ratio"] > 1.0 + self.tolerance:
            return False
        if entry["robust_congestion_ratio"] > 1.0 + EPSILON_RATIO + self.tolerance:
            return False
        if any(self._dominates(existing, entry) for existing in self.entries):
            return False
        for existing in self.entries:
            if (
                abs(existing["hpwl_ratio"] - entry["hpwl_ratio"]) <= self.tolerance
                and abs(existing["robust_congestion_ratio"] - entry["robust_congestion_ratio"])
                <= self.tolerance
            ):
                return False
        self.entries = [item for item in self.entries if not self._dominates(entry, item)]
        self.counter += 1
        entry.update(metadata)
        entry["archive_id"] = f"candidate_{self.counter:04d}"
        self.entries.append(entry)
        self.entries.sort(key=lambda item: (item["hpwl_ratio"], item["robust_congestion_ratio"]))
        if len(self.entries) > self.max_size:
            # Preserve objective endpoints and evenly sample the ordered trade-off curve.
            indices = np.linspace(0, len(self.entries) - 1, self.max_size).round().astype(int)
            self.entries = [self.entries[index] for index in sorted(set(indices))]
        return True

    def frame(self) -> pd.DataFrame:
        rows = []
        for item in self.entries:
            rows.append({
                key: value for key, value in item.items()
                if key not in {"centers", "congestion_by_resolution"}
            })
        return pd.DataFrame(rows)


class ParetoSafeRefiner:
    def __init__(
        self,
        state: SearchState,
        seed: int,
        archive: ParetoArchive,
        start_id: str,
    ) -> None:
        self.state = state
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.archive = archive
        self.start_id = str(start_id)
        self.accepted_single = 0
        self.accepted_swap = 0
        self.history: List[Dict[str, Any]] = []
        self.start_time = time.perf_counter()

    def _rank(
        self,
        hpwl_ratio: float,
        congestion_ratio: float,
        current_hpwl_ratio: float,
        current_congestion_ratio: float,
    ) -> Tuple[float, ...] | None:
        tol = SEARCH_CFG.objective_tolerance
        if hpwl_ratio > 1.0 + tol:
            return None
        if congestion_ratio > 1.0 + EPSILON_RATIO + tol:
            return None

        dominates_current = (
            hpwl_ratio <= current_hpwl_ratio + tol
            and congestion_ratio <= current_congestion_ratio + tol
            and (
                hpwl_ratio < current_hpwl_ratio - tol
                or congestion_ratio < current_congestion_ratio - tol
            )
        )
        if dominates_current:
            return (0.0, hpwl_ratio + congestion_ratio, hpwl_ratio, congestion_ratio)

        current_strict = current_hpwl_ratio < 1.0 - tol and current_congestion_ratio <= 1.0 + tol
        candidate_strict = hpwl_ratio < 1.0 - tol and congestion_ratio <= 1.0 + tol
        if candidate_strict and not current_strict:
            return (1.0, hpwl_ratio, congestion_ratio)

        preserves_gain = hpwl_ratio <= 1.0 - MIN_HPWL_GAIN_RATIO + tol
        if current_congestion_ratio > 1.0 + tol and preserves_gain:
            if congestion_ratio < current_congestion_ratio - tol:
                return (2.0, congestion_ratio, hpwl_ratio)

        # ε-bridge: explore more HPWL reduction while remaining inside the pre-registered budget.
        if hpwl_ratio < current_hpwl_ratio - tol:
            return (3.0, hpwl_ratio, congestion_ratio)

        # Congestion repair can sacrifice some HPWL, but must preserve a fixed minimum gain.
        if preserves_gain and congestion_ratio < current_congestion_ratio - tol:
            return (4.0, congestion_ratio, hpwl_ratio)
        return None

    def _target_center(self, macro_id: int) -> np.ndarray:
        targets = []
        for net_id in MACRO_TO_NET_IDS[macro_id]:
            net = NETS[net_id]
            positions = np.flatnonzero(net["ids"] == macro_id)
            if not len(positions):
                continue
            local_pin = int(positions[0])
            mask = np.arange(len(net["ids"])) != local_pin
            other_pins = self.state.centers[net["ids"][mask]] + net["offsets"][mask]
            desired_pin = np.median(other_pins, axis=0)
            targets.append(desired_pin - net["offsets"][local_pin])
        if not targets:
            return self.state.centers[macro_id].copy()
        return np.median(np.asarray(targets), axis=0)

    def _cool_direction(self, macro_id: int) -> np.ndarray:
        heatmap = self.state.maps[PRIMARY_RESOLUTION]["blocked_rudy"]
        pitch = CANVAS_EXTENT / PRIMARY_RESOLUTION
        ix = int(np.clip(self.state.centers[macro_id, 0] / pitch, 0, PRIMARY_RESOLUTION - 1))
        iy = int(np.clip(self.state.centers[macro_id, 1] / pitch, 0, PRIMARY_RESOLUTION - 1))
        directions = np.asarray([
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ], dtype=np.int64)
        values = []
        for dx, dy in directions:
            nx = int(np.clip(ix + dx, 0, PRIMARY_RESOLUTION - 1))
            ny = int(np.clip(iy + dy, 0, PRIMARY_RESOLUTION - 1))
            values.append(float(heatmap[ny, nx]))
        return directions[int(np.argmin(values))].astype(np.float64)

    def _candidate_centers(self, macro_id: int, radius_cells: int) -> List[np.ndarray]:
        size = SIZES[macro_id]
        current_lower = self.state.centers[macro_id] - size / 2.0
        target_lower = self._target_center(macro_id) - size / 2.0
        radius = max(1, int(radius_cells))
        half = max(1, radius // 2)
        directions = np.asarray([
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ], dtype=np.float64)
        lowers = [target_lower]
        lowers.extend(current_lower + directions * radius * GRID_PITCH)
        lowers.extend(current_lower + directions[:4] * half * GRID_PITCH)
        cool = self._cool_direction(macro_id)
        lowers.append(current_lower + cool * radius * GRID_PITCH)

        random_needed = max(CANDIDATES_PER_MACRO - len(lowers), 0)
        if random_needed:
            offsets = self.rng.integers(-radius, radius + 1, size=(random_needed, 2))
            lowers.extend(current_lower + offsets * GRID_PITCH)

        candidate_lower = np.asarray(lowers, dtype=np.float64)
        max_lower = CANVAS_EXTENT - size
        candidate_lower = np.clip(candidate_lower, 0.0, max_lower)
        candidate_lower = np.rint(candidate_lower / GRID_PITCH) * GRID_PITCH
        candidate_lower = np.clip(candidate_lower, 0.0, max_lower)
        candidate_lower = np.unique(np.round(candidate_lower, 8), axis=0)
        centers = [lower + size / 2.0 for lower in candidate_lower]
        return [
            center for center in centers
            if not np.allclose(center, self.state.centers[macro_id], rtol=0.0, atol=1e-9)
        ]

    def _macro_order(self) -> np.ndarray:
        criticality = np.asarray([
            sum(self.state.net_values[net_id] for net_id in MACRO_TO_NET_IDS[macro_id])
            for macro_id in range(len(MACRO_NAMES))
        ], dtype=np.float64)
        heatmap = self.state.maps[PRIMARY_RESOLUTION]["blocked_rudy"]
        pitch = CANVAS_EXTENT / PRIMARY_RESOLUTION
        ix = np.clip((self.state.centers[:, 0] / pitch).astype(int), 0, PRIMARY_RESOLUTION - 1)
        iy = np.clip((self.state.centers[:, 1] / pitch).astype(int), 0, PRIMARY_RESOLUTION - 1)
        exposure = heatmap[iy, ix]
        criticality /= max(float(criticality.max()), 1e-12)
        exposure /= max(float(exposure.max()), 1e-12)
        if self.state.robust_congestion_ratio > 1.0:
            priority = 0.35 * criticality + 0.65 * exposure
        else:
            priority = 0.65 * criticality + 0.35 * exposure
        priority += self.rng.uniform(0.0, 1e-10, size=len(priority))
        return np.argsort(-priority)[:MACROS_PER_SWEEP]

    def _choose_exact(self, proposals: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        ranked_primary = []
        primary_current = (
            self.state.congestion_by_resolution[PRIMARY_RESOLUTION]
            / BASE_CONGESTION[PRIMARY_RESOLUTION]
        )
        for proposal in proposals:
            rank = self._rank(
                proposal["hpwl_ratio"],
                proposal["robust_congestion_ratio"],
                self.state.hpwl_ratio,
                primary_current,
            )
            if rank is not None:
                ranked_primary.append((rank, proposal))
        ranked_primary.sort(key=lambda item: item[0])

        exact_candidates = []
        for _, proposal in ranked_primary[: SEARCH_CFG.exact_shortlist]:
            exact = self.state.evaluate_moves(proposal["moves"], CFG.rudy_resolutions)
            if exact is None:
                continue
            rank = self._rank(
                exact["hpwl_ratio"],
                exact["robust_congestion_ratio"],
                self.state.hpwl_ratio,
                self.state.robust_congestion_ratio,
            )
            if rank is not None:
                exact_candidates.append((rank, exact))
        if not exact_candidates:
            return None
        exact_candidates.sort(key=lambda item: item[0])
        return exact_candidates[0][1]

    def try_single(self, macro_id: int, radius_cells: int) -> bool:
        proposals = []
        for center in self._candidate_centers(macro_id, radius_cells):
            candidate = self.state.evaluate_moves(
                {int(macro_id): center}, (PRIMARY_RESOLUTION,)
            )
            if candidate is not None:
                proposals.append(candidate)
        chosen = self._choose_exact(proposals)
        if chosen is None:
            return False
        self.accepted_single += 1
        move_label = f"single_seed{self.seed}_{self.accepted_single + self.accepted_swap}"
        self.state.commit(chosen, move_label)
        self.archive.add(self.state, {
            "origin": "stage2b_single",
            "search_seed": self.seed,
            "start_id": self.start_id,
            "move_type": "single",
        })
        return True

    def try_swaps(self, order: np.ndarray) -> int:
        pairs = []
        seen = set()
        for macro_id in order:
            neighbors = list(CONNECTED_MACROS[int(macro_id)])
            self.rng.shuffle(neighbors)
            for other in neighbors[:4]:
                pair = tuple(sorted((int(macro_id), int(other))))
                if pair[0] != pair[1] and pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
        self.rng.shuffle(pairs)
        accepted = 0
        for first, second in pairs[:SWAP_TRIALS]:
            lower_first = self.state.centers[first] - SIZES[first] / 2.0
            lower_second = self.state.centers[second] - SIZES[second] / 2.0
            moves = {
                first: lower_second + SIZES[first] / 2.0,
                second: lower_first + SIZES[second] / 2.0,
            }
            primary = self.state.evaluate_moves(moves, (PRIMARY_RESOLUTION,))
            if primary is None:
                continue
            chosen = self._choose_exact([primary])
            if chosen is None:
                continue
            self.accepted_swap += 1
            accepted += 1
            move_label = f"swap_seed{self.seed}_{self.accepted_single + self.accepted_swap}"
            self.state.commit(chosen, move_label)
            self.archive.add(self.state, {
                "origin": "stage2b_swap",
                "search_seed": self.seed,
                "start_id": self.start_id,
                "move_type": "swap",
            })
        return accepted

    def run(self) -> pd.DataFrame:
        for radius in RADII_CELLS:
            accepted_before = self.accepted_single + self.accepted_swap
            order = self._macro_order()
            for macro_id in order:
                self.try_single(int(macro_id), int(radius))
            swap_accepted = self.try_swaps(order)

            full_legality = legality_metrics(self.state.centers)
            exact_hpwl = hpwl_pin(self.state.centers)
            if not full_legality["legal"]:
                raise AssertionError(f"Legality failed after radius {radius}")
            if not math.isclose(
                exact_hpwl, self.state.hpwl, rel_tol=0.0, abs_tol=CFG.metric_tolerance
            ):
                raise AssertionError("Incremental HPWL drift detected")
            row = {
                "search_seed": self.seed,
                "start_id": self.start_id,
                "radius_cells": int(radius),
                "accepted_this_radius": (
                    self.accepted_single + self.accepted_swap - accepted_before
                ),
                "accepted_single_total": self.accepted_single,
                "accepted_swap_total": self.accepted_swap,
                "swap_accepted_this_radius": swap_accepted,
                "hpwl": self.state.hpwl,
                "hpwl_improvement_pct": 100.0 * (1.0 - self.state.hpwl_ratio),
                "robust_congestion_delta_pct": 100.0 * (
                    self.state.robust_congestion_ratio - 1.0
                ),
                "strictly_dominates_baseline": bool(
                    self.state.hpwl_ratio < 1.0 - SEARCH_CFG.objective_tolerance
                    and self.state.robust_congestion_ratio <= 1.0 + SEARCH_CFG.objective_tolerance
                ),
                "runtime_s": time.perf_counter() - self.start_time,
            }
            self.history.append(row)
            print(
                f"seed={self.seed} start={self.start_id:>9} radius={radius:>2} "
                f"accepted={row['accepted_this_radius']:>2} "
                f"HPWL_gain={row['hpwl_improvement_pct']:.6f}% "
                f"robust_RUDY_delta={row['robust_congestion_delta_pct']:.6f}% "
                f"strict={row['strictly_dominates_baseline']}"
            )
        return pd.DataFrame(self.history)
'''
    ),
    code_cell(
        r'''
# Cell 11 — Chạy smoke/full multi-start và xây Pareto archive
ARCHIVE = ParetoArchive(
    max_size=SEARCH_CFG.max_archive_size,
    tolerance=SEARCH_CFG.objective_tolerance,
)

INITIAL_STATES: Dict[str, SearchState] = {}
for placement_id, payload in placements.items():
    state = SearchState(payload["centers"], placement_id, MAPS[placement_id])
    INITIAL_STATES[placement_id] = state
    ARCHIVE.add(state, {
        "origin": "efficientplace" if placement_id == "baseline" else "stage1_hpwl_only",
        "search_seed": None,
        "start_id": placement_id,
        "move_type": "initial",
    })

BEST_STAGE1_ID = min(
    (placement_id for placement_id in INITIAL_STATES if placement_id != "baseline"),
    key=lambda placement_id: INITIAL_STATES[placement_id].hpwl,
)
START_IDS = ("baseline", BEST_STAGE1_ID)
print("Search starts:", START_IDS)

run_records = []
histories = []
final_states: Dict[str, SearchState] = {}
for search_seed in SEARCH_SEEDS:
    for start_id in START_IDS:
        print(f"\n===== Stage 2B seed={search_seed}, start={start_id} =====")
        source = INITIAL_STATES[start_id]
        state = SearchState(source.centers, start_id, source.maps)
        refiner = ParetoSafeRefiner(state, int(search_seed), ARCHIVE, start_id)
        history = refiner.run()
        histories.append(history)
        final_id = f"seed_{search_seed}_from_{start_id}"
        final_states[final_id] = state

        final_dir = STAGE2B_RUN_DIR / final_id
        final_dir.mkdir(parents=True, exist_ok=False)
        pd.DataFrame({
            "macro": MACRO_NAMES,
            "x_center": state.centers[:, 0],
            "y_center": state.centers[:, 1],
            "width": SIZES[:, 0],
            "height": SIZES[:, 1],
        }).to_csv(final_dir / "centers.csv", index=False)
        history.to_csv(final_dir / "history.csv", index=False)

        run_records.append({
            "search_seed": int(search_seed),
            "start_id": start_id,
            "final_id": final_id,
            "accepted_single": refiner.accepted_single,
            "accepted_swap": refiner.accepted_swap,
            "hpwl": state.hpwl,
            "hpwl_improvement_pct": 100.0 * (1.0 - state.hpwl_ratio),
            "robust_congestion_delta_pct": 100.0 * (
                state.robust_congestion_ratio - 1.0
            ),
            "strictly_dominates_baseline": bool(
                state.hpwl_ratio < 1.0 - SEARCH_CFG.objective_tolerance
                and state.robust_congestion_ratio <= 1.0 + SEARCH_CFG.objective_tolerance
            ),
            "legal": bool(legality_metrics(state.centers)["legal"]),
            "runtime_s": float(history["runtime_s"].iloc[-1]) if len(history) else 0.0,
        })

RUN_RESULTS = pd.DataFrame(run_records)
HISTORY = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
ARCHIVE_FRAME = ARCHIVE.frame()
RUN_RESULTS.to_csv(STAGE2B_RUN_DIR / "run_results.csv", index=False)
HISTORY.to_csv(STAGE2B_RUN_DIR / "search_history.csv", index=False)
ARCHIVE_FRAME.to_csv(STAGE2B_RUN_DIR / "pareto_archive.csv", index=False)

display(RUN_RESULTS)
display(ARCHIVE_FRAME.sort_values(["hpwl_ratio", "robust_congestion_ratio"]))
'''
    ),
    code_cell(
        r'''
# Cell 12 — Chọn strict/practical solution, xuất Pareto placements và đồ thị
tol = SEARCH_CFG.objective_tolerance
STRICT_ENTRIES = [
    item for item in ARCHIVE.entries
    if item["hpwl_ratio"] < 1.0 - tol
    and item["robust_congestion_ratio"] <= 1.0 + tol
]
PRACTICAL_ENTRIES = [
    item for item in ARCHIVE.entries
    if item["hpwl_ratio"] <= 1.0 + tol
    and item["robust_congestion_ratio"] <= 1.0 + EPSILON_RATIO + tol
]
BEST_STRICT = min(STRICT_ENTRIES, key=lambda item: item["hpwl_ratio"], default=None)
BEST_PRACTICAL = min(PRACTICAL_ENTRIES, key=lambda item: item["hpwl_ratio"])
SELECTED = BEST_STRICT if BEST_STRICT is not None else BEST_PRACTICAL

archive_dir = STAGE2B_RUN_DIR / "pareto_placements"
archive_dir.mkdir(parents=True, exist_ok=False)
for entry in ARCHIVE.entries:
    frame = pd.DataFrame({
        "macro": MACRO_NAMES,
        "x_center": entry["centers"][:, 0],
        "y_center": entry["centers"][:, 1],
        "width": SIZES[:, 0],
        "height": SIZES[:, 1],
    })
    frame.to_csv(archive_dir / f"{entry['archive_id']}.csv", index=False)

selected_frame = pd.DataFrame({
    "macro": MACRO_NAMES,
    "x_center": SELECTED["centers"][:, 0],
    "y_center": SELECTED["centers"][:, 1],
    "width": SIZES[:, 0],
    "height": SIZES[:, 1],
})
selected_frame.to_csv(STAGE2B_RUN_DIR / "selected_centers.csv", index=False)

figure, axes = plt.subplots(1, 3, figsize=(19, 5.5))

# Pareto archive; right and down are better.
archive_plot = ARCHIVE.frame()
x = 100.0 * (1.0 - archive_plot["hpwl_ratio"])
y = 100.0 * (archive_plot["robust_congestion_ratio"] - 1.0)
colors = [
    "tab:red" if origin == "efficientplace" else
    "tab:gray" if origin == "stage1_hpwl_only" else "tab:green"
    for origin in archive_plot["origin"]
]
axes[0].scatter(x, y, c=colors, s=55)
axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
axes[0].axhline(
    SEARCH_CFG.congestion_epsilon_pct, color="tab:orange", linestyle=":", linewidth=1.2,
    label="ε budget",
)
for _, row in archive_plot.iterrows():
    if row["origin"] != "stage2b_single" or len(archive_plot) <= 15:
        axes[0].annotate(
            row["archive_id"],
            (100.0 * (1.0 - row["hpwl_ratio"]), 100.0 * (row["robust_congestion_ratio"] - 1.0)),
            xytext=(3, 3), textcoords="offset points", fontsize=7,
        )
axes[0].set_xlabel("Pin-HPWL improvement (%) →")
axes[0].set_ylabel("Worst-resolution RUDY delta (%) ↓")
axes[0].set_title("Stage 2B Pareto archive")
axes[0].grid(alpha=0.25)
axes[0].legend()

# Search trajectories.
for (seed, start_id), group in HISTORY.groupby(["search_seed", "start_id"]):
    axes[1].plot(
        np.arange(1, len(group) + 1), group["hpwl_improvement_pct"], marker="o",
        label=f"seed {seed}/{start_id}",
    )
axes[1].set_xlabel("Completed radius")
axes[1].set_ylabel("HPWL improvement (%)")
axes[1].set_title("HPWL trajectory")
axes[1].grid(alpha=0.25)
axes[1].legend(fontsize=7)

for (seed, start_id), group in HISTORY.groupby(["search_seed", "start_id"]):
    axes[2].plot(
        np.arange(1, len(group) + 1), group["robust_congestion_delta_pct"], marker="o",
        label=f"seed {seed}/{start_id}",
    )
axes[2].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
axes[2].axhline(SEARCH_CFG.congestion_epsilon_pct, color="tab:orange", linestyle=":")
axes[2].set_xlabel("Completed radius")
axes[2].set_ylabel("Worst-resolution RUDY delta (%)")
axes[2].set_title("Congestion trajectory")
axes[2].grid(alpha=0.25)
axes[2].legend(fontsize=7)

figure.tight_layout()
pareto_plot_path = STAGE2B_RUN_DIR / "stage2b_pareto_and_trajectories.png"
figure.savefig(pareto_plot_path, dpi=180, bbox_inches="tight")
plt.show()

selected_maps = {
    int(resolution): congestion_maps(SELECTED["centers"], int(resolution))
    for resolution in CFG.rudy_resolutions
}
heatmap_ids = [
    ("baseline", MAPS["baseline"][PRIMARY_RESOLUTION]["blocked_rudy"]),
    (BEST_STAGE1_ID, MAPS[BEST_STAGE1_ID][PRIMARY_RESOLUTION]["blocked_rudy"]),
    ("stage2b_selected", selected_maps[PRIMARY_RESOLUTION]["blocked_rudy"]),
]
vmax = float(np.quantile(np.concatenate([value.ravel() for _, value in heatmap_ids]), 0.995))
figure, axes = plt.subplots(1, 3, figsize=(18, 5))
for axis, (label, values) in zip(axes, heatmap_ids):
    image = axis.imshow(values, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
    axis.set_title(label)
    axis.set_xlabel("bin x")
    axis.set_ylabel("bin y")
figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, label="capacity-adjusted RUDY")
figure.suptitle(f"Blocked-RUDY comparison ({PRIMARY_RESOLUTION}×{PRIMARY_RESOLUTION})")
heatmap_path = STAGE2B_RUN_DIR / "stage2b_blocked_rudy_heatmaps.png"
figure.savefig(heatmap_path, dpi=180, bbox_inches="tight")
plt.show()

print("Selected archive id:", SELECTED["archive_id"])
print("Strict solution     :", BEST_STRICT is not None)
print("Pareto plot         :", pareto_plot_path)
print("Heatmap             :", heatmap_path)
'''
    ),
    code_cell(
        r'''
# Cell 13 — Manifest, scientific gate và archive
selected_legality = legality_metrics(SELECTED["centers"])
selected_hpwl_exact = hpwl_pin(SELECTED["centers"])
assert selected_legality["legal"]
assert math.isclose(
    selected_hpwl_exact, SELECTED["hpwl"], rel_tol=0.0, abs_tol=CFG.metric_tolerance
)
assert SELECTED["hpwl_ratio"] <= 1.0 + SEARCH_CFG.objective_tolerance
assert (
    SELECTED["robust_congestion_ratio"]
    <= 1.0 + EPSILON_RATIO + SEARCH_CFG.objective_tolerance
)

manifest = {
    "experiment": "stage2b_pareto_safe_refinement",
    "evaluation_config": asdict(CFG),
    "search_config": asdict(SEARCH_CFG),
    "metric_scope": METRIC_SCOPE,
    "full_chip_comparable": False,
    "checkpoint_x_csv_sha256": actual_checkpoint_hash,
    "stage1_manifest_sha256": sha256_file(STAGE1_MANIFEST_PATH),
    "search_starts": list(START_IDS),
    "archive_size": len(ARCHIVE.entries),
    "strict_solution_found": BEST_STRICT is not None,
    "selected": {
        key: value for key, value in SELECTED.items()
        if key not in {"centers", "congestion_by_resolution"}
    },
    "selected_congestion_by_resolution": SELECTED["congestion_by_resolution"],
    "selected_legality": selected_legality,
    "selected_hpwl_improvement_pct": 100.0 * (1.0 - SELECTED["hpwl_ratio"]),
    "selected_robust_congestion_delta_pct": 100.0 * (
        SELECTED["robust_congestion_ratio"] - 1.0
    ),
    "all_run_outputs_legal": bool(RUN_RESULTS["legal"].all()),
}
manifest_path = STAGE2B_RUN_DIR / "run_manifest_stage2b.json"
manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")

report_lines = [
    "# Stage 2B result — bigblue1",
    "",
    f"- Run mode: `{SEARCH_CFG.run_mode}`",
    f"- Strict solution found: `{BEST_STRICT is not None}`",
    f"- Selected archive id: `{SELECTED['archive_id']}`",
    f"- Selected HPWL improvement: `{100.0 * (1.0 - SELECTED['hpwl_ratio']):.6f}%`",
    f"- Selected worst-resolution RUDY delta: `{100.0 * (SELECTED['robust_congestion_ratio'] - 1.0):.6f}%`",
    f"- Overlap pairs: `{selected_legality['overlap_pairs']}`",
    f"- Boundary violations: `{selected_legality['boundary_violations']}`",
    f"- Pareto archive size: `{len(ARCHIVE.entries)}`",
    "",
    "Scientific guard: macro-only HPWL and deterministic RUDY proxy; not full-chip placement/routing.",
]
report_path = STAGE2B_RUN_DIR / "stage2b_report.md"
report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

archive_path = shutil.make_archive(str(STAGE2B_RUN_DIR), "zip", root_dir=STAGE2B_RUN_DIR)
final_summary = {
    "run_mode": SEARCH_CFG.run_mode,
    "strict_solution_found": BEST_STRICT is not None,
    "selected_archive_id": SELECTED["archive_id"],
    "selected_hpwl_improvement_pct": 100.0 * (1.0 - SELECTED["hpwl_ratio"]),
    "selected_robust_congestion_delta_pct": 100.0 * (
        SELECTED["robust_congestion_ratio"] - 1.0
    ),
    "archive_size": len(ARCHIVE.entries),
    "overlap_pairs": selected_legality["overlap_pairs"],
    "boundary_violations": selected_legality["boundary_violations"],
    "full_chip_comparable": False,
    "archive": archive_path,
}
print(json.dumps(final_summary, indent=2))
print("\nNEXT ACTION:")
if SEARCH_CFG.run_mode == "smoke":
    print("Send run_manifest_stage2b.json, run_results.csv, pareto_archive.csv and both PNG files.")
    print("Do not switch to full until the incremental-map and scientific gates are audited.")
else:
    print("Proceed to downstream standard-cell placement only after cross-seed audit.")
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
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {DESTINATION}")

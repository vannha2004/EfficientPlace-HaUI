from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "cplex_resolver" / "adaptec1_quality_max_hybrid.ipynb"
BASE_CANDIDATES = (
    ROOT / "cplex_resolver" / "adaptec1_drl_pdca_cplex_workingset.ipynb",
    DESTINATION,
    ROOT / "cplex_resolver" / "adaptec1_drl_pdca_cplex.ipynb",
)
BASE = next((path for path in BASE_CANDIDATES if path.exists()), None)
if BASE is None:
    raise FileNotFoundError("No source notebook is available for cells 1-5")


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


base = json.loads(BASE.read_text(encoding="utf-8"))
cells = []
for cell in base["cells"]:
    copied = json.loads(json.dumps(cell))
    copied["execution_count"] = None if copied["cell_type"] == "code" else copied.get("execution_count")
    copied["outputs"] = [] if copied["cell_type"] == "code" else copied.get("outputs", [])
    cells.append(copied)
    source = "".join(copied.get("source", []))
    if source.startswith("# Cell 5"):
        break

cells[0] = markdown_cell(
    r'''
# adaptec1 — quality-max hybrid: hard legalization + CPLEX LP/LNS-MIP

Notebook này ưu tiên **nghiệm hợp lệ tuyệt đối trước, HPWL sau**:

1. tùy chọn huấn luyện EfficientPlace dài 2.000 loops, 5 seeds, grid 256;
2. thu thập checkpoint DRL/DCA/EfficientPlace và chọn anchor tốt;
3. Tetris site-row nhiều restart tạo nghiệm `overlap = 0`, `boundary = 0`;
4. CPLEX LP giảm pin-HPWL bằng hard separation một hướng/cặp và separation loop;
5. CPLEX LNS-MIP thay đổi topology của 128 cặp nghẽn nhất qua 3 vòng;
6. nghiệm xuất cuối được full-pair audit và báo cáo theo `F(x) = HPWL / 10^5`.

Thiết kế dựa trên [EfficientPlace (ICML 2024)](https://proceedings.mlr.press/v235/geng24b.html),
[mã nguồn EfficientPlace](https://github.com/MIRALab-USTC/AI4EDA-EfficientPlace) và flow
macro legalization/refinement của [DREAMPlace](https://github.com/limbo018/DREAMPlace).

Mặc định `RUN_LONG_TRAINING=False` để notebook vẫn chạy được trong kernel CPLEX hiện tại.
Đổi thành `True` khi `TRAIN_PYTHON` trỏ tới môi trường đã có PyTorch + dependencies của EfficientPlace.
'''
)

for cell in cells:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    source = source.replace('"workingset_pin_hpwl"', '"quality_max_hybrid"')
    if "import sys\n" not in source:
        source = source.replace("import time\n", "import time\nimport sys\n")
    source = re.sub(r"(?:import sys\n){2,}", "import sys\n", source)
    matplotlib_block = (
        "import matplotlib\n"
        "if 'ipykernel' not in sys.modules:\n"
        "    matplotlib.use('Agg')\n"
    )
    while source.count(matplotlib_block) > 1:
        source = source.replace(matplotlib_block + matplotlib_block, matplotlib_block)
    if "import matplotlib.pyplot as plt" in source and matplotlib_block not in source:
        source = source.replace(
            "import matplotlib.pyplot as plt",
            matplotlib_block + "import matplotlib.pyplot as plt",
        )
    if source.startswith("# Cell 5"):
        function_start = (
            source.index("FULL_PAIR_AUDIT_CHUNK_SIZE")
            if "FULL_PAIR_AUDIT_CHUNK_SIZE" in source
            else source.index("def full_metrics(")
        )
        function_end = source.index("\n\nINITIAL_METRICS_BASE_GAMMA")
        chunked_metrics = r'''FULL_PAIR_AUDIT_CHUNK_SIZE = 65_536


def chunked_full_pair_audit(
    centers_local: np.ndarray,
    chunk_size: int = FULL_PAIR_AUDIT_CHUNK_SIZE,
) -> Dict[str, Any]:
    """Audit every pair in NumPy batches; never create pairwise CPLEX binaries."""
    overlap_pairs = 0
    overlap_hat_sum = 0.0
    overlap_hat_max = 0.0
    nij_min = float("inf")
    nij_max = -float("inf")
    for start in range(0, PAIR_I.size, chunk_size):
        stop = min(PAIR_I.size, start + chunk_size)
        pair = pair_values(
            centers_local,
            PAIR_I[start:stop],
            PAIR_J[start:stop],
            PAIR_W[start:stop],
            PAIR_H[start:stop],
        )
        overlap_pairs += int((pair["Nij"] < 0.0).sum())
        overlap_hat_sum += float(pair["phi"].sum())
        if pair["phi"].size:
            overlap_hat_max = max(overlap_hat_max, float(pair["phi"].max()))
            nij_min = min(nij_min, float(pair["Nij"].min()))
            nij_max = max(nij_max, float(pair["Nij"].max()))
    if PAIR_I.size == 0:
        nij_min = nij_max = 0.0
    return {
        "total_pairs": int(PAIR_I.size),
        "overlap_pairs": overlap_pairs,
        "overlap_hat_sum": overlap_hat_sum,
        "overlap_hat_max": overlap_hat_max,
        "Nij_min": nij_min,
        "Nij_max": nij_max,
    }


def full_metrics(
    centers_local: np.ndarray,
    gamma_boundary: float,
    gamma_overlap: float,
) -> Dict[str, Any]:
    centers_local = np.asarray(centers_local, dtype=np.float64)
    if centers_local.shape != (DB.num_macros, 2):
        raise ValueError(f"Unexpected placement shape {centers_local.shape}")
    pair = chunked_full_pair_audit(centers_local)
    boundary = boundary_penalty(centers_local)
    hpwl_pin = pin_hpwl(centers_local)
    hpwl_center = center_hpwl_audit(centers_local)
    overlap_sum = float(pair["overlap_hat_sum"])
    boundary_sum = float(boundary.sum())
    return {
        "hpwl_pin": hpwl_pin,
        "hpwl_center_audit": hpwl_center,
        **pair,
        "boundary_violating_macros": int((boundary > 0.0).sum()),
        "boundary_penalty_sum": boundary_sum,
        "weighted_overlap_penalty": gamma_overlap * overlap_sum,
        "weighted_boundary_penalty": gamma_boundary * boundary_sum,
        "penalized_objective": (
            hpwl_pin + gamma_overlap * overlap_sum + gamma_boundary * boundary_sum
        ),
    }
'''
        source = source[:function_start] + chunked_metrics + source[function_end:]
    cell["source"] = source.splitlines(keepends=True)

cells.extend(
    [
        markdown_cell(
            r'''
## 1. Chế độ huấn luyện dài và multi-seed

Cấu hình cũ của checkpoint hiện tại là grid 32, 800 updates. Cấu hình chính thức của adaptec1
trong repository EfficientPlace là grid 256, 1.000 loops × 5 episodes/loop và 128 macro đầu
được điều khiển bởi RL. Chế độ dưới đây tăng lên 2.000 loops và chạy 5 seeds tuần tự.

Huấn luyện lâu hơn chỉ có ích khi vẫn giữ global tree search/solution pool và chọn best-of-seeds;
không nên chỉ tăng penalty DCA vì DCA đã dừng ở một local minimum còn overlap.
'''
        ),
        code_cell(
            r'''
# Cell 6 - Optional research-grade EfficientPlace training launcher
import importlib.util
import pickle
import subprocess
import sys

RUN_LONG_TRAINING = False
TRAIN_PYTHON = sys.executable  # Change to the Python executable that has torch/hydra.
LONG_TRAIN_LOOPS = 2_000
LONG_TRAIN_SEEDS = (0, 1, 2, 3, 4)
SKIP_COMPLETED_SEEDS = True

REPO_ROOT = PROJECT_ROOT.parent
LONG_TRAIN_ROOT = (
    PROJECT_ROOT / "experiments" / BENCHMARK / "efficientplace_long_grid256"
)
LONG_TRAIN_ROOT.mkdir(parents=True, exist_ok=True)


def assert_training_environment(python_executable: str) -> None:
    probe = (
        "import torch, hydra, omegaconf, treelib, tensorboardX; "
        "print(torch.__version__)"
    )
    completed = subprocess.run(
        [python_executable, "-c", probe],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "TRAIN_PYTHON is not an EfficientPlace training environment.\n"
            f"Executable: {python_executable}\n"
            f"stderr: {completed.stderr.strip()}"
        )


def run_long_training() -> None:
    assert_training_environment(TRAIN_PYTHON)
    benchmark_dir = BENCH_DIR.resolve().as_posix()
    for seed in LONG_TRAIN_SEEDS:
        seed_dir = LONG_TRAIN_ROOT / f"seed_{seed}"
        solution_dir = seed_dir / "sol"
        if SKIP_COMPLETED_SEEDS and list(solution_dir.glob("best_placement_*.pkl")):
            log_msg(f"Skip completed long-training seed {seed}: {solution_dir}")
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(TRAIN_PYTHON),
            "main.py",
            "benchmark@_global_=adaptec1",
            f"seed={seed}",
            f"trainer.num_loops={LONG_TRAIN_LOOPS}",
            f"env.benchmark_dir={benchmark_dir}",
            f"hydra.run.dir={seed_dir.resolve().as_posix()}",
            f"tb_dir={(seed_dir / 'tb').resolve().as_posix()}",
            f"model_dir={(seed_dir / 'model').resolve().as_posix()}",
            f"solution_dir={solution_dir.resolve().as_posix()}",
        ]
        log_msg("Long training command: " + " ".join(command))
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if RUN_LONG_TRAINING:
    run_long_training()
else:
    log_msg(
        "Long training is disabled. Set RUN_LONG_TRAINING=True and TRAIN_PYTHON "
        "to a PyTorch EfficientPlace environment to run 5 seeds."
    )
'''
        ),
        markdown_cell(
            r'''
## 2. Candidate pool

Notebook không ghi đè kết quả cũ. Nó đọc:

- checkpoint grid 32 hiện tại;
- mọi nghiệm `z_star` từ các lần working-set DCA đã hoàn tất;
- mọi `best_placement_*.pkl` từ long-training ở trên;
- mọi top-K DRL fine-tune đã qua ngưỡng acceptance sau legalization;
- nghiệm quality-max hợp lệ của các lần chạy trước (nếu có).

Mỗi candidate đều được audit bằng cùng pin offsets và full-pair metric trước khi chọn anchor.
'''
        ),
        code_cell(
            r'''
# Cell 7 - Build and audit a candidate pool

def official_ranked_macro_names() -> List[str]:
    areas = WIDTHS * HEIGHTS
    area_sum = np.zeros(DB.num_macros, dtype=np.float64)
    for net in DB.nets:
        ids = np.unique(net.macro_ids)
        net_area = float(areas[ids].sum())
        area_sum[ids] += net_area
    order = sorted(
        range(DB.num_macros),
        key=lambda i: (-float(areas[i]), -float(area_sum[i])),
    )
    return [DB.macros[i].name for i in order]


def official_canvas_side() -> float:
    max_x = 0.0
    max_y = 0.0
    source = BENCH_DIR / f"{BENCHMARK}.pl"
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            fields = raw.strip().split()
            if len(fields) < 3 or fields[0] not in DB.macro_name_to_id:
                continue
            i = DB.macro_name_to_id[fields[0]]
            max_x = max(max_x, float(fields[1]) + WIDTHS[i])
            max_y = max(max_y, float(fields[2]) + HEIGHTS[i])
    return min(max_x, max_y)


OFFICIAL_RANKED_NAMES = official_ranked_macro_names()
OFFICIAL_SIDE = official_canvas_side()


def efficientplace_pickle_to_centers(path: Path, grid: int = 256) -> np.ndarray:
    with path.open("rb") as handle:
        actions = np.asarray(pickle.load(handle), dtype=np.int64).reshape(-1)
    if actions.size != DB.num_macros:
        raise ValueError(f"{path} has {actions.size} actions; expected {DB.num_macros}")
    centers = np.zeros((DB.num_macros, 2), dtype=np.float64)
    ratio = OFFICIAL_SIDE / grid
    for rank, name in enumerate(OFFICIAL_RANKED_NAMES):
        i = DB.macro_name_to_id[name]
        action = int(actions[rank])
        lower_x = (action // grid) * ratio
        lower_y = (action % grid) * ratio
        centers[i] = (lower_x + WIDTHS[i] / 2.0, lower_y + HEIGHTS[i] / 2.0)
    return centers


def candidate_score(metrics: Dict[str, Any]) -> float:
    return float(
        metrics["hpwl_pin"]
        + 100_000.0 * metrics["overlap_hat_sum"]
        + 10_000.0 * metrics["boundary_penalty_sum"]
    )


CANDIDATES: List[Dict[str, Any]] = []


def add_candidate(name: str, centers: np.ndarray, source: Path) -> None:
    centers = np.asarray(centers, dtype=np.float64)
    metrics = full_metrics(centers, 0.0, 0.0)
    CANDIDATES.append({
        "name": name,
        "source": str(source),
        "centers": centers,
        "metrics": metrics,
        "score": candidate_score(metrics),
    })


add_candidate("current_grid32", Z0_PAPER_CENTERS, PLACEMENT_GRID_PATH)

working_root = PROJECT_ROOT / "experiments" / BENCHMARK / "workingset_pin_hpwl"
for path in sorted(working_root.glob("run_*/placements/z_star_paper_centers.npy")):
    try:
        add_candidate(f"dca_{path.parents[1].name}", np.load(path), path)
    except Exception as exc:
        log_msg(f"Skip invalid DCA candidate {path}: {exc}")

quality_root = PROJECT_ROOT / "experiments" / BENCHMARK / "quality_max_hybrid"
for path in sorted(quality_root.glob("run_*/placements/z_quality_max_centers.npy")):
    if EXP_DIR in path.parents:
        continue
    try:
        add_candidate(f"quality_{path.parents[1].name}", np.load(path), path)
    except Exception as exc:
        log_msg(f"Skip invalid quality candidate {path}: {exc}")

for path in sorted(LONG_TRAIN_ROOT.glob("seed_*/sol/best_placement_*.pkl")):
    try:
        add_candidate(f"efficientplace_{path.parents[1].name}_{path.stem}",
                      efficientplace_pickle_to_centers(path), path)
    except Exception as exc:
        log_msg(f"Skip invalid EfficientPlace candidate {path}: {exc}")

FINETUNE_ROOT = PROJECT_ROOT / "experiments" / BENCHMARK / "drl_checkpoint_finetune"
for metrics_path in sorted(FINETUNE_ROOT.glob("seed_*/top_k/rank_*/metrics.json")):
    try:
        metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not metadata.get("accepted_for_cplex", False):
            log_msg(f"Skip unaccepted DRL fine-tune candidate: {metrics_path}")
            continue
        centers_path = metrics_path.parent / "placement_centers_local.npy"
        if not centers_path.is_file():
            raise FileNotFoundError(centers_path)
        seed_name = metrics_path.parents[2].name
        rank_name = metrics_path.parent.name
        add_candidate(
            f"drl_finetune_{seed_name}_{rank_name}",
            np.load(centers_path),
            centers_path,
        )
    except Exception as exc:
        log_msg(f"Skip invalid DRL fine-tune candidate {metrics_path}: {exc}")

if not CANDIDATES:
    raise RuntimeError("No placement candidate was found")

CANDIDATES.sort(key=lambda item: item["score"])
F_SCALE = 100_000.0


def f_value(metrics: Dict[str, Any]) -> float:
    """Reported objective scale: F(x) = pin HPWL / 1e5."""
    return float(metrics["hpwl_pin"] / F_SCALE)


print(f"{'candidate':<48} {'F(x)=HPWL/1e5':>16} {'overlaps':>10} {'boundary':>10} {'score':>16}")
print("-" * 104)
for item in CANDIDATES:
    metrics = item["metrics"]
    print(
        f"{item['name']:<48} {f_value(metrics):>16.7f} "
        f"{metrics['overlap_pairs']:>10d} {metrics['boundary_violating_macros']:>10d} "
        f"{item['score']:>16.3f}"
    )

MAX_ANCHORS_TO_LEGALIZE = 2
ANCHOR_CANDIDATES = CANDIDATES[:1]
FINE_TUNE_CANDIDATES = [
    item for item in CANDIDATES if item["name"].startswith("drl_finetune_")
]
if FINE_TUNE_CANDIDATES and FINE_TUNE_CANDIDATES[0] not in ANCHOR_CANDIDATES:
    ANCHOR_CANDIDATES.append(FINE_TUNE_CANDIDATES[0])
ANCHOR_CANDIDATES = ANCHOR_CANDIDATES[:MAX_ANCHORS_TO_LEGALIZE]
log_msg("Selected anchors: " + ", ".join(item["name"] for item in ANCHOR_CANDIDATES))
'''
        ),
        markdown_cell(
            r'''
## 3. Hard legalization bằng Tetris site-row

Đây là “safety net” mà penalty DCA không có. Macro được rasterize theo site 12×12,
đặt macro lớn trước và chọn vị trí trống gần anchor nhất. Vì mỗi ô chỉ thuộc một macro,
kết quả có chứng nhận hình học `overlap_pairs == 0`; full-pair audit vẫn chạy sau mỗi restart.
'''
        ),
        code_cell(
            r'''
# Cell 8 - Multi-anchor, multi-restart Tetris legalizer
TETRIS_SITE = 12.0
TETRIS_RESTARTS = 5
TETRIS_SEED = 20260919


def tetris_grid_legalize(reference: np.ndarray, order: np.ndarray,
                         site: float = TETRIS_SITE) -> Optional[np.ndarray]:
    cells_x = int(np.floor(DB.canvas_width / site))
    cells_y = int(np.floor(DB.canvas_height / site))
    cell_w = np.ceil(WIDTHS / site).astype(np.int64)
    cell_h = np.ceil(HEIGHTS / site).astype(np.int64)
    occupancy = np.zeros((cells_y, cells_x), dtype=np.uint8)
    lower = np.zeros((DB.num_macros, 2), dtype=np.int64)

    for macro_id in order:
        i = int(macro_id)
        width_cells = int(cell_w[i])
        height_cells = int(cell_h[i])
        integral = np.pad(occupancy, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        window_sum = (
            integral[height_cells:, width_cells:]
            - integral[:-height_cells, width_cells:]
            - integral[height_cells:, :-width_cells]
            + integral[:-height_cells, :-width_cells]
        )
        free_y, free_x = np.nonzero(window_sum == 0)
        if free_x.size == 0:
            return None

        target_x = np.clip(
            (reference[i, 0] - WIDTHS[i] / 2.0) / site,
            0,
            cells_x - width_cells,
        )
        target_y = np.clip(
            (reference[i, 1] - HEIGHTS[i] / 2.0) / site,
            0,
            cells_y - height_cells,
        )
        distance = np.abs(free_x - target_x) + np.abs(free_y - target_y)
        pick = int(np.argmin(distance))
        x0, y0 = int(free_x[pick]), int(free_y[pick])
        lower[i] = (x0, y0)
        occupancy[y0:y0 + height_cells, x0:x0 + width_cells] = 1

    centers = lower.astype(np.float64) * site
    centers += np.column_stack((WIDTHS, HEIGHTS)) / 2.0
    return centers


def tetris_orders(rng: np.random.Generator) -> List[np.ndarray]:
    area = WIDTHS * HEIGHTS
    orders = [
        np.argsort(-area),
        np.lexsort((-area, -np.maximum(WIDTHS, HEIGHTS))),
    ]
    while len(orders) < TETRIS_RESTARTS:
        jitter = rng.normal(0.0, 0.03, DB.num_macros)
        orders.append(np.argsort(-area * (1.0 + jitter)))
    return orders


rng = np.random.default_rng(TETRIS_SEED)
LEGAL_CANDIDATES: List[Dict[str, Any]] = []
for anchor_item in ANCHOR_CANDIDATES:
    reference = anchor_item["centers"]
    metrics = anchor_item["metrics"]
    if metrics["overlap_pairs"] == 0 and metrics["boundary_violating_macros"] == 0:
        LEGAL_CANDIDATES.append({
            "name": anchor_item["name"] + "_already_legal",
            "centers": reference.copy(),
            "metrics": metrics,
        })
        log_msg(
            f"Skip Tetris restarts for already-legal anchor={anchor_item['name']}; "
            f"F(x)={f_value(metrics):.7f}"
        )
        continue
    for restart, order in enumerate(tetris_orders(rng)):
        started = time.perf_counter()
        candidate = tetris_grid_legalize(reference, order)
        if candidate is None:
            log_msg(f"Tetris failed: anchor={anchor_item['name']}, restart={restart}")
            continue
        candidate_metrics = full_metrics(candidate, 0.0, 0.0)
        elapsed = time.perf_counter() - started
        log_msg(
            f"Tetris anchor={anchor_item['name']} restart={restart}: "
            f"F(x)={f_value(candidate_metrics):.7f}, "
            f"overlap={candidate_metrics['overlap_pairs']}, "
            f"boundary={candidate_metrics['boundary_violating_macros']}, "
            f"time={elapsed:.2f}s"
        )
        if (
            candidate_metrics["overlap_pairs"] == 0
            and candidate_metrics["boundary_violating_macros"] == 0
        ):
            LEGAL_CANDIDATES.append({
                "name": f"{anchor_item['name']}_tetris_{restart}",
                "centers": candidate,
                "metrics": candidate_metrics,
            })

if not LEGAL_CANDIDATES:
    raise RuntimeError("Tetris could not construct a legal adaptec1 placement")

LEGAL_CANDIDATES.sort(key=lambda item: item["metrics"]["hpwl_pin"])
TETRIS_BEST = LEGAL_CANDIDATES[0]["centers"].copy()
TETRIS_BEST_METRICS = LEGAL_CANDIDATES[0]["metrics"]
np.save(PLACEMENTS_DIR / "z_tetris_legal_centers.npy", TETRIS_BEST)
log_msg(
    f"Best hard-legal candidate={LEGAL_CANDIDATES[0]['name']}, "
    f"F(x)={f_value(TETRIS_BEST_METRICS):.7f}"
)
'''
        ),
        markdown_cell(
            r'''
## 4. CPLEX LP pin-HPWL polish với hard separation

Từ một nghiệm đã hợp lệ, mỗi cặp gần nhau nhận đúng **một** hướng separation đang hợp lệ.
LP tối ưu pin-HPWL trong trust region. Nếu nghiệm mới tạo overlap với cặp chưa có trong model,
full-pair NumPy audit thêm cặp đó rồi giải lại. Một nghiệm chỉ được nhận khi full audit trả về 0.

Cách này dùng khoảng 1–2 nghìn constraints pair cho adaptec1 thay vì 147.153 cặp × 4 constraints,
nên tránh nút thắt RAM của full-pairs.
'''
        ),
        code_cell(
            r'''
# Cell 9 - Hard-separation LP polish with expanding trust regions
POLISH_RADIUS_SCHEDULE = (0.02, 0.04, 0.08, 0.12, 0.16)
POLISH_MOVE_WEIGHT = 0.02
POLISH_NEAR_MARGIN = 0.02
POLISH_MAX_REFINEMENTS = 12
POLISH_CLEARANCE = 1e-3
POLISH_HISTORY: List[Dict[str, Any]] = []


def polish_legal_placement(
    base: np.ndarray,
    radius_fraction: float,
    move_weight: float = POLISH_MOVE_WEIGHT,
) -> np.ndarray:
    radius = radius_fraction * min(DB.canvas_width, DB.canvas_height)
    base_pair = pair_values(base)
    keys: Set[int] = {
        int(key) for key in PAIR_KEYS[base_pair["Nij"] <= POLISH_NEAR_MARGIN]
    }

    def direction_for(key: int) -> str:
        k = PAIR_KEY_TO_INDEX[key]
        i, j = int(PAIR_I[k]), int(PAIR_J[k])
        delta_x = float(base[i, 0] - base[j, 0])
        delta_y = float(base[i, 1] - base[j, 1])
        ratio_x = abs(delta_x) / float(PAIR_W[k])
        ratio_y = abs(delta_y) / float(PAIR_H[k])
        if ratio_x >= ratio_y:
            return "x_le" if delta_x <= 0 else "x_ge"
        return "y_le" if delta_y <= 0 else "y_ge"

    directions = {key: direction_for(key) for key in keys}
    for refinement in range(POLISH_MAX_REFINEMENTS):
        model = Model(name=f"quality_max_lp_{radius_fraction}_{refinement}")
        model.parameters.threads = CPLEX_THREADS
        infinity = model.infinity

        x = [
            model.continuous_var(
                lb=max(float(WIDTHS[i] / 2.0), float(base[i, 0] - radius)),
                ub=min(float(DB.canvas_width - WIDTHS[i] / 2.0), float(base[i, 0] + radius)),
                name=f"x_{i}",
            )
            for i in range(DB.num_macros)
        ]
        y = [
            model.continuous_var(
                lb=max(float(HEIGHTS[i] / 2.0), float(base[i, 1] - radius)),
                ub=min(float(DB.canvas_height - HEIGHTS[i] / 2.0), float(base[i, 1] + radius)),
                name=f"y_{i}",
            )
            for i in range(DB.num_macros)
        ]
        xmax = model.continuous_var_list(len(DB.nets), lb=-infinity, name="xmax")
        xmin = model.continuous_var_list(len(DB.nets), lb=-infinity, name="xmin")
        ymax = model.continuous_var_list(len(DB.nets), lb=-infinity, name="ymax")
        ymin = model.continuous_var_list(len(DB.nets), lb=-infinity, name="ymin")
        dx = model.continuous_var_list(DB.num_macros, lb=0.0, name="dx")
        dy = model.continuous_var_list(DB.num_macros, lb=0.0, name="dy")

        constraints = []
        for net_id, net in enumerate(DB.nets):
            for macro_id, offset in zip(net.pin_macro_ids, net.pin_offsets):
                i = int(macro_id)
                ox, oy = float(offset[0]), float(offset[1])
                constraints.extend([
                    xmax[net_id] >= x[i] + ox,
                    xmin[net_id] <= x[i] + ox,
                    ymax[net_id] >= y[i] + oy,
                    ymin[net_id] <= y[i] + oy,
                ])
        for i in range(DB.num_macros):
            constraints.extend([
                dx[i] >= x[i] - float(base[i, 0]),
                dx[i] >= float(base[i, 0]) - x[i],
                dy[i] >= y[i] - float(base[i, 1]),
                dy[i] >= float(base[i, 1]) - y[i],
            ])
        model.add_constraints(constraints)

        for key in sorted(keys):
            k = PAIR_KEY_TO_INDEX[key]
            i, j = int(PAIR_I[k]), int(PAIR_J[k])
            direction = directions[key]
            if direction == "x_le":
                model.add_constraint(x[i] - x[j] <= -float(PAIR_W[k]) - POLISH_CLEARANCE)
            elif direction == "x_ge":
                model.add_constraint(x[j] - x[i] <= -float(PAIR_W[k]) - POLISH_CLEARANCE)
            elif direction == "y_le":
                model.add_constraint(y[i] - y[j] <= -float(PAIR_H[k]) - POLISH_CLEARANCE)
            else:
                model.add_constraint(y[j] - y[i] <= -float(PAIR_H[k]) - POLISH_CLEARANCE)

        hpwl = model.sum(
            xmax[e] - xmin[e] + ymax[e] - ymin[e] for e in range(len(DB.nets))
        )
        displacement = model.sum(dx) + model.sum(dy)
        model.minimize(hpwl + move_weight * displacement)
        solution = model.solve(log_output=False)
        status = str(model.solve_details.status)
        if solution is None:
            model.end()
            log_msg(f"LP polish failed: radius={radius_fraction}, status={status}")
            return base

        candidate = np.column_stack((
            [solution.get_value(variable) for variable in x],
            [solution.get_value(variable) for variable in y],
        ))
        model.end()
        metrics = full_metrics(candidate, 0.0, 0.0)
        overlap_mask = pair_values(candidate)["Nij"] < -1e-9
        new_keys = [
            int(key) for key in PAIR_KEYS[overlap_mask] if int(key) not in keys
        ]
        for key in new_keys:
            keys.add(key)
            directions[key] = direction_for(key)

        row = {
            "radius_fraction": radius_fraction,
            "refinement": refinement,
            "status": status,
            "pair_constraints": len(keys),
            "new_pairs": len(new_keys),
            "hpwl_pin": metrics["hpwl_pin"],
            "F_x": f_value(metrics),
            "overlap_pairs": metrics["overlap_pairs"],
            "boundary_violating_macros": metrics["boundary_violating_macros"],
        }
        POLISH_HISTORY.append(row)
        log_msg("LP polish: " + json.dumps(row))
        if not new_keys and metrics["boundary_violating_macros"] == 0:
            return candidate
    return base


Z_QUALITY = TETRIS_BEST.copy()
for radius_fraction in POLISH_RADIUS_SCHEDULE:
    trial = polish_legal_placement(Z_QUALITY, radius_fraction)
    trial_metrics = full_metrics(trial, 0.0, 0.0)
    current_metrics = full_metrics(Z_QUALITY, 0.0, 0.0)
    if (
        trial_metrics["overlap_pairs"] == 0
        and trial_metrics["boundary_violating_macros"] == 0
        and trial_metrics["hpwl_pin"] < current_metrics["hpwl_pin"] - 1e-6
    ):
        Z_QUALITY = trial
        log_msg(
            f"Accepted radius={radius_fraction}: "
            f"F(x) {f_value(current_metrics):.7f} -> {f_value(trial_metrics):.7f}"
        )
    else:
        log_msg(f"Rejected radius={radius_fraction}: no legal F(x) improvement")

FINAL_METRICS = full_metrics(Z_QUALITY, 0.0, 0.0)
if FINAL_METRICS["overlap_pairs"] != 0:
    raise AssertionError("Final full-pair audit found overlap")
if FINAL_METRICS["boundary_violating_macros"] != 0:
    raise AssertionError("Final full-pair audit found a boundary violation")
log_msg(
    f"Pre-LNS legal result: F(x)={f_value(FINAL_METRICS):.7f}, "
    f"overlap={FINAL_METRICS['overlap_pairs']}, "
    f"boundary={FINAL_METRICS['boundary_violating_macros']}"
)
'''
        ),
        markdown_cell(
            r'''
## 5. Topology-changing LNS-MIP

LP ở trên giữ nguyên hướng separation của từng cặp nên sớm chạm local minimum. Pha này
vẫn chỉ dùng **một hard-separation cho mỗi cặp**, nhưng cho phép 128 cặp gần và có áp lực
HPWL lớn nhất tự chọn lại một trong bốn quan hệ trái/phải/dưới/trên bằng biến nhị phân.

Đây là large-neighborhood search (LNS): sau mỗi vòng, nghiệm tốt hơn trở thành tâm của vòng
tiếp theo và tập cặp critical được tính lại từ subgradient pin-HPWL. Cấu hình mặc định chạy
3 vòng, mỗi vòng tối đa 120 giây. Mọi incumbent chỉ được nhận sau full-pair audit.
'''
        ),
        code_cell(
            r'''
# Cell 10 - Topology-changing LNS-MIP with one separation constraint per pair
import gc

RUN_TOPOLOGY_LNS = True
TOPOLOGY_LNS_ROUNDS = 3
TOPOLOGY_LNS_CRITICAL_PAIRS = 128
TOPOLOGY_LNS_NEAR_MARGIN = 0.02
TOPOLOGY_LNS_TIME_LIMIT_SEC = 120
TOPOLOGY_LNS_MIP_GAP = 1e-3
TOPOLOGY_LNS_DISPLACEMENT_WEIGHT = 1e-4
TOPOLOGY_LNS_MIN_REL_IMPROVEMENT = 1e-5
TOPOLOGY_LNS_HISTORY: List[Dict[str, Any]] = []


def pin_hpwl_subgradient(centers: np.ndarray) -> np.ndarray:
    gradient = np.zeros_like(centers, dtype=np.float64)
    for net in DB.nets:
        pins = centers[net.pin_macro_ids] + net.pin_offsets
        for axis in (0, 1):
            values = pins[:, axis]
            max_mask = values >= values.max() - 1e-9
            min_mask = values <= values.min() + 1e-9
            np.add.at(gradient[:, axis], net.pin_macro_ids[max_mask], 1.0)
            np.add.at(gradient[:, axis], net.pin_macro_ids[min_mask], -1.0)
    return gradient


def select_topology_pairs(
    base: np.ndarray,
    choose_x: np.ndarray,
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    limit: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    normalized_gap = np.maximum(
        np.abs(delta_x) / PAIR_W,
        np.abs(delta_y) / PAIR_H,
    ) - 1.0
    near_indices = np.flatnonzero(normalized_gap <= TOPOLOGY_LNS_NEAR_MARGIN)
    desired_motion = -pin_hpwl_subgradient(base)
    pressure = np.full(PAIR_I.size, -np.inf, dtype=np.float64)

    for k in near_indices:
        i, j = int(PAIR_I[k]), int(PAIR_J[k])
        if choose_x[k]:
            pressure[k] = (
                desired_motion[i, 0] - desired_motion[j, 0]
                if delta_x[k] <= 0
                else desired_motion[j, 0] - desired_motion[i, 0]
            )
        else:
            pressure[k] = (
                desired_motion[i, 1] - desired_motion[j, 1]
                if delta_y[k] <= 0
                else desired_motion[j, 1] - desired_motion[i, 1]
            )

    selected = near_indices[
        np.argsort(-pressure[near_indices])[: min(limit, near_indices.size)]
    ]
    critical = np.zeros(PAIR_I.size, dtype=bool)
    critical[selected] = True
    return critical, pressure, int(near_indices.size)


def solve_topology_lns_round(
    base: np.ndarray,
    round_id: int,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    base_metrics = full_metrics(base, 0.0, 0.0)
    delta_x = base[PAIR_I, 0] - base[PAIR_J, 0]
    delta_y = base[PAIR_I, 1] - base[PAIR_J, 1]
    choose_x = np.abs(delta_x) / PAIR_W >= np.abs(delta_y) / PAIR_H
    critical, pressure, near_count = select_topology_pairs(
        base,
        choose_x,
        delta_x,
        delta_y,
        TOPOLOGY_LNS_CRITICAL_PAIRS,
    )

    model = Model(name=f"topology_lns_round_{round_id}")
    model.parameters.threads = CPLEX_THREADS
    model.parameters.timelimit = TOPOLOGY_LNS_TIME_LIMIT_SEC
    model.parameters.mip.tolerances.mipgap = TOPOLOGY_LNS_MIP_GAP
    model.parameters.emphasis.mip = 1
    infinity = model.infinity

    x = [
        model.continuous_var(
            lb=float(WIDTHS[i] / 2.0),
            ub=float(DB.canvas_width - WIDTHS[i] / 2.0),
            name=f"x_{i}",
        )
        for i in range(DB.num_macros)
    ]
    y = [
        model.continuous_var(
            lb=float(HEIGHTS[i] / 2.0),
            ub=float(DB.canvas_height - HEIGHTS[i] / 2.0),
            name=f"y_{i}",
        )
        for i in range(DB.num_macros)
    ]
    xmax = model.continuous_var_list(len(DB.nets), lb=-infinity, name="xmax")
    xmin = model.continuous_var_list(len(DB.nets), lb=-infinity, name="xmin")
    ymax = model.continuous_var_list(len(DB.nets), lb=-infinity, name="ymax")
    ymin = model.continuous_var_list(len(DB.nets), lb=-infinity, name="ymin")
    dx = model.continuous_var_list(DB.num_macros, lb=0.0, name="dx")
    dy = model.continuous_var_list(DB.num_macros, lb=0.0, name="dy")

    linear_constraints = []
    for net_id, net in enumerate(DB.nets):
        for macro_id, offset in zip(net.pin_macro_ids, net.pin_offsets):
            i = int(macro_id)
            ox, oy = float(offset[0]), float(offset[1])
            linear_constraints.extend([
                xmax[net_id] >= x[i] + ox,
                xmin[net_id] <= x[i] + ox,
                ymax[net_id] >= y[i] + oy,
                ymin[net_id] <= y[i] + oy,
            ])
    for i in range(DB.num_macros):
        linear_constraints.extend([
            dx[i] >= x[i] - float(base[i, 0]),
            dx[i] >= float(base[i, 0]) - x[i],
            dy[i] >= y[i] - float(base[i, 1]),
            dy[i] >= float(base[i, 1]) - y[i],
        ])
    model.add_constraints(linear_constraints)
    del linear_constraints

    binary_groups = []
    big_m_x = float(DB.canvas_width + POLISH_CLEARANCE)
    big_m_y = float(DB.canvas_height + POLISH_CLEARANCE)
    for k in range(PAIR_I.size):
        i, j = int(PAIR_I[k]), int(PAIR_J[k])
        if critical[k]:
            left = model.binary_var(name=f"l_{i}_{j}")
            right = model.binary_var(name=f"r_{i}_{j}")
            below = model.binary_var(name=f"b_{i}_{j}")
            above = model.binary_var(name=f"a_{i}_{j}")
            binary_groups.append((k, left, right, below, above))
            model.add_constraint(left + right + below + above == 1)
            model.add_constraint(
                x[i] - x[j]
                <= -float(PAIR_W[k]) - POLISH_CLEARANCE + big_m_x * (1 - left)
            )
            model.add_constraint(
                x[j] - x[i]
                <= -float(PAIR_W[k]) - POLISH_CLEARANCE + big_m_x * (1 - right)
            )
            model.add_constraint(
                y[i] - y[j]
                <= -float(PAIR_H[k]) - POLISH_CLEARANCE + big_m_y * (1 - below)
            )
            model.add_constraint(
                y[j] - y[i]
                <= -float(PAIR_H[k]) - POLISH_CLEARANCE + big_m_y * (1 - above)
            )
        elif choose_x[k]:
            if delta_x[k] <= 0:
                model.add_constraint(
                    x[i] - x[j] <= -float(PAIR_W[k]) - POLISH_CLEARANCE
                )
            else:
                model.add_constraint(
                    x[j] - x[i] <= -float(PAIR_W[k]) - POLISH_CLEARANCE
                )
        elif delta_y[k] <= 0:
            model.add_constraint(
                y[i] - y[j] <= -float(PAIR_H[k]) - POLISH_CLEARANCE
            )
        else:
            model.add_constraint(
                y[j] - y[i] <= -float(PAIR_H[k]) - POLISH_CLEARANCE
            )

    hpwl = model.sum(
        xmax[e] - xmin[e] + ymax[e] - ymin[e] for e in range(len(DB.nets))
    )
    displacement = model.sum(dx) + model.sum(dy)
    model.minimize(hpwl + TOPOLOGY_LNS_DISPLACEMENT_WEIGHT * displacement)

    warm_start = model.new_solution()
    for i in range(DB.num_macros):
        warm_start.add_var_value(x[i], float(base[i, 0]))
        warm_start.add_var_value(y[i], float(base[i, 1]))
    for k, left, right, below, above in binary_groups:
        values = [0, 0, 0, 0]
        if choose_x[k]:
            values[0 if delta_x[k] <= 0 else 1] = 1
        else:
            values[2 if delta_y[k] <= 0 else 3] = 1
        for variable, value in zip((left, right, below, above), values):
            warm_start.add_var_value(variable, value)
    model.add_mip_start(warm_start)

    finite_pressure = pressure[critical]
    log_msg(
        f"LNS round={round_id}: base F(x)={f_value(base_metrics):.7f}, "
        f"near={near_count}, binary_pairs={int(critical.sum())}, "
        f"pressure=[{finite_pressure.min():.3f}, {finite_pressure.max():.3f}]"
    )
    solution = model.solve(log_output=False)
    details = model.solve_details
    status = str(details.status)
    solve_time = float(getattr(details, "time", 0.0) or 0.0)
    if solution is None:
        record = {
            "round": round_id,
            "status": status,
            "solve_time_sec": solve_time,
            "near_pairs": near_count,
            "binary_pairs": int(critical.sum()),
            "base_F_x": f_value(base_metrics),
            "accepted": False,
        }
        model.end()
        gc.collect()
        return None, record

    candidate = np.column_stack((
        [solution.get_value(variable) for variable in x],
        [solution.get_value(variable) for variable in y],
    ))
    model.end()
    gc.collect()

    # Remove only solver-scale boundary noise (typically 1e-11), then audit all pairs.
    candidate[:, 0] = np.clip(
        candidate[:, 0], WIDTHS / 2.0, DB.canvas_width - WIDTHS / 2.0
    )
    candidate[:, 1] = np.clip(
        candidate[:, 1], HEIGHTS / 2.0, DB.canvas_height - HEIGHTS / 2.0
    )
    candidate_metrics = full_metrics(candidate, 0.0, 0.0)
    legal = (
        candidate_metrics["overlap_pairs"] == 0
        and candidate_metrics["boundary_violating_macros"] == 0
    )
    improved = (
        f_value(candidate_metrics)
        < f_value(base_metrics) * (1.0 - TOPOLOGY_LNS_MIN_REL_IMPROVEMENT)
    )
    record = {
        "round": round_id,
        "status": status,
        "solve_time_sec": solve_time,
        "near_pairs": near_count,
        "binary_pairs": int(critical.sum()),
        "base_F_x": f_value(base_metrics),
        "candidate_F_x": f_value(candidate_metrics),
        "overlap_pairs": candidate_metrics["overlap_pairs"],
        "boundary_violating_macros": candidate_metrics["boundary_violating_macros"],
        "accepted": bool(legal and improved),
    }
    return candidate, record


PRE_LNS_METRICS = FINAL_METRICS.copy()
if RUN_TOPOLOGY_LNS:
    for round_id in range(1, TOPOLOGY_LNS_ROUNDS + 1):
        trial, record = solve_topology_lns_round(Z_QUALITY, round_id)
        TOPOLOGY_LNS_HISTORY.append(record)
        log_msg("LNS result: " + json.dumps(record))
        if trial is None or not record["accepted"]:
            log_msg(f"Stopping LNS after round {round_id}: no audited improvement")
            break
        Z_QUALITY = trial
        np.save(PLACEMENTS_DIR / f"z_topology_lns_round_{round_id}.npy", Z_QUALITY)

FINAL_METRICS = full_metrics(Z_QUALITY, 0.0, 0.0)
if FINAL_METRICS["overlap_pairs"] != 0:
    raise AssertionError("Final LNS full-pair audit found overlap")
if FINAL_METRICS["boundary_violating_macros"] != 0:
    raise AssertionError("Final LNS full-pair audit found a boundary violation")
log_msg(
    f"Final audited result: F(x)={f_value(FINAL_METRICS):.7f}, "
    f"overlap={FINAL_METRICS['overlap_pairs']}, "
    f"boundary={FINAL_METRICS['boundary_violating_macros']}"
)
'''
        ),
        markdown_cell(
            r'''
## 6. Export, biểu đồ và báo cáo kiểm toán

File `.pl` giữ nguyên tất cả dòng/node trong Bookshelf gốc và chỉ thay tọa độ macro.
Tọa độ tối ưu nội bộ là center-local; khi export được đổi đúng sang lower-left-global.
'''
        ),
        code_cell(
            r'''
# Cell 11 - Export a legal Bookshelf placement and an auditable report

def paper_centers_to_local_lower_left(centers_local: np.ndarray) -> np.ndarray:
    return centers_local - np.column_stack((WIDTHS, HEIGHTS)) / 2.0


def paper_centers_to_global_lower_left(centers_local: np.ndarray) -> np.ndarray:
    return paper_centers_to_local_lower_left(centers_local) + np.asarray(
        [DB.x_min, DB.y_min], dtype=np.float64
    )


def global_lower_left_to_paper_centers(global_lower_left: np.ndarray) -> np.ndarray:
    local_lower_left = global_lower_left - np.asarray([DB.x_min, DB.y_min])
    return local_lower_left + np.column_stack((WIDTHS, HEIGHTS)) / 2.0


def write_bookshelf_pl(centers_local: np.ndarray, destination: Path) -> None:
    global_lower_left = paper_centers_to_global_lower_left(centers_local)
    coordinates = {
        macro.name: global_lower_left[idx] for idx, macro in enumerate(DB.macros)
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
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


FINAL_PL_PATH = PLACEMENTS_DIR / f"{BENCHMARK}.quality_max_legal.pl"
GLOBAL_LOWER_LEFT = paper_centers_to_global_lower_left(Z_QUALITY)
roundtrip = global_lower_left_to_paper_centers(GLOBAL_LOWER_LEFT)
if not np.allclose(roundtrip, Z_QUALITY, atol=1e-9):
    raise AssertionError("Local/global coordinate round-trip failed")

np.save(PLACEMENTS_DIR / "z_quality_max_centers.npy", Z_QUALITY)
np.save(PLACEMENTS_DIR / "z_quality_max_lower_left_global.npy", GLOBAL_LOWER_LEFT)
write_bookshelf_pl(Z_QUALITY, FINAL_PL_PATH)

with (METRICS_DIR / "candidate_pool.json").open("w", encoding="utf-8") as handle:
    json.dump([
        {
            "name": item["name"],
            "source": item["source"],
            "score": item["score"],
            "metrics": item["metrics"],
        }
        for item in CANDIDATES
    ], handle, indent=2)
with (METRICS_DIR / "polish_history.json").open("w", encoding="utf-8") as handle:
    json.dump(POLISH_HISTORY, handle, indent=2)
with (METRICS_DIR / "topology_lns_history.json").open("w", encoding="utf-8") as handle:
    json.dump(TOPOLOGY_LNS_HISTORY, handle, indent=2)

EXPORTED_CENTERS = global_lower_left_to_paper_centers(
    np.round(GLOBAL_LOWER_LEFT, decimals=6)
)
EXPORTED_METRICS = full_metrics(EXPORTED_CENTERS, 0.0, 0.0)
if EXPORTED_METRICS["overlap_pairs"] != 0:
    raise AssertionError("Rounded .pl coordinates create an overlap")
if EXPORTED_METRICS["boundary_violating_macros"] != 0:
    raise AssertionError("Rounded .pl coordinates create a boundary violation")

summary = {
    "status": "legal_quality_max",
    "feasible": True,
    "benchmark": BENCHMARK,
    "F_definition": "pin_HPWL / 1e5",
    "F_scale": F_SCALE,
    "full_pairs_checked": int(PAIR_I.size),
    "raw_grid_F_x": f_value(full_metrics(Z0_PAPER_CENTERS, 0.0, 0.0)),
    "tetris_F_x": f_value(TETRIS_BEST_METRICS),
    "pre_lns_F_x": f_value(PRE_LNS_METRICS),
    "final_F_x": f_value(FINAL_METRICS),
    "exported_pl_F_x": f_value(EXPORTED_METRICS),
    "raw_grid_metrics": full_metrics(Z0_PAPER_CENTERS, 0.0, 0.0),
    "tetris_metrics": TETRIS_BEST_METRICS,
    "pre_lns_metrics": PRE_LNS_METRICS,
    "final_metrics": FINAL_METRICS,
    "exported_pl_metrics": EXPORTED_METRICS,
    "selected_anchor_name": LEGAL_CANDIDATES[0]["name"],
    "accepted_finetune_candidates": len(FINE_TUNE_CANDIDATES),
    "fine_tune_final_acceptance": bool(
        LEGAL_CANDIDATES[0]["name"].startswith("drl_finetune_")
        and f_value(TETRIS_BEST_METRICS) < 16.2531734
        and FINAL_METRICS["overlap_pairs"] == 0
        and FINAL_METRICS["boundary_violating_macros"] == 0
        and f_value(FINAL_METRICS) < 14.4531515
    ),
    "comparison": [
        {
            "candidate": "DRL old",
            "raw_F_x": 20.6480950,
            "raw_overlap_pairs": 1086,
            "raw_boundary_violations": 5,
            "legalized_F_x": 16.2531734,
            "lns_F_x": 14.4531515,
        },
        *([
            {
                "candidate": "DRL fine-tune best",
                "raw_F_x": f_value(FINE_TUNE_CANDIDATES[0]["metrics"]),
                "raw_overlap_pairs": FINE_TUNE_CANDIDATES[0]["metrics"]["overlap_pairs"],
                "raw_boundary_violations": FINE_TUNE_CANDIDATES[0]["metrics"]["boundary_violating_macros"],
                "legalized_F_x": f_value(FINE_TUNE_CANDIDATES[0]["metrics"]),
                "lns_F_x": (
                    f_value(FINAL_METRICS)
                    if LEGAL_CANDIDATES[0]["name"].startswith("drl_finetune_")
                    else None
                ),
            }
        ] if FINE_TUNE_CANDIDATES else []),
    ],
    "output_pl": str(FINAL_PL_PATH),
    "output_centers": str(PLACEMENTS_DIR / "z_quality_max_centers.npy"),
    "long_training_enabled": RUN_LONG_TRAINING,
    "long_training_loops": LONG_TRAIN_LOOPS,
    "long_training_seeds": list(LONG_TRAIN_SEEDS),
}
with (METRICS_DIR / "summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print(json.dumps(summary, indent=2))
'''
        ),
        code_cell(
            r'''
# Cell 12 - Visual comparison

def draw_placement(axis, centers: np.ndarray, title: str) -> None:
    lower_left = paper_centers_to_local_lower_left(centers)
    for idx, macro in enumerate(DB.macros):
        axis.add_patch(plt.Rectangle(
            lower_left[idx], macro.width, macro.height,
            fill=False, linewidth=0.30, alpha=0.65,
        ))
    axis.set_xlim(0, DB.canvas_width)
    axis.set_ylim(0, DB.canvas_height)
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_xlabel("local x")
    axis.set_ylabel("local y")


figure, axes = plt.subplots(1, 3, figsize=(18, 6))
draw_placement(
    axes[0], Z0_PAPER_CENTERS,
    f"Grid-32 input\nF(x)={pin_hpwl(Z0_PAPER_CENTERS) / F_SCALE:.6f}",
)
draw_placement(
    axes[1], TETRIS_BEST,
    f"Hard legal Tetris\nF(x)={f_value(TETRIS_BEST_METRICS):.6f}",
)
draw_placement(
    axes[2], Z_QUALITY,
    f"Quality-max legal\nF(x)={f_value(FINAL_METRICS):.6f}",
)
figure.tight_layout()
plot_path = PLOTS_DIR / "quality_max_comparison.png"
figure.savefig(plot_path, dpi=180, bbox_inches="tight")
plt.show()

print(f"Notebook output directory: {EXP_DIR}")
print(f"Legal Bookshelf placement: {FINAL_PL_PATH}")
print(f"Comparison plot: {plot_path}")
'''
        ),
    ]
)

notebook = {
    "cells": cells,
    "metadata": base["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5,
}
DESTINATION.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DESTINATION}")

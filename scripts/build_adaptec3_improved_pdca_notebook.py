from __future__ import annotations

import copy
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cplex_resolver" / "adaptec3_drl_pdca_cplex.ipynb"
DESTINATION = ROOT / "cplex_resolver" / "adaptec3_drl_pdca_cplex_improved.ipynb"


def source_lines(source: str) -> list[str]:
    return textwrap.dedent(source).lstrip("\n").splitlines(keepends=True)


def code_cell(source: str, template: dict | None = None) -> dict:
    cell = copy.deepcopy(template) if template is not None else {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [],
    }
    cell["cell_type"] = "code"
    cell["execution_count"] = None
    cell["outputs"] = []
    cell["source"] = source_lines(source)
    return cell


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(source),
    }


with SOURCE.open("r", encoding="utf-8") as handle:
    notebook = json.load(handle)

original = notebook["cells"]
if len(original) != 17:
    raise RuntimeError(f"Expected 17 source cells, found {len(original)}")


intro = markdown_cell(
    r'''
    # Adaptec3 — baseline-first active-set PDCA with safeguarded acceleration

    Notebook này được tạo từ `adaptec3_drl_pdca_cplex.ipynb` nhưng giữ nguyên notebook
    gốc. Các thay đổi chính:

    - boundary là hard bounds thay vì soft penalty;
    - tự động nạp nghiệm baseline `gamma=5000, rho=0.01, 300 iterations` làm incumbent
      nếu artifact còn tồn tại; nếu không có thì fallback về DRL checkpoint;
    - giữ cố định `GAMMA_OVERLAP=5000` và `rho=0.01`, không continuation/adaptive rho;
    - một QP active-set mỗi vòng; các cặp vi phạm mới được carry sang vòng kế tiếp;
    - acceleration có cửa sổ non-monotone 3 bước và chặn mức tăng tương đối 0.15%;
    - luôn lưu/trả nghiệm tốt nhất giữa incumbent ban đầu và mọi candidate;
    - stopping criterion theo relative step và relative objective;
    - optional topology-changing LNS-MIQP cho các cặp critical;
    - xuất Bookshelf `.pl` có cộng lại origin của `.scl`.

    `F_ref`, `HPWL` và các penalty trong notebook dùng đơn vị thô. Giá trị báo cáo trong bài
    có thể chia cho `1e5`.
    '''
)


configuration = code_cell(
    r'''
    # Cell 1 - Imports and improved experiment constants

    from __future__ import annotations

    import os
    import sys
    import time
    import json
    import csv
    import math
    import re
    import gc
    import warnings
    from math import sqrt
    from pathlib import Path
    from dataclasses import dataclass, field
    from typing import List, Dict, Tuple, Any, Optional, Set

    import numpy as np

    # Patch NumPy 2.x compatibility for DOcplex
    if not hasattr(np, "float_"):
        np.float_ = np.float64

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # ======================================================================
    # REFERENCE OBJECTIVE (used to compare every stage and every candidate)
    # ======================================================================
    GAMMA_BOUNDARY: float = 100.0
    EVAL_GAMMA_OVERLAP: float = 5000.0
    GAMMA_OVERLAP: float = EVAL_GAMMA_OVERLAP  # compatibility alias

    # Baseline-matched single stage. F_ref and the stage objective are identical.
    STAGE_SCHEDULE: List[Dict[str, Any]] = [
        {"name": "baseline_recovery", "gamma_overlap": 5000.0, "rho_init": 0.01, "max_iter": 300},
    ]
    MAX_DCA_ITER: int = sum(int(stage["max_iter"]) for stage in STAGE_SCHEDULE)
    RHO: float = float(STAGE_SCHEDULE[0]["rho_init"])  # compatibility/audit alias

    # Fixed proximal control: earlier experiments showed rho=0.01 is the safer basin.
    RHO_MIN: float = 0.01
    RHO_MAX: float = 0.01
    RHO_INCREASE: float = 2.0
    RHO_DECREASE: float = 0.8
    RHO_MAX_RETRIES: int = 0
    RHO_GOOD_REL_DECREASE: float = 1e-3

    # Active-set + controlled non-monotone safeguards
    ACTIVE_SET_MARGIN: float = 1.0
    ACTIVE_SET_MAX_REFINEMENTS: int = 1
    ACTIVE_VIOLATION_TOL: float = 1e-9
    USE_ACCELERATION: bool = True
    LINE_SEARCH_FACTORS: Tuple[float, ...] = (0.5, 0.25, 0.125, 0.0625)
    MONOTONE_RTOL: float = 1e-10
    MONOTONE_ATOL: float = 1e-5
    NONMONOTONE_WINDOW: int = 3
    MAX_REL_OBJECTIVE_INCREASE: float = 1.5e-3

    # Scale-aware stopping
    REL_STEP_TOL: float = 1e-5
    EARLY_STOP_PATIENCE: int = 30
    EARLY_STOP_RTOL: float = 1e-5

    # CPLEX. parallel=1 below requests deterministic parallel optimization.
    CPLEX_THREADS: int = 4
    CPLEX_WORKMEM_MB: int = 8192

    # Optional topology-changing phase after PDCA
    # Keep disabled for the first long PDCA run. Enable only after inspecting
    # the saved PDCA placement; every LNS candidate is still full-pair audited.
    RUN_TOPOLOGY_LNS: bool = False
    TOPOLOGY_LNS_ROUNDS: int = 3
    TOPOLOGY_LNS_CRITICAL_PAIRS: int = 64
    TOPOLOGY_LNS_NEAR_MARGIN: float = 0.05
    TOPOLOGY_LNS_TIME_LIMIT_SEC: int = 180
    TOPOLOGY_LNS_MIP_GAP: float = 1e-3
    TOPOLOGY_LNS_TRUST_RADIUS_FRACTION: float = 0.05
    TOPOLOGY_LNS_MOVE_WEIGHT: float = 20.0
    TOPOLOGY_LNS_MIN_REL_IMPROVEMENT: float = 1e-6

    BENCHMARK: str = "adaptec3"
    PROJECT_ROOT: Path = Path(r"D:\HaUI\scholarship\research\EfficientPlace-HaUI-main\EfficientPlace-HaUI-main\cplex_resolver")
    if not (PROJECT_ROOT / "ISPD2005" / BENCHMARK).is_dir():
        for candidate in [Path(r"d:\HaUI\scholarship\research\cplex_resolver"), Path.cwd()]:
            if (candidate / "ISPD2005" / BENCHMARK).is_dir():
                PROJECT_ROOT = candidate
                break
    GRID_SIZE: int = 32

    print("Cell 1: Improved configuration loaded.")
    print(f"  Benchmark              : {BENCHMARK}")
    print(f"  Reference gamma overlap: {EVAL_GAMMA_OVERLAP}")
    print(f"  Stage schedule         : {STAGE_SCHEDULE}")
    print(f"  Fixed rho              : {RHO}")
    print(f"  Active margin/refines  : {ACTIVE_SET_MARGIN} / {ACTIVE_SET_MAX_REFINEMENTS}")
    print(f"  Controlled acceleration: {USE_ACCELERATION}")
    print(f"  Non-monotone window/cap: {NONMONOTONE_WINDOW} / {MAX_REL_OBJECTIVE_INCREASE:.4%}")
    print(f"  Optional topology LNS  : {RUN_TOPOLOGY_LNS}")
    ''',
    original[0],
)


logging_cell = code_cell(
    r'''
    # Cell 2 - Experiment directories and structured logging

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = (
        f"baseline_first_gov_{int(EVAL_GAMMA_OVERLAP)}_rho_1e-02_"
        f"nonmono_q{NONMONOTONE_WINDOW}"
    )
    exp_dir = PROJECT_ROOT / "experiments" / BENCHMARK / exp_name / f"run_{timestamp}"

    config_dir = exp_dir / "config"
    logs_dir = exp_dir / "logs"
    metrics_dir = exp_dir / "metrics"
    placements_dir = exp_dir / "placements"
    plots_dir = exp_dir / "plots"
    cplex_models_dir = exp_dir / "cplex_models"

    for d in [config_dir, logs_dir, metrics_dir, placements_dir, plots_dir, cplex_models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_file_path = logs_dir / "run.log"

    def log_msg(msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{ts}] {msg}"
        print(formatted)
        with log_file_path.open("a", encoding="utf-8") as f:
            f.write(formatted + "\n")

    exp_config = {
        "benchmark": BENCHMARK,
        "gamma_boundary_eval": GAMMA_BOUNDARY,
        "gamma_overlap_eval": EVAL_GAMMA_OVERLAP,
        "stage_schedule": STAGE_SCHEDULE,
        "rho_min": RHO_MIN,
        "rho_max": RHO_MAX,
        "rho_increase": RHO_INCREASE,
        "rho_decrease": RHO_DECREASE,
        "rho_max_retries": RHO_MAX_RETRIES,
        "active_set_margin": ACTIVE_SET_MARGIN,
        "active_set_max_refinements": ACTIVE_SET_MAX_REFINEMENTS,
        "use_controlled_nonmonotone_acceleration": USE_ACCELERATION,
        "nonmonotone_window": NONMONOTONE_WINDOW,
        "max_relative_objective_increase": MAX_REL_OBJECTIVE_INCREASE,
        "line_search_factors": list(LINE_SEARCH_FACTORS),
        "relative_step_tolerance": REL_STEP_TOL,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_rtol": EARLY_STOP_RTOL,
        "cplex_threads": CPLEX_THREADS,
        "cplex_workmem_mb": CPLEX_WORKMEM_MB,
        "run_topology_lns": RUN_TOPOLOGY_LNS,
        "topology_lns_rounds": TOPOLOGY_LNS_ROUNDS,
        "topology_lns_critical_pairs": TOPOLOGY_LNS_CRITICAL_PAIRS,
        "topology_lns_time_limit_sec": TOPOLOGY_LNS_TIME_LIMIT_SEC,
        "project_root": str(PROJECT_ROOT),
        "experiment_root": str(exp_dir),
        "timestamp": timestamp,
    }
    with (config_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    log_msg("Improved experiment output directories initialized.")
    log_msg(f"Experiment root: {exp_dir}")
    ''',
    original[1],
)


# Keep installation validation and dataset parser, but clear their old outputs.
validation_cell = code_cell("".join(original[2]["source"]), original[2])
database_cell = code_cell("".join(original[3]["source"]), original[3])
load_cell = code_cell("".join(original[4]["source"]), original[4])


metrics_source = "".join(original[5]["source"])
metrics_source = metrics_source.replace(
    'log_msg("DRL CHECKPOINT (z0) FULL METRICS (all 156,520 pairs):")',
    'log_msg(f"DRL CHECKPOINT (z0) FULL METRICS (all {METRICS_BEFORE[\'eq3_total_pairs\']:,} pairs):")',
)
metrics_cell = code_cell(metrics_source, original[5])


active_set_cell = code_cell(
    r'''
    # Cell 7 - Active-set selection with carry-over and refinement support

    @dataclass
    class ActiveSetData:
        active_idx: np.ndarray
        act_i: np.ndarray
        act_j: np.ndarray
        act_wij: np.ndarray
        act_hij: np.ndarray
        num_active: int
        total_pairs: int


    def active_set_from_indices(indices: np.ndarray, db: PlacementDB) -> ActiveSetData:
        pair_i, pair_j = np.triu_indices(db.num_macros, k=1)
        widths, heights = macro_geometry(db)
        indices = np.unique(np.asarray(indices, dtype=np.int64))
        wij = (widths[pair_i] + widths[pair_j]) / 2.0
        hij = (heights[pair_i] + heights[pair_j]) / 2.0
        return ActiveSetData(
            active_idx=indices,
            act_i=pair_i[indices],
            act_j=pair_j[indices],
            act_wij=wij[indices],
            act_hij=hij[indices],
            num_active=int(indices.size),
            total_pairs=int(pair_i.size),
        )


    def select_active_set(
        z: np.ndarray,
        db: PlacementDB,
        margin: float = ACTIVE_SET_MARGIN,
        extra_indices: Optional[np.ndarray] = None,
    ) -> ActiveSetData:
        pair = pair_terms_eq3_eq8(z, db)
        indices = np.flatnonzero(pair["Nij"] <= margin)
        if extra_indices is not None and len(extra_indices):
            indices = np.union1d(indices, np.asarray(extra_indices, dtype=np.int64))
        return active_set_from_indices(indices, db)


    def violating_pair_indices(
        z: np.ndarray,
        db: PlacementDB,
        tolerance: float = ACTIVE_VIOLATION_TOL,
    ) -> np.ndarray:
        pair = pair_terms_eq3_eq8(z, db)
        return np.flatnonzero(pair["Nij"] < -tolerance)


    ACT0 = select_active_set(Z0_PAPER_CENTERS, DB, ACTIVE_SET_MARGIN)
    log_msg(
        f"Active-Set at z0 (margin={ACTIVE_SET_MARGIN}): "
        f"{ACT0.num_active} / {ACT0.total_pairs} pairs "
        f"({ACT0.num_active / ACT0.total_pairs * 100:.2f}%)"
    )
    ''',
    original[6],
)


audit_cell = code_cell("".join(original[7]["source"]), original[7])
subgradient_source = "".join(original[8]["source"]).replace("\\partial", "\\\\partial")
subgradient_cell = code_cell(subgradient_source, original[8])


model_cell = code_cell(
    r'''
    # Cell 10 - Hard-boundary convex QP builder

    def build_activeset_subproblem_model(
        db: PlacementDB,
        active_set: ActiveSetData,
        yk: np.ndarray,
        gamma_overlap: float,
        rho: float,
        threads: int = CPLEX_THREADS,
        workmem_mb: int = CPLEX_WORKMEM_MB,
    ) -> Tuple[Model, list, list]:
        """Build the convex PDCA subproblem with hard boundary bounds."""
        N = db.num_macros
        num_nets = len(db.nets)
        num_act = active_set.num_active
        widths, heights = macro_geometry(db)
        W, H = db.canvas_width, db.canvas_height

        mdl = Model(name="improved_activeset_eq17")
        mdl.parameters.threads = threads
        mdl.parameters.parallel = 1
        mdl.parameters.solutiontype = 2
        mdl.parameters.workmem = workmem_mb
        mdl.parameters.barrier.display = 0
        inf = mdl.infinity

        # Hard boundary: acceleration is projected to the same box later.
        x = [
            mdl.continuous_var(
                lb=float(widths[i] / 2.0),
                ub=float(W - widths[i] / 2.0),
                name=f"x_{i}",
            )
            for i in range(N)
        ]
        y = [
            mdl.continuous_var(
                lb=float(heights[i] / 2.0),
                ub=float(H - heights[i] / 2.0),
                name=f"y_{i}",
            )
            for i in range(N)
        ]

        xmax = mdl.continuous_var_list(num_nets, lb=-inf, ub=inf, name="xmax")
        xmin = mdl.continuous_var_list(num_nets, lb=-inf, ub=inf, name="xmin")
        ymax = mdl.continuous_var_list(num_nets, lb=-inf, ub=inf, name="ymax")
        ymin = mdl.continuous_var_list(num_nets, lb=-inf, ub=inf, name="ymin")
        q = mdl.continuous_var_list(num_act, lb=0, ub=inf, name="q")

        constraints = []
        for e, net in enumerate(db.nets):
            for m_id in net.macro_ids:
                i = int(m_id)
                constraints.extend([
                    xmax[e] >= x[i],
                    xmin[e] <= x[i],
                    ymax[e] >= y[i],
                    ymin[e] <= y[i],
                ])

        inv_w = 1.0 / active_set.act_wij
        inv_h = 1.0 / active_set.act_hij
        for k in range(num_act):
            i = int(active_set.act_i[k])
            j = int(active_set.act_j[k])
            iw, ih = float(inv_w[k]), float(inv_h[k])
            qk = q[k]
            constraints.extend([
                qk - iw * x[i] + iw * x[j] >= -1.0,
                qk + iw * x[i] - iw * x[j] >= -1.0,
                qk - ih * y[i] + ih * y[j] >= -1.0,
                qk + ih * y[i] - ih * y[j] >= -1.0,
            ])
        mdl.add_constraints(constraints)

        hpwl_term = mdl.sum(
            xmax[e] - xmin[e] + ymax[e] - ymin[e]
            for e in range(num_nets)
        )
        overlap_term = gamma_overlap * mdl.sum(q)
        prox_term = 0.5 * rho * mdl.sum(
            x[i] ** 2 + y[i] ** 2 for i in range(N)
        )
        lin_term = -mdl.scal_prod(x, yk[:, 0].tolist()) - mdl.scal_prod(
            y, yk[:, 1].tolist()
        )
        mdl.minimize(hpwl_term + overlap_term + prox_term + lin_term)
        return mdl, x, y


    log_msg("Hard-boundary QP builder ready.")
    ''',
    original[9],
)


solve_cell = code_cell(
    r'''
    # Cell 11 - CPLEX solve with explicit model cleanup

    def solve_convex_subproblem_cplex_activeset(
        db: PlacementDB,
        active_set: ActiveSetData,
        yk: np.ndarray,
        gamma_overlap: float,
        rho: float,
        global_iter: int,
        refinement: int,
        cplex_models_folder: Path,
    ) -> Tuple[Optional[np.ndarray], float, float, str]:
        N = db.num_macros
        build_start = time.time()
        mdl, x_vars, y_vars = build_activeset_subproblem_model(
            db=db,
            active_set=active_set,
            yk=yk,
            gamma_overlap=gamma_overlap,
            rho=rho,
        )
        build_time = time.time() - build_start

        if global_iter == 0:
            lp_path = cplex_models_folder / f"iter_000_refine_{refinement}.lp"
            try:
                mdl.export_as_lp(str(lp_path))
            except Exception as exc:
                log_msg(f"Warning exporting audit LP: {exc}")

        solve_start = time.time()
        solution = mdl.solve(log_output=False)
        solve_time = time.time() - solve_start
        status_name = str(mdl.solve_details.status) if mdl.solve_details else "unknown"

        if solution is None:
            failed_lp = cplex_models_folder / (
                f"failed_iter_{global_iter:03d}_refine_{refinement}.lp"
            )
            try:
                mdl.export_as_lp(str(failed_lp))
            except Exception:
                pass
            mdl.end()
            gc.collect()
            return None, float("nan"), solve_time + build_time, status_name

        z_next = np.empty((N, 2), dtype=np.float64)
        z_next[:, 0] = [solution.get_value(x_vars[i]) for i in range(N)]
        z_next[:, 1] = [solution.get_value(y_vars[i]) for i in range(N)]
        surrogate_value = float(solution.objective_value)
        mdl.end()
        gc.collect()

        if not np.isfinite(z_next).all():
            return None, float("nan"), solve_time + build_time, "non_finite_solution"
        return z_next, surrogate_value, solve_time + build_time, status_name


    log_msg("Refined active-set CPLEX solver ready.")
    ''',
    original[10],
)


algorithm_cell = code_cell(
    r'''
    # Cell 12 - Baseline-first PDCA with controlled non-monotone acceleration

    @dataclass
    class ControlledAccelerator:
        enabled: bool = True
        t: float = 1.0

        def candidate(
            self, current: np.ndarray, qp_next: np.ndarray
        ) -> Tuple[np.ndarray, float, float]:
            t_next = (1.0 + sqrt(1.0 + 4.0 * self.t * self.t)) / 2.0
            beta = (self.t - 1.0) / t_next
            return qp_next + beta * (qp_next - current), t_next, beta

        def accept(self, t_next: float) -> None:
            self.t = float(t_next)

        def restart(self) -> None:
            self.t = 1.0


    @dataclass
    class DCAResult:
        z_star: np.ndarray
        z_last: np.ndarray
        history: List[Dict[str, Any]]
        final_status: str
        total_time_sec: float
        converged: bool
        iterations_run: int
        best_reference_objective: float


    def project_to_hard_boundary(z: np.ndarray, db: PlacementDB) -> np.ndarray:
        widths, heights = macro_geometry(db)
        projected = np.asarray(z, dtype=np.float64).copy()
        projected[:, 0] = np.clip(
            projected[:, 0], widths / 2.0, db.canvas_width - widths / 2.0
        )
        projected[:, 1] = np.clip(
            projected[:, 1], heights / 2.0, db.canvas_height - heights / 2.0
        )
        return projected


    def objective_metrics(z: np.ndarray, db: PlacementDB, gamma_overlap: float) -> Dict[str, Any]:
        return paper_metrics_eq2_to_eq8(
            z,
            db,
            gamma_boundary=GAMMA_BOUNDARY,
            gamma_overlap=gamma_overlap,
        )


    def descent_tolerance(value: float) -> float:
        return max(MONOTONE_ATOL, MONOTONE_RTOL * abs(value))


    def solve_with_active_refinement(
        base: np.ndarray,
        db: PlacementDB,
        carry_indices: np.ndarray,
        gamma_overlap: float,
        rho: float,
        global_iter: int,
        cplex_models_folder: Path,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        active_set = select_active_set(
            base, db, margin=ACTIVE_SET_MARGIN, extra_indices=carry_indices
        )
        total_new_pairs = 0
        total_cplex_time = 0.0
        surrogate = float("nan")
        status = "not_solved"
        trial = None

        for refinement in range(ACTIVE_SET_MAX_REFINEMENTS):
            yk = compute_subgradient_H_activeset(
                base,
                db,
                active_set=active_set,
                gamma_overlap=gamma_overlap,
                rho=rho,
            )
            trial, surrogate, elapsed, status = solve_convex_subproblem_cplex_activeset(
                db=db,
                active_set=active_set,
                yk=yk,
                gamma_overlap=gamma_overlap,
                rho=rho,
                global_iter=global_iter,
                refinement=refinement,
                cplex_models_folder=cplex_models_folder,
            )
            total_cplex_time += elapsed
            if trial is None:
                return None, {
                    "status": status,
                    "active_pairs": active_set.num_active,
                    "refinements": refinement + 1,
                    "new_pairs_added": total_new_pairs,
                    "unresolved_new_pairs": 0,
                    "cplex_time": total_cplex_time,
                    "surrogate": surrogate,
                }

            trial = project_to_hard_boundary(trial, db)
            violating = violating_pair_indices(trial, db)
            new_indices = np.setdiff1d(
                violating, active_set.active_idx, assume_unique=False
            )
            if new_indices.size == 0:
                break
            total_new_pairs += int(new_indices.size)
            if refinement + 1 >= ACTIVE_SET_MAX_REFINEMENTS:
                # With the one-QP policy, audit now and carry these pairs into
                # the next outer iteration instead of rebuilding immediately.
                break
            active_set = active_set_from_indices(
                np.union1d(active_set.active_idx, new_indices), db
            )

        unresolved = 0
        if trial is not None:
            unresolved = int(
                np.setdiff1d(
                    violating_pair_indices(trial, db),
                    active_set.active_idx,
                    assume_unique=False,
                ).size
            )
        return trial, {
            "status": status,
            "active_pairs": active_set.num_active,
            "refinements": refinement + 1,
            "new_pairs_added": total_new_pairs,
            "unresolved_new_pairs": unresolved,
            "cplex_time": total_cplex_time,
            "surrogate": surrogate,
        }


    def run_baseline_first_pdca(
        z0: np.ndarray,
        db: PlacementDB,
        stage_schedule: List[Dict[str, Any]],
        cplex_models_folder: Path,
    ) -> DCAResult:
        start_time = time.time()
        zk = project_to_hard_boundary(z0, db)
        history: List[Dict[str, Any]] = []
        accelerator = ControlledAccelerator(enabled=USE_ACCELERATION)
        carry_indices = violating_pair_indices(zk, db)

        reference_metrics = objective_metrics(zk, db, EVAL_GAMMA_OVERLAP)
        best_reference_F = float(reference_metrics["eq6_penalized_objective"])
        best_reference_z = zk.copy()
        global_iter = 0
        completed_all_stages = True
        final_status = "completed_schedule"

        log_msg("=" * 96)
        log_msg(
            f"START BASELINE-FIRST PDCA | total planned iterations={MAX_DCA_ITER} | "
            f"reference gamma={EVAL_GAMMA_OVERLAP}"
        )
        log_msg("=" * 96)

        for stage_id, stage in enumerate(stage_schedule):
            stage_name = str(stage["name"])
            gamma_overlap = float(stage["gamma_overlap"])
            rho = float(np.clip(stage["rho_init"], RHO_MIN, RHO_MAX))
            max_iter = int(stage["max_iter"])
            accelerator.restart()

            current_metrics = objective_metrics(zk, db, gamma_overlap)
            F_curr = float(current_metrics["eq6_penalized_objective"])
            best_stage_F = F_curr
            no_improve_count = 0
            recent_stage_F: List[float] = [F_curr]
            log_msg(
                f"STAGE {stage_id + 1}/{len(stage_schedule)} {stage_name}: "
                f"gamma={gamma_overlap}, rho={rho}, max_iter={max_iter}, "
                f"F_stage={F_curr:.6e}"
            )

            for stage_iter in range(max_iter):
                iter_start = time.time()
                accepted = False
                selected_z = None
                selected_metrics = None
                selected_kind = "none"
                line_search_alpha = 0.0
                solve_info: Dict[str, Any] = {}
                rho_retries = 0
                acceleration_beta = 0.0
                acceptance_ceiling = F_curr
                accepted_nonmonotone = False

                while rho_retries <= RHO_MAX_RETRIES and not accepted:
                    qp_next, solve_info = solve_with_active_refinement(
                        base=zk,
                        db=db,
                        carry_indices=carry_indices,
                        gamma_overlap=gamma_overlap,
                        rho=rho,
                        global_iter=global_iter,
                        cplex_models_folder=cplex_models_folder,
                    )
                    if qp_next is None:
                        final_status = f"cplex_failed_{solve_info.get('status', 'unknown')}"
                        completed_all_stages = False
                        break

                    qp_metrics = objective_metrics(qp_next, db, gamma_overlap)
                    F_qp = float(qp_metrics["eq6_penalized_objective"])
                    tolerance = descent_tolerance(F_curr)
                    candidate_options = [("qp", qp_next, qp_metrics, F_qp)]
                    t_next: Optional[float] = None

                    # Advance momentum after the bootstrap QP, then compare QP
                    # and accelerated candidates under one bounded envelope.
                    if accelerator.enabled:
                        accelerated, t_next, acceleration_beta = accelerator.candidate(
                            zk, qp_next
                        )
                        accelerated = project_to_hard_boundary(accelerated, db)
                        accelerated_metrics = objective_metrics(
                            accelerated, db, gamma_overlap
                        )
                        F_acc = float(accelerated_metrics["eq6_penalized_objective"])
                        candidate_options.append(
                            ("accelerated", accelerated, accelerated_metrics, F_acc)
                        )

                    window_ceiling = max(recent_stage_F[-NONMONOTONE_WINDOW:])
                    relative_ceiling = F_curr * (1.0 + MAX_REL_OBJECTIVE_INCREASE)
                    acceptance_ceiling = min(window_ceiling, relative_ceiling)
                    admissible = [
                        item for item in candidate_options
                        if item[3] <= acceptance_ceiling + tolerance
                    ]
                    if admissible:
                        selected_kind, selected_z, selected_metrics, F_selected = min(
                            admissible, key=lambda item: item[3]
                        )
                        if selected_kind == "qp" and abs(acceleration_beta) <= 1e-15:
                            selected_kind = "qp_bootstrap"
                        accepted_nonmonotone = F_selected > F_curr + tolerance
                        if accepted_nonmonotone:
                            selected_kind += "_nonmonotone"
                        line_search_alpha = 1.0
                        if accelerator.enabled and t_next is not None:
                            accelerator.accept(t_next)
                        accepted = True
                        break

                    # Safeguard for an incomplete active model: restart and seek
                    # a strictly descending point along the raw QP direction.
                    accelerator.restart()
                    direction = qp_next - zk
                    for alpha in LINE_SEARCH_FACTORS:
                        line_trial = project_to_hard_boundary(
                            zk + float(alpha) * direction, db
                        )
                        line_metrics = objective_metrics(
                            line_trial, db, gamma_overlap
                        )
                        if (
                            float(line_metrics["eq6_penalized_objective"])
                            <= F_curr - tolerance
                        ):
                            selected_z = line_trial
                            selected_metrics = line_metrics
                            selected_kind = "backtracking"
                            line_search_alpha = float(alpha)
                            accepted = True
                            break

                    if accepted:
                        break
                    rho_retries += 1

                if not completed_all_stages:
                    break
                if not accepted or selected_z is None or selected_metrics is None:
                    final_status = f"stalled_no_admissible_step_stage_{stage_name}"
                    completed_all_stages = False
                    log_msg(
                        f"Stage {stage_name} stalled at local iter {stage_iter + 1}; "
                        f"no admissible step after {rho_retries} fixed-rho attempt(s)."
                    )
                    break

                F_next = float(selected_metrics["eq6_penalized_objective"])
                reference_next = objective_metrics(
                    selected_z, db, EVAL_GAMMA_OVERLAP
                )
                F_reference = float(reference_next["eq6_penalized_objective"])
                step_norm = float(np.linalg.norm(selected_z - zk))
                scale = sqrt(db.num_macros) * math.hypot(
                    db.canvas_width, db.canvas_height
                )
                relative_step = step_norm / max(scale, 1.0)
                relative_decrease = (F_curr - F_next) / max(abs(F_curr), 1.0)

                carry_indices = violating_pair_indices(selected_z, db)
                if F_reference < best_reference_F - descent_tolerance(best_reference_F):
                    best_reference_F = F_reference
                    best_reference_z = selected_z.copy()

                row = {
                    "global_iteration": global_iter + 1,
                    "stage_id": stage_id + 1,
                    "stage_name": stage_name,
                    "stage_iteration": stage_iter + 1,
                    "gamma_overlap_stage": gamma_overlap,
                    "rho": rho,
                    "rho_retries": rho_retries,
                    "candidate_kind": selected_kind,
                    "acceleration_beta": acceleration_beta,
                    "acceptance_ceiling": acceptance_ceiling,
                    "accepted_nonmonotone": accepted_nonmonotone,
                    "line_search_alpha": line_search_alpha,
                    "F_stage_before": F_curr,
                    "F_stage_after": F_next,
                    "F_ref_after": F_reference,
                    "HPWL_after": reference_next["eq5_hpwl"],
                    "boundary_penalty_sum": reference_next["eq7_boundary_penalty_sum"],
                    "overlap_hat_sum": reference_next["eq8_overlap_hat_sum"],
                    "overlap_violating_pairs": reference_next["eq3_nonoverlap_violating_pairs"],
                    "active_pairs": solve_info.get("active_pairs", 0),
                    "active_refinements": solve_info.get("refinements", 0),
                    "new_pairs_added": solve_info.get("new_pairs_added", 0),
                    "unresolved_new_pairs": solve_info.get("unresolved_new_pairs", 0),
                    "step_norm": step_norm,
                    "relative_step": relative_step,
                    "relative_stage_decrease": relative_decrease,
                    "cplex_surrogate_objective": solve_info.get("surrogate", float("nan")),
                    "cplex_solve_time_sec": solve_info.get("cplex_time", 0.0),
                    "cplex_status": solve_info.get("status", "unknown"),
                    "total_runtime_sec": time.time() - iter_start,
                }
                history.append(row)

                log_msg(
                    f"Iter {global_iter + 1:3d} | stage={stage_name:<16} | "
                    f"F_ref={F_reference:.6e} | HPWL={reference_next['eq5_hpwl']:.2f} | "
                    f"OvSum={reference_next['eq8_overlap_hat_sum']:.3f} | "
                    f"OvPairs={reference_next['eq3_nonoverlap_violating_pairs']} | "
                    f"rho={rho:.4g} | {selected_kind} | rel_step={relative_step:.2e}"
                )

                zk = selected_z
                F_curr = F_next
                recent_stage_F.append(F_next)
                global_iter += 1

                if F_next < best_stage_F * (1.0 - EARLY_STOP_RTOL):
                    best_stage_F = F_next
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if relative_step <= REL_STEP_TOL:
                    log_msg(
                        f"Stage {stage_name} stopped by relative step "
                        f"{relative_step:.3e} <= {REL_STEP_TOL:.3e}."
                    )
                    break
                if no_improve_count >= EARLY_STOP_PATIENCE:
                    log_msg(
                        f"Stage {stage_name} early-stopped after "
                        f"{EARLY_STOP_PATIENCE} iterations without material improvement."
                    )
                    break

            if not completed_all_stages:
                break

        total_time = time.time() - start_time
        log_msg(
            f"PDCA finished: status={final_status}, iterations={global_iter}, "
            f"best F_ref={best_reference_F:.6e}, time={total_time:.2f}s"
        )
        return DCAResult(
            z_star=best_reference_z,
            z_last=zk,
            history=history,
            final_status=final_status,
            total_time_sec=total_time,
            converged=completed_all_stages,
            iterations_run=global_iter,
            best_reference_objective=best_reference_F,
        )


    log_msg("Baseline-first controlled non-monotone PDCA ready.")
    ''',
    original[11],
)


run_cell = code_cell(
    r'''
    # Cell 13 - Select the strongest available incumbent, then run PDCA

    def select_start_incumbent() -> Tuple[np.ndarray, str, Dict[str, Any]]:
        candidates: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []

        drl_start = project_to_hard_boundary(Z0_PAPER_CENTERS, DB)
        candidates.append((
            "drl_checkpoint",
            drl_start,
            objective_metrics(drl_start, DB, EVAL_GAMMA_OVERLAP),
        ))

        baseline_root = (
            PROJECT_ROOT / "experiments" / BENCHMARK /
            "activeset_gb_100_gov_5000_rho_1e-02_iter300_acc_q3"
        )
        baseline_paths = sorted(
            baseline_root.glob("run_*/placements/z_star_paper_centers.npy")
        )
        for baseline_path in baseline_paths:
            try:
                baseline = np.asarray(np.load(baseline_path), dtype=np.float64)
                if baseline.shape != (DB.num_macros, 2) or not np.isfinite(baseline).all():
                    log_msg(f"Skip invalid baseline artifact: {baseline_path}")
                    continue
                baseline = project_to_hard_boundary(baseline, DB)
                candidates.append((
                    str(baseline_path),
                    baseline,
                    objective_metrics(baseline, DB, EVAL_GAMMA_OVERLAP),
                ))
            except Exception as exc:
                log_msg(f"Skip unreadable baseline artifact {baseline_path}: {exc}")

        for source, _, metrics in candidates:
            log_msg(
                f"START CANDIDATE | source={source} | "
                f"F_ref/1e5={metrics['eq6_penalized_objective']/1e5:.6f} | "
                f"HPWL/1e5={metrics['eq5_hpwl']/1e5:.6f}"
            )
        source, placement, metrics = min(
            candidates,
            key=lambda item: float(item[2]["eq6_penalized_objective"]),
        )
        log_msg(f"SELECTED START INCUMBENT: {source}")
        return placement.copy(), source, metrics


    Z_RUN_START, START_SOURCE, METRICS_START = select_start_incumbent()
    np.save(placements_dir / "z_start_incumbent_centers.npy", Z_RUN_START)

    DCA_RES = run_baseline_first_pdca(
        z0=Z_RUN_START,
        db=DB,
        stage_schedule=STAGE_SCHEDULE,
        cplex_models_folder=cplex_models_dir,
    )

    Z_PDCA = DCA_RES.z_star.copy()
    np.save(placements_dir / "z_best_reference_pdca_centers.npy", Z_PDCA)
    ''',
    original[12],
)


lns_cell = code_cell(
    r'''
    # Cell 14 - Optional topology-changing soft LNS-MIQP

    TOPOLOGY_LNS_HISTORY: List[Dict[str, Any]] = []


    def center_hpwl_subgradient(centers: np.ndarray, db: PlacementDB) -> np.ndarray:
        gradient = np.zeros_like(centers, dtype=np.float64)
        for net in db.nets:
            ids = net.macro_ids
            points = centers[ids]
            for axis in (0, 1):
                values = points[:, axis]
                max_mask = values >= values.max() - 1e-9
                min_mask = values <= values.min() + 1e-9
                np.add.at(gradient[:, axis], ids[max_mask], 1.0 / max(int(max_mask.sum()), 1))
                np.add.at(gradient[:, axis], ids[min_mask], -1.0 / max(int(min_mask.sum()), 1))
        return gradient


    def select_topology_critical_pairs(base: np.ndarray, db: PlacementDB, limit: int) -> np.ndarray:
        pair = pair_terms_eq3_eq8(base, db)
        radius = TOPOLOGY_LNS_TRUST_RADIUS_FRACTION * min(
            db.canvas_width, db.canvas_height
        )
        gap_x = np.maximum(pair["wij"] - pair["dx"], 0.0)
        gap_y = np.maximum(pair["hij"] - pair["dy"], 0.0)
        separable_in_trust_box = np.minimum(gap_x, gap_y) <= 2.0 * radius
        near = np.flatnonzero(
            (pair["Nij"] <= TOPOLOGY_LNS_NEAR_MARGIN) & separable_in_trust_box
        )
        if near.size == 0:
            return np.empty(0, dtype=np.int64)

        desired = -center_hpwl_subgradient(base, db)
        i = pair["pair_i"][near]
        j = pair["pair_j"][near]
        dx = base[i, 0] - base[j, 0]
        dy = base[i, 1] - base[j, 1]
        choose_x = np.abs(dx) / pair["wij"][near] >= np.abs(dy) / pair["hij"][near]
        pressure_x = np.abs(desired[i, 0] - desired[j, 0])
        pressure_y = np.abs(desired[i, 1] - desired[j, 1])
        pressure = np.where(choose_x, pressure_x, pressure_y)
        severity = np.maximum(-pair["Nij"][near], 0.0)
        score = 1000.0 * severity + pressure
        order = np.argsort(-score)
        return near[order[: min(limit, near.size)]]


    def solve_topology_lns_round(
        base: np.ndarray,
        db: PlacementDB,
        round_id: int,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        critical_idx = select_topology_critical_pairs(
            base, db, TOPOLOGY_LNS_CRITICAL_PAIRS
        )
        if critical_idx.size == 0:
            return None, {"round": round_id, "status": "no_critical_pairs", "accepted": False}

        full_active = select_active_set(
            base, db, margin=ACTIVE_SET_MARGIN, extra_indices=critical_idx
        )
        soft_idx = np.setdiff1d(full_active.active_idx, critical_idx)
        soft_active = active_set_from_indices(soft_idx, db)
        yk = compute_subgradient_H_activeset(
            base,
            db,
            active_set=soft_active,
            gamma_overlap=EVAL_GAMMA_OVERLAP,
            # Keep the proximal term centered explicitly below.  Excluding
            # rho*base here avoids large cancelling MIQP coefficients.
            rho=0.0,
        )

        widths, heights = macro_geometry(db)
        pair_i, pair_j = np.triu_indices(db.num_macros, k=1)
        pair_w = (widths[pair_i] + widths[pair_j]) / 2.0
        pair_h = (heights[pair_i] + heights[pair_j]) / 2.0
        trust_radius = TOPOLOGY_LNS_TRUST_RADIUS_FRACTION * min(
            db.canvas_width, db.canvas_height
        )
        model = Model(name=f"adaptec3_topology_lns_{round_id}")
        model.parameters.threads = CPLEX_THREADS
        model.parameters.parallel = 1
        model.parameters.timelimit = TOPOLOGY_LNS_TIME_LIMIT_SEC
        model.parameters.mip.tolerances.mipgap = TOPOLOGY_LNS_MIP_GAP
        model.parameters.emphasis.mip = 1
        infinity = model.infinity

        x = [
            model.continuous_var(
                lb=max(float(widths[k] / 2.0), float(base[k, 0] - trust_radius)),
                ub=min(float(db.canvas_width - widths[k] / 2.0), float(base[k, 0] + trust_radius)),
                name=f"x_{k}",
            )
            for k in range(db.num_macros)
        ]
        y = [
            model.continuous_var(
                lb=max(float(heights[k] / 2.0), float(base[k, 1] - trust_radius)),
                ub=min(float(db.canvas_height - heights[k] / 2.0), float(base[k, 1] + trust_radius)),
                name=f"y_{k}",
            )
            for k in range(db.num_macros)
        ]
        xmax = model.continuous_var_list(len(db.nets), lb=-infinity, name="xmax")
        xmin = model.continuous_var_list(len(db.nets), lb=-infinity, name="xmin")
        ymax = model.continuous_var_list(len(db.nets), lb=-infinity, name="ymax")
        ymin = model.continuous_var_list(len(db.nets), lb=-infinity, name="ymin")
        q = model.continuous_var_list(soft_active.num_active, lb=0.0, name="q")
        move_x = model.continuous_var_list(db.num_macros, lb=0.0, name="move_x")
        move_y = model.continuous_var_list(db.num_macros, lb=0.0, name="move_y")

        constraints = []
        for net_id, net in enumerate(db.nets):
            for macro_id in net.macro_ids:
                k = int(macro_id)
                constraints.extend([
                    xmax[net_id] >= x[k],
                    xmin[net_id] <= x[k],
                    ymax[net_id] >= y[k],
                    ymin[net_id] <= y[k],
                ])
        for local_idx in range(soft_active.num_active):
            i = int(soft_active.act_i[local_idx])
            j = int(soft_active.act_j[local_idx])
            iw = 1.0 / float(soft_active.act_wij[local_idx])
            ih = 1.0 / float(soft_active.act_hij[local_idx])
            qk = q[local_idx]
            constraints.extend([
                qk - iw * x[i] + iw * x[j] >= -1.0,
                qk + iw * x[i] - iw * x[j] >= -1.0,
                qk - ih * y[i] + ih * y[j] >= -1.0,
                qk + ih * y[i] - ih * y[j] >= -1.0,
            ])
        for k in range(db.num_macros):
            constraints.extend([
                move_x[k] >= x[k] - float(base[k, 0]),
                move_x[k] >= float(base[k, 0]) - x[k],
                move_y[k] >= y[k] - float(base[k, 1]),
                move_y[k] >= float(base[k, 1]) - y[k],
            ])
        model.add_constraints(constraints)
        del constraints

        big_m_x = float(2.0 * db.canvas_width)
        big_m_y = float(2.0 * db.canvas_height)
        clearance = 1e-3
        for pair_index in critical_idx:
            i = int(pair_i[pair_index])
            j = int(pair_j[pair_index])
            left = model.binary_var(name=f"left_{i}_{j}")
            right = model.binary_var(name=f"right_{i}_{j}")
            below = model.binary_var(name=f"below_{i}_{j}")
            above = model.binary_var(name=f"above_{i}_{j}")
            model.add_constraint(left + right + below + above == 1)
            model.add_constraint(
                x[i] - x[j] <= -float(pair_w[pair_index]) - clearance + big_m_x * (1 - left)
            )
            model.add_constraint(
                x[j] - x[i] <= -float(pair_w[pair_index]) - clearance + big_m_x * (1 - right)
            )
            model.add_constraint(
                y[i] - y[j] <= -float(pair_h[pair_index]) - clearance + big_m_y * (1 - below)
            )
            model.add_constraint(
                y[j] - y[i] <= -float(pair_h[pair_index]) - clearance + big_m_y * (1 - above)
            )

        hpwl = model.sum(
            xmax[e] - xmin[e] + ymax[e] - ymin[e]
            for e in range(len(db.nets))
        )
        overlap = EVAL_GAMMA_OVERLAP * model.sum(q)
        displacement = TOPOLOGY_LNS_MOVE_WEIGHT * (
            model.sum(move_x) + model.sum(move_y)
        )
        linearized_h = -model.scal_prod(x, yk[:, 0].tolist()) - model.scal_prod(
            y, yk[:, 1].tolist()
        )
        # MILP rather than MIQP: the L1 trust term is numerically more robust
        # for CPLEX MIP subproblems at Adaptec3's coordinate scale.
        model.minimize(hpwl + overlap + displacement + linearized_h)

        solve_start = time.time()
        solution = model.solve(log_output=False)
        solve_time = time.time() - solve_start
        status = str(model.solve_details.status) if model.solve_details else "unknown"
        if solution is None:
            model.end()
            gc.collect()
            return None, {
                "round": round_id,
                "status": status,
                "solve_time_sec": solve_time,
                "critical_pairs": int(critical_idx.size),
                "accepted": False,
            }

        candidate = np.column_stack((
            [solution.get_value(variable) for variable in x],
            [solution.get_value(variable) for variable in y],
        ))
        model.end()
        gc.collect()
        candidate = project_to_hard_boundary(candidate, db)

        base_metrics = objective_metrics(base, db, EVAL_GAMMA_OVERLAP)
        candidate_metrics = objective_metrics(candidate, db, EVAL_GAMMA_OVERLAP)
        base_F = float(base_metrics["eq6_penalized_objective"])
        candidate_F = float(candidate_metrics["eq6_penalized_objective"])
        accepted = candidate_F < base_F * (1.0 - TOPOLOGY_LNS_MIN_REL_IMPROVEMENT)
        record = {
            "round": round_id,
            "status": status,
            "solve_time_sec": solve_time,
            "critical_pairs": int(critical_idx.size),
            "soft_active_pairs": soft_active.num_active,
            "base_F_ref": base_F,
            "candidate_F_ref": candidate_F,
            "base_HPWL": base_metrics["eq5_hpwl"],
            "candidate_HPWL": candidate_metrics["eq5_hpwl"],
            "base_overlap_sum": base_metrics["eq8_overlap_hat_sum"],
            "candidate_overlap_sum": candidate_metrics["eq8_overlap_hat_sum"],
            "candidate_overlap_pairs": candidate_metrics["eq3_nonoverlap_violating_pairs"],
            "accepted": bool(accepted),
        }
        return candidate, record


    def run_topology_lns(base: np.ndarray, db: PlacementDB) -> np.ndarray:
        current = base.copy()
        if not RUN_TOPOLOGY_LNS:
            log_msg("Topology LNS disabled; using best PDCA placement.")
            return current
        for round_id in range(1, TOPOLOGY_LNS_ROUNDS + 1):
            trial, record = solve_topology_lns_round(current, db, round_id)
            TOPOLOGY_LNS_HISTORY.append(record)
            log_msg("Topology LNS: " + json.dumps(record))
            if trial is None or not record.get("accepted", False):
                break
            current = trial
            np.save(
                placements_dir / f"z_topology_lns_round_{round_id}.npy",
                current,
            )
        return current


    Z_STAR = run_topology_lns(Z_PDCA, DB)
    ''')


export_cell = code_cell(
    r'''
    # Cell 15 - Save final placement and write Bookshelf .pl with global origin

    def paper_centers_to_local_lower_left(
        centers_local: np.ndarray, db: PlacementDB
    ) -> np.ndarray:
        widths, heights = macro_geometry(db)
        return np.column_stack((
            centers_local[:, 0] - widths / 2.0,
            centers_local[:, 1] - heights / 2.0,
        )).astype(np.float64)


    def paper_centers_to_global_lower_left(
        centers_local: np.ndarray, db: PlacementDB
    ) -> np.ndarray:
        lower_left = paper_centers_to_local_lower_left(centers_local, db)
        lower_left[:, 0] += db.x_min
        lower_left[:, 1] += db.y_min
        return lower_left


    Z_STAR_LOWER_LEFT = paper_centers_to_global_lower_left(Z_STAR, DB)
    np.save(placements_dir / "z_star_paper_centers.npy", Z_STAR)
    np.save(placements_dir / "z_star_lower_left_global.npy", Z_STAR_LOWER_LEFT)


    def write_bookshelf_pl_from_paper_centers(
        centers_local: np.ndarray,
        db: PlacementDB,
        source_pl: Path,
        dest_pl: Path,
    ):
        lower_left = paper_centers_to_global_lower_left(centers_local, db)
        coord_by_name = {
            macro.name: lower_left[idx]
            for idx, macro in enumerate(db.macros)
        }
        out_lines = []
        with source_pl.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                fields = raw.strip().split()
                if fields and fields[0] in coord_by_name and len(fields) >= 3:
                    x_value, y_value = coord_by_name[fields[0]]
                    fields[1] = f"{x_value:.6f}"
                    fields[2] = f"{y_value:.6f}"
                    out_lines.append("\t".join(fields))
                else:
                    out_lines.append(raw.rstrip("\r\n"))
        with dest_pl.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(out_lines) + "\n")


    final_pl_path = placements_dir / f"{BENCHMARK}.improved_drl_pdca_lns.pl"
    write_bookshelf_pl_from_paper_centers(
        centers_local=Z_STAR,
        db=DB,
        source_pl=bench_dir / f"{BENCHMARK}.pl",
        dest_pl=final_pl_path,
    )
    log_msg(f"Saved final global Bookshelf placement to: {final_pl_path}")
    ''',
    original[13],
)


save_metrics_cell = code_cell(
    r'''
    # Cell 16 - Save metrics, histories and comparison table

    metrics_start_payload = dict(METRICS_START)
    metrics_start_payload["source"] = START_SOURCE
    with (metrics_dir / "metrics_start.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_start_payload, handle, indent=2)

    METRICS_AFTER = objective_metrics(Z_STAR, DB, EVAL_GAMMA_OVERLAP)
    with (metrics_dir / "metrics_after.json").open("w", encoding="utf-8") as handle:
        json.dump(METRICS_AFTER, handle, indent=2)

    csv_path = metrics_dir / "baseline_first_pdca_history.csv"
    if DCA_RES.history:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(DCA_RES.history[0].keys())
            )
            writer.writeheader()
            writer.writerows(DCA_RES.history)
    else:
        csv_path.write_text("", encoding="utf-8")

    lns_history_path = metrics_dir / "topology_lns_history.json"
    with lns_history_path.open("w", encoding="utf-8") as handle:
        json.dump(TOPOLOGY_LNS_HISTORY, handle, indent=2)

    log_msg(f"Saved DCA history: {csv_path}")
    log_msg(f"Saved topology LNS history: {lns_history_path}")

    header = (
        f"{'Metric':<30} | {'DRL':>14} | {'Start incumbent':>16} | "
        f"{'Final best':>14} | {'Final-Start':>14}"
    )
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    comparison_rows = [
        ("F_ref / 1e5", METRICS_BEFORE["eq6_penalized_objective"] / 1e5, METRICS_START["eq6_penalized_objective"] / 1e5, METRICS_AFTER["eq6_penalized_objective"] / 1e5),
        ("HPWL / 1e5", METRICS_BEFORE["eq5_hpwl"] / 1e5, METRICS_START["eq5_hpwl"] / 1e5, METRICS_AFTER["eq5_hpwl"] / 1e5),
        ("Overlap penalty sum", METRICS_BEFORE["eq8_overlap_hat_sum"], METRICS_START["eq8_overlap_hat_sum"], METRICS_AFTER["eq8_overlap_hat_sum"]),
        ("Overlap violating pairs", METRICS_BEFORE["eq3_nonoverlap_violating_pairs"], METRICS_START["eq3_nonoverlap_violating_pairs"], METRICS_AFTER["eq3_nonoverlap_violating_pairs"]),
        ("Boundary violating macros", METRICS_BEFORE["eq4_boundary_violating_macros"], METRICS_START["eq4_boundary_violating_macros"], METRICS_AFTER["eq4_boundary_violating_macros"]),
    ]
    for name, before, start, after in comparison_rows:
        print(
            f"{name:<30} | {before:>14.6f} | {start:>16.6f} | "
            f"{after:>14.6f} | {after-start:>+14.6f}"
        )
    print("-" * len(header) + "\n")
    ''',
    original[14],
)


plot_cell = code_cell(
    r'''
    # Cell 17 - Placement and convergence plots

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    W, H = DB.canvas_width, DB.canvas_height
    widths, heights = macro_geometry(DB)
    ll_before = paper_centers_to_local_lower_left(Z_RUN_START, DB)
    ll_after = paper_centers_to_local_lower_left(Z_STAR, DB)

    for axis, lower_left, metrics, title, color in [
        (axes[0], ll_before, METRICS_START, "Selected start incumbent", "cyan"),
        (axes[1], ll_after, METRICS_AFTER, "Final best placement", "lightgreen"),
    ]:
        axis.set_xlim(0, W)
        axis.set_ylim(0, H)
        axis.set_aspect("equal")
        axis.set_title(
            f"{title}\nHPWL/1e5={metrics['eq5_hpwl']/1e5:.3f}, "
            f"F_ref/1e5={metrics['eq6_penalized_objective']/1e5:.3f}, "
            f"overlaps={metrics['eq3_nonoverlap_violating_pairs']}"
        )
        for i in range(DB.num_macros):
            axis.add_patch(patches.Rectangle(
                (lower_left[i, 0], lower_left[i, 1]),
                widths[i], heights[i], linewidth=0.35,
                edgecolor="black", facecolor=color, alpha=0.35,
            ))

    plt.tight_layout()
    placement_plot_path = plots_dir / "start_incumbent_vs_final_best.png"
    plt.savefig(placement_plot_path, dpi=200)
    plt.close()

    if DCA_RES.history:
        iterations = [row["global_iteration"] for row in DCA_RES.history]
        f_values = [row["F_ref_after"] / 1e5 for row in DCA_RES.history]
        hpwl_values = [row["HPWL_after"] / 1e5 for row in DCA_RES.history]
        overlap_sums = [row["overlap_hat_sum"] for row in DCA_RES.history]
        rho_values = [row["rho"] for row in DCA_RES.history]

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs[0, 0].plot(iterations, f_values, "b-")
        axs[0, 0].axhline(62.84, color="black", linestyle="--", label="target 62.84")
        axs[0, 0].set_title("Reference F / 1e5")
        axs[0, 0].legend()
        axs[0, 1].plot(iterations, hpwl_values, "g-")
        axs[0, 1].axhline(50.0, color="black", linestyle="--", label="target 50")
        axs[0, 1].set_title("HPWL / 1e5")
        axs[0, 1].legend()
        axs[1, 0].plot(iterations, overlap_sums, "r-")
        axs[1, 0].axhline(256.8, color="black", linestyle="--", label="target budget 256.8")
        axs[1, 0].set_title("Overlap penalty sum")
        axs[1, 0].legend()
        axs[1, 1].plot(iterations, rho_values, "m-")
        axs[1, 1].set_title("Fixed rho")
        for axis in axs.flat:
            axis.set_xlabel("Global iteration")
            axis.grid(True)
        plt.tight_layout()
        convergence_plot_path = plots_dir / "improved_convergence.png"
        plt.savefig(convergence_plot_path, dpi=200)
        plt.close()

    log_msg("All improved plots generated.")
    ''',
    original[15],
)


summary_cell = code_cell(
    r'''
    # Cell 18 - Final auditable summary

    log_msg("=" * 96)
    log_msg("IMPROVED ADAPTEC3 EXPERIMENT SUMMARY")
    log_msg("=" * 96)
    log_msg(f"PDCA status                 : {DCA_RES.final_status}")
    log_msg(f"PDCA iterations             : {DCA_RES.iterations_run}")
    log_msg(f"PDCA runtime                : {DCA_RES.total_time_sec:.2f}s")
    log_msg(f"Start incumbent source      : {START_SOURCE}")
    log_msg(f"Start F_ref / 1e5           : {METRICS_START['eq6_penalized_objective']/1e5:.6f}")
    log_msg(f"Start HPWL / 1e5            : {METRICS_START['eq5_hpwl']/1e5:.6f}")
    log_msg(f"Topology LNS rounds attempted: {len(TOPOLOGY_LNS_HISTORY)}")
    log_msg(f"Final F_ref / 1e5           : {METRICS_AFTER['eq6_penalized_objective']/1e5:.6f}")
    log_msg(f"Final HPWL / 1e5            : {METRICS_AFTER['eq5_hpwl']/1e5:.6f}")
    log_msg(f"Final overlap sum           : {METRICS_AFTER['eq8_overlap_hat_sum']:.6f}")
    log_msg(f"Final overlap pairs         : {METRICS_AFTER['eq3_nonoverlap_violating_pairs']}")
    log_msg(f"Final boundary violations   : {METRICS_AFTER['eq4_boundary_violating_macros']}")
    log_msg(f"Output placement            : {final_pl_path}")
    log_msg("=" * 96)

    assert final_pl_path.is_file(), "Final .pl file must exist"
    assert csv_path.is_file(), "PDCA history must exist"
    assert (metrics_dir / "metrics_start.json").is_file(), "Start metrics must exist"
    assert (metrics_dir / "metrics_after.json").is_file(), "Final metrics must exist"
    assert (
        METRICS_AFTER["eq6_penalized_objective"]
        <= METRICS_START["eq6_penalized_objective"]
        + descent_tolerance(METRICS_START["eq6_penalized_objective"])
    ), "Best-incumbent safeguard failed: final F_ref is worse than the selected start"
    assert METRICS_AFTER["eq4_boundary_violating_macros"] == 0, (
        "Hard boundary failed; inspect solver tolerance and projection"
    )
    print("SUCCESS: improved notebook outputs verified.")
    ''',
    original[16],
)


notebook["cells"] = [
    intro,
    configuration,
    logging_cell,
    validation_cell,
    database_cell,
    load_cell,
    metrics_cell,
    active_set_cell,
    audit_cell,
    subgradient_cell,
    model_cell,
    solve_cell,
    algorithm_cell,
    run_cell,
    lns_cell,
    export_cell,
    save_metrics_cell,
    plot_cell,
    summary_cell,
]

notebook.setdefault("metadata", {})["improved_from"] = SOURCE.name
notebook["metadata"]["improvement_notes"] = (
    "baseline-first incumbent, hard boundary, fixed gamma=5000 and rho=0.01, "
    "single active-set QP, bounded non-monotone acceleration, best-reference tracking, "
    "optional topology LNS, global origin export"
)

with DESTINATION.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(notebook, handle, indent=1, ensure_ascii=False)
    handle.write("\n")

print(DESTINATION)

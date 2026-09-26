from __future__ import annotations

import copy
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "cplex_resolver"


SETTINGS = {
    "adaptec3": {
        "rho_candidates": [0.005, 0.0125, 0.025],
        "screen_iterations": 8,
        "rho_min": 0.003,
        "rho_max": 0.04,
        "active_margin": 1.25,
        "nonmonotone_window": 5,
        "max_rel_increase": 5.0e-3,
        "beta_max": 0.90,
        "rel_step_tol": 1.0e-6,
        "early_stop_patience": 24,
        "early_stop_rtol": 1.0e-6,
        "main_stages": [
            {"name": "hpwl_tilt", "gamma_overlap": 4500.0, "rho_multiplier": 0.75, "max_iter": 20},
            {"name": "reference_opt", "gamma_overlap": 5000.0, "rho_multiplier": 1.00, "max_iter": 80},
            {"name": "overlap_polish", "gamma_overlap": 6500.0, "rho_multiplier": 1.50, "max_iter": 20},
        ],
    },
    "adaptec4": {
        "rho_candidates": [0.0075, 0.015, 0.03],
        "screen_iterations": 6,
        "rho_min": 0.004,
        "rho_max": 0.05,
        "active_margin": 1.0,
        "nonmonotone_window": 3,
        "max_rel_increase": 2.0e-3,
        "beta_max": 0.75,
        "rel_step_tol": 2.0e-6,
        "early_stop_patience": 20,
        "early_stop_rtol": 2.0e-6,
        "main_stages": [
            {"name": "hpwl_tilt", "gamma_overlap": 4000.0, "rho_multiplier": 0.80, "max_iter": 15},
            {"name": "reference_opt", "gamma_overlap": 5000.0, "rho_multiplier": 1.00, "max_iter": 60},
            {"name": "overlap_polish", "gamma_overlap": 6000.0, "rho_multiplier": 1.50, "max_iter": 15},
        ],
    },
}


def source_lines(source: str) -> list[str]:
    return textwrap.dedent(source).lstrip("\n").splitlines(keepends=True)


def set_code(cell: dict, source: str) -> None:
    cell["source"] = source_lines(source)
    cell["execution_count"] = None
    cell["outputs"] = []


def replace_required(source: str, old: str, new: str, label: str) -> str:
    if old not in source:
        raise RuntimeError(f"Could not find {label!r} in notebook source")
    return source.replace(old, new)


def build_notebook(benchmark: str) -> Path:
    cfg = SETTINGS[benchmark]
    source_path = NOTEBOOK_DIR / f"{benchmark}_drl_pdca_cplex_improved.ipynb"
    destination = NOTEBOOK_DIR / f"{benchmark}_drl_pdca_cplex_param_tuned.ipynb"
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    notebook = copy.deepcopy(notebook)

    title = benchmark.capitalize()
    intro = f"""
    # {title} — accelerated rho-screening PDCA

    Notebook mới này giữ nguyên notebook `*_improved.ipynb` và chuyển sang chiến lược
    tuning tham số có safeguard:

    - bắt đầu từ incumbent tốt nhất giữa DRL checkpoint và baseline cũ;
    - chạy độc lập các ứng viên `rho={cfg['rho_candidates']}` với
      `{cfg['screen_iterations']}` vòng sàng lọc mỗi ứng viên;
    - chọn `rho` cho `F_ref` thấp nhất rồi chạy ba stage HPWL/reference/overlap;
    - dùng accelerated candidate với `beta <= {cfg['beta_max']}` và cửa sổ
      non-monotone {cfg['nonmonotone_window']} bước;
    - active-set margin `{cfg['active_margin']}` để bắt các cặp gần va chạm sớm hơn;
    - luôn lưu incumbent tốt nhất nên thử nghiệm tham số không được phép làm kết quả cuối tệ hơn.

    Đây là full parameter-tuning run, không phải smoke test. Mọi giá trị báo cáo dùng
    `EVAL_GAMMA_OVERLAP=5000`; giá trị trong bài có thể chia cho `1e5`.
    """
    notebook["cells"][0]["source"] = source_lines(intro)

    config = "".join(notebook["cells"][1]["source"])
    old_schedule = '''STAGE_SCHEDULE: List[Dict[str, Any]] = [
    {"name": "baseline_recovery", "gamma_overlap": 5000.0, "rho_init": 0.01, "max_iter": 300},
]
MAX_DCA_ITER: int = sum(int(stage["max_iter"]) for stage in STAGE_SCHEDULE)
RHO: float = float(STAGE_SCHEDULE[0]["rho_init"])  # compatibility/audit alias'''
    new_schedule = f'''RHO_SCREEN_CANDIDATES: Tuple[float, ...] = {tuple(cfg["rho_candidates"])!r}
RHO_SCREEN_ITERATIONS: int = {cfg["screen_iterations"]}
MAIN_STAGE_TEMPLATES: List[Dict[str, Any]] = {cfg["main_stages"]!r}

# Compatibility schedule; the run cell builds each screening/main schedule explicitly.
STAGE_SCHEDULE: List[Dict[str, Any]] = [
    {{"name": "rho_screen", "gamma_overlap": 5000.0, "rho_init": RHO_SCREEN_CANDIDATES[0], "max_iter": RHO_SCREEN_ITERATIONS}},
]
MAX_DCA_ITER: int = sum(int(stage["max_iter"]) for stage in STAGE_SCHEDULE)
RHO: float = float(STAGE_SCHEDULE[0]["rho_init"])  # compatibility/audit alias'''
    config = replace_required(config, old_schedule, new_schedule, "stage schedule")
    config = replace_required(config, "RHO_MIN: float = 0.01", f"RHO_MIN: float = {cfg['rho_min']}", "rho min")
    config = replace_required(config, "RHO_MAX: float = 0.01", f"RHO_MAX: float = {cfg['rho_max']}", "rho max")
    config = replace_required(config, "ACTIVE_SET_MARGIN: float = 1.0", f"ACTIVE_SET_MARGIN: float = {cfg['active_margin']}", "active margin")
    config = replace_required(config, "NONMONOTONE_WINDOW: int = 3", f"NONMONOTONE_WINDOW: int = {cfg['nonmonotone_window']}", "non-monotone window")
    config = replace_required(
        config,
        "MAX_REL_OBJECTIVE_INCREASE: float = 1.5e-3",
        f"MAX_REL_OBJECTIVE_INCREASE: float = {cfg['max_rel_increase']!r}\nACCELERATION_BETA_MAX: float = {cfg['beta_max']}",
        "non-monotone cap",
    )
    config = replace_required(config, "REL_STEP_TOL: float = 1e-5", f"REL_STEP_TOL: float = {cfg['rel_step_tol']!r}", "step tolerance")
    config = replace_required(config, "EARLY_STOP_PATIENCE: int = 30", f"EARLY_STOP_PATIENCE: int = {cfg['early_stop_patience']}", "patience")
    config = replace_required(config, "EARLY_STOP_RTOL: float = 1e-5", f"EARLY_STOP_RTOL: float = {cfg['early_stop_rtol']!r}", "early-stop tolerance")
    config = replace_required(
        config,
        'print(f"  Fixed rho              : {RHO}")',
        'print(f"  Rho screen candidates  : {RHO_SCREEN_CANDIDATES}")',
        "rho print",
    )
    config = replace_required(
        config,
        'print(f"  Non-monotone window/cap: {NONMONOTONE_WINDOW} / {MAX_REL_OBJECTIVE_INCREASE:.4%}")',
        'print(f"  Non-monotone window/cap: {NONMONOTONE_WINDOW} / {MAX_REL_OBJECTIVE_INCREASE:.4%}")\nprint(f"  Acceleration beta cap  : {ACCELERATION_BETA_MAX}")',
        "acceleration print",
    )
    set_code(notebook["cells"][1], config)

    logging_source = "".join(notebook["cells"][2]["source"])
    old_exp_name = '''exp_name = (
    f"baseline_first_gov_{int(EVAL_GAMMA_OVERLAP)}_rho_1e-02_"
    f"nonmono_q{NONMONOTONE_WINDOW}"
)'''
    logging_source = replace_required(
        logging_source,
        old_exp_name,
        'exp_name = f"parameter_tuned_rho_accel_q{NONMONOTONE_WINDOW}"',
        "experiment name",
    )
    logging_source = replace_required(
        logging_source,
        '"stage_schedule": STAGE_SCHEDULE,',
        '"stage_schedule": STAGE_SCHEDULE,\n    "rho_screen_candidates": list(RHO_SCREEN_CANDIDATES),\n    "rho_screen_iterations": RHO_SCREEN_ITERATIONS,\n    "main_stage_templates": MAIN_STAGE_TEMPLATES,\n    "acceleration_beta_max": ACCELERATION_BETA_MAX,',
        "experiment config",
    )
    set_code(notebook["cells"][2], logging_source)

    algorithm = "".join(notebook["cells"][12]["source"])
    algorithm = replace_required(
        algorithm,
        "beta = (self.t - 1.0) / t_next",
        "beta = min((self.t - 1.0) / t_next, ACCELERATION_BETA_MAX)",
        "acceleration beta",
    )
    set_code(notebook["cells"][12], algorithm)

    original_run = "".join(notebook["cells"][13]["source"])
    marker = "Z_RUN_START, START_SOURCE, METRICS_START = select_start_incumbent()"
    if marker not in original_run:
        raise RuntimeError("Could not locate run-cell incumbent marker")
    run_prefix = original_run[: original_run.index(marker)]
    run_suffix = r'''
    Z_RUN_START, START_SOURCE, METRICS_START = select_start_incumbent()
    np.save(placements_dir / "z_start_incumbent_centers.npy", Z_RUN_START)

    PARAMETER_HISTORY: List[Dict[str, Any]] = []
    PARAMETER_TRIAL_SUMMARY: List[Dict[str, Any]] = []
    SCREEN_RESULTS: List[DCAResult] = []

    for trial_index, rho_value in enumerate(RHO_SCREEN_CANDIDATES, start=1):
        trial_name = f"screen_{trial_index:02d}_rho_{rho_value:.4g}".replace(".", "p")
        trial_model_dir = cplex_models_dir / trial_name
        trial_model_dir.mkdir(parents=True, exist_ok=True)
        trial_schedule = [{
            "name": trial_name,
            "gamma_overlap": EVAL_GAMMA_OVERLAP,
            "rho_init": float(rho_value),
            "max_iter": RHO_SCREEN_ITERATIONS,
        }]
        MAX_DCA_ITER = RHO_SCREEN_ITERATIONS
        log_msg("-" * 96)
        log_msg(f"RHO SCREEN {trial_index}/{len(RHO_SCREEN_CANDIDATES)}: rho={rho_value}")
        result = run_baseline_first_pdca(
            z0=Z_RUN_START,
            db=DB,
            stage_schedule=trial_schedule,
            cplex_models_folder=trial_model_dir,
        )
        SCREEN_RESULTS.append(result)
        for row in result.history:
            enriched = dict(row)
            enriched["parameter_trial"] = trial_name
            enriched["trial_kind"] = "rho_screen"
            PARAMETER_HISTORY.append(enriched)
        trial_metrics = objective_metrics(result.z_star, DB, EVAL_GAMMA_OVERLAP)
        PARAMETER_TRIAL_SUMMARY.append({
            "trial": trial_name,
            "kind": "rho_screen",
            "rho": float(rho_value),
            "iterations": result.iterations_run,
            "status": result.final_status,
            "F_ref": float(trial_metrics["eq6_penalized_objective"]),
            "HPWL": float(trial_metrics["eq5_hpwl"]),
            "overlap_sum": float(trial_metrics["eq8_overlap_hat_sum"]),
        })
        np.save(placements_dir / f"z_{trial_name}_centers.npy", result.z_star)

    best_screen_index = int(np.argmin([
        result.best_reference_objective for result in SCREEN_RESULTS
    ]))
    BEST_SCREEN_RESULT = SCREEN_RESULTS[best_screen_index]
    SELECTED_RHO = float(RHO_SCREEN_CANDIDATES[best_screen_index])
    Z_SCREEN_BEST = BEST_SCREEN_RESULT.z_star.copy()
    log_msg(
        f"SELECTED RHO={SELECTED_RHO} with "
        f"F_ref/1e5={BEST_SCREEN_RESULT.best_reference_objective/1e5:.6f}"
    )

    MAIN_STAGE_SCHEDULE: List[Dict[str, Any]] = []
    for template in MAIN_STAGE_TEMPLATES:
        MAIN_STAGE_SCHEDULE.append({
            "name": str(template["name"]),
            "gamma_overlap": float(template["gamma_overlap"]),
            "rho_init": float(np.clip(
                SELECTED_RHO * float(template["rho_multiplier"]),
                RHO_MIN,
                RHO_MAX,
            )),
            "max_iter": int(template["max_iter"]),
        })

    main_model_dir = cplex_models_dir / "selected_rho_main_run"
    main_model_dir.mkdir(parents=True, exist_ok=True)
    MAX_DCA_ITER = sum(int(stage["max_iter"]) for stage in MAIN_STAGE_SCHEDULE)
    log_msg("=" * 96)
    log_msg(f"MAIN PARAMETER-TUNED RUN | selected rho={SELECTED_RHO}")
    log_msg(f"Main schedule: {MAIN_STAGE_SCHEDULE}")
    DCA_RES = run_baseline_first_pdca(
        z0=Z_SCREEN_BEST,
        db=DB,
        stage_schedule=MAIN_STAGE_SCHEDULE,
        cplex_models_folder=main_model_dir,
    )
    for row in DCA_RES.history:
        enriched = dict(row)
        enriched["parameter_trial"] = "selected_rho_main_run"
        enriched["trial_kind"] = "main_run"
        PARAMETER_HISTORY.append(enriched)

    main_metrics = objective_metrics(DCA_RES.z_star, DB, EVAL_GAMMA_OVERLAP)
    PARAMETER_TRIAL_SUMMARY.append({
        "trial": "selected_rho_main_run",
        "kind": "main_run",
        "rho": SELECTED_RHO,
        "stage_schedule": MAIN_STAGE_SCHEDULE,
        "iterations": DCA_RES.iterations_run,
        "status": DCA_RES.final_status,
        "F_ref": float(main_metrics["eq6_penalized_objective"]),
        "HPWL": float(main_metrics["eq5_hpwl"]),
        "overlap_sum": float(main_metrics["eq8_overlap_hat_sum"]),
    })

    with (config_dir / "selected_parameters.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "selected_rho": SELECTED_RHO,
            "rho_screen_candidates": list(RHO_SCREEN_CANDIDATES),
            "main_stage_schedule": MAIN_STAGE_SCHEDULE,
            "nonmonotone_window": NONMONOTONE_WINDOW,
            "max_relative_objective_increase": MAX_REL_OBJECTIVE_INCREASE,
            "acceleration_beta_max": ACCELERATION_BETA_MAX,
            "active_set_margin": ACTIVE_SET_MARGIN,
        }, handle, indent=2)

    Z_PDCA = DCA_RES.z_star.copy()
    np.save(placements_dir / "z_best_parameter_tuned_pdca_centers.npy", Z_PDCA)
    '''
    set_code(notebook["cells"][13], run_prefix + textwrap.dedent(run_suffix).lstrip("\n"))

    export_source = "".join(notebook["cells"][15]["source"])
    export_source = replace_required(
        export_source,
        'f"{BENCHMARK}.improved_drl_pdca_lns.pl"',
        'f"{BENCHMARK}.parameter_tuned_pdca.pl"',
        "output placement name",
    )
    set_code(notebook["cells"][15], export_source)

    metrics_cell = r'''
    # Cell 16 - Save parameter-tuning metrics and histories

    metrics_start_payload = dict(METRICS_START)
    metrics_start_payload["source"] = START_SOURCE
    with (metrics_dir / "metrics_start.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_start_payload, handle, indent=2)

    METRICS_AFTER = objective_metrics(Z_STAR, DB, EVAL_GAMMA_OVERLAP)
    with (metrics_dir / "metrics_after.json").open("w", encoding="utf-8") as handle:
        json.dump(METRICS_AFTER, handle, indent=2)

    csv_path = metrics_dir / "parameter_tuning_history.csv"
    if PARAMETER_HISTORY:
        fieldnames: List[str] = []
        for row in PARAMETER_HISTORY:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(PARAMETER_HISTORY)
    else:
        csv_path.write_text("", encoding="utf-8")

    trial_summary_path = metrics_dir / "parameter_trial_summary.json"
    with trial_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(PARAMETER_TRIAL_SUMMARY, handle, indent=2)

    lns_history_path = metrics_dir / "topology_lns_history.json"
    with lns_history_path.open("w", encoding="utf-8") as handle:
        json.dump(TOPOLOGY_LNS_HISTORY, handle, indent=2)

    log_msg(f"Saved parameter history: {csv_path}")
    log_msg(f"Saved trial summary: {trial_summary_path}")

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
    '''
    set_code(notebook["cells"][16], metrics_cell)

    target_lines = ""
    if benchmark == "adaptec3":
        target_lines = '''
        axs[0, 0].axhline(62.84, color="black", linestyle="--", label="target 62.84")
        axs[0, 1].axhline(50.0, color="black", linestyle="--", label="target 50")
        axs[1, 0].axhline(256.8, color="black", linestyle="--", label="target budget 256.8")
'''
    plot_cell = f'''
    # Cell 17 - Parameter-trial placement and convergence plots

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    W, H = DB.canvas_width, DB.canvas_height
    widths, heights = macro_geometry(DB)
    ll_before = paper_centers_to_local_lower_left(Z_RUN_START, DB)
    ll_after = paper_centers_to_local_lower_left(Z_STAR, DB)

    for axis, lower_left, metrics, title_text, color in [
        (axes[0], ll_before, METRICS_START, "Selected start incumbent", "cyan"),
        (axes[1], ll_after, METRICS_AFTER, "Parameter-tuned final best", "lightgreen"),
    ]:
        axis.set_xlim(0, W)
        axis.set_ylim(0, H)
        axis.set_aspect("equal")
        axis.set_title(
            f"{{title_text}}\\nHPWL/1e5={{metrics['eq5_hpwl']/1e5:.3f}}, "
            f"F_ref/1e5={{metrics['eq6_penalized_objective']/1e5:.3f}}, "
            f"overlaps={{metrics['eq3_nonoverlap_violating_pairs']}}"
        )
        for i in range(DB.num_macros):
            axis.add_patch(patches.Rectangle(
                (lower_left[i, 0], lower_left[i, 1]), widths[i], heights[i],
                linewidth=0.35, edgecolor="black", facecolor=color, alpha=0.35,
            ))
    plt.tight_layout()
    placement_plot_path = plots_dir / "start_vs_parameter_tuned_final.png"
    plt.savefig(placement_plot_path, dpi=200)
    plt.close()

    if PARAMETER_HISTORY:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        trial_names = list(dict.fromkeys(row["parameter_trial"] for row in PARAMETER_HISTORY))
        for trial_name in trial_names:
            rows = [row for row in PARAMETER_HISTORY if row["parameter_trial"] == trial_name]
            iterations = list(range(1, len(rows) + 1))
            axs[0, 0].plot(iterations, [row["F_ref_after"] / 1e5 for row in rows], label=trial_name)
            axs[0, 1].plot(iterations, [row["HPWL_after"] / 1e5 for row in rows], label=trial_name)
            axs[1, 0].plot(iterations, [row["overlap_hat_sum"] for row in rows], label=trial_name)
            axs[1, 1].plot(iterations, [row["rho"] for row in rows], label=trial_name)
        axs[0, 0].set_title("Reference F / 1e5")
        axs[0, 1].set_title("HPWL / 1e5")
        axs[1, 0].set_title("Overlap penalty sum")
        axs[1, 1].set_title("Rho by parameter trial")
{target_lines.rstrip()}
        for axis in axs.flat:
            axis.set_xlabel("Iteration within trial")
            axis.grid(True)
            axis.legend(fontsize=7)
        plt.tight_layout()
        convergence_plot_path = plots_dir / "parameter_tuning_convergence.png"
        plt.savefig(convergence_plot_path, dpi=200)
        plt.close()

    log_msg("All parameter-tuning plots generated.")
    '''
    set_code(notebook["cells"][17], plot_cell)

    summary_cell = f'''
    # Cell 18 - Final auditable parameter-tuning summary

    log_msg("=" * 96)
    log_msg("PARAMETER-TUNED {benchmark.upper()} EXPERIMENT SUMMARY")
    log_msg("=" * 96)
    log_msg(f"Selected rho                 : {{SELECTED_RHO}}")
    log_msg(f"Rho candidates               : {{RHO_SCREEN_CANDIDATES}}")
    log_msg(f"Acceleration beta cap        : {{ACCELERATION_BETA_MAX}}")
    log_msg(f"Non-monotone window/cap      : {{NONMONOTONE_WINDOW}} / {{MAX_REL_OBJECTIVE_INCREASE:.4%}}")
    log_msg(f"Main PDCA status             : {{DCA_RES.final_status}}")
    log_msg(f"Main PDCA iterations         : {{DCA_RES.iterations_run}}")
    log_msg(f"Start incumbent source       : {{START_SOURCE}}")
    log_msg(f"Start F_ref / 1e5            : {{METRICS_START['eq6_penalized_objective']/1e5:.6f}}")
    log_msg(f"Final F_ref / 1e5            : {{METRICS_AFTER['eq6_penalized_objective']/1e5:.6f}}")
    log_msg(f"Final HPWL / 1e5             : {{METRICS_AFTER['eq5_hpwl']/1e5:.6f}}")
    log_msg(f"Final overlap sum            : {{METRICS_AFTER['eq8_overlap_hat_sum']:.6f}}")
    log_msg(f"Final overlap pairs          : {{METRICS_AFTER['eq3_nonoverlap_violating_pairs']}}")
    log_msg(f"Final boundary violations    : {{METRICS_AFTER['eq4_boundary_violating_macros']}}")
    log_msg(f"Output placement             : {{final_pl_path}}")
    log_msg("=" * 96)

    assert final_pl_path.is_file(), "Final .pl file must exist"
    assert csv_path.is_file(), "Parameter history must exist"
    assert trial_summary_path.is_file(), "Trial summary must exist"
    assert (config_dir / "selected_parameters.json").is_file(), "Selected parameters must exist"
    assert (metrics_dir / "metrics_start.json").is_file(), "Start metrics must exist"
    assert (metrics_dir / "metrics_after.json").is_file(), "Final metrics must exist"
    assert METRICS_AFTER["eq4_boundary_violating_macros"] == 0
    assert (
        METRICS_AFTER["eq6_penalized_objective"]
        <= METRICS_START["eq6_penalized_objective"]
        + descent_tolerance(METRICS_START["eq6_penalized_objective"])
    ), "Best-incumbent safeguard failed"
    print("SUCCESS: full parameter-tuning notebook outputs verified.")
    '''
    set_code(notebook["cells"][18], summary_cell)

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    notebook.setdefault("metadata", {})["parameter_tuned_from"] = source_path.name
    notebook["metadata"]["parameter_tuning_notes"] = (
        "rho screening, benchmark-specific accelerated non-monotone PDCA, "
        "gamma tilt/reference/polish stages, best-incumbent safeguard"
    )
    destination.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


if __name__ == "__main__":
    for benchmark_name in SETTINGS:
        print(build_notebook(benchmark_name))

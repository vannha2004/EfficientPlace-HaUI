"""Update the existing Adaptec4 notebook; preserve its data/model cells."""
import json
import shutil
from datetime import datetime
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'cplex_resolver/adaptec4_drl_pdca_cplex_improved.ipynb'
nb = json.loads(path.read_text(encoding='utf-8'))
if nb.get('metadata', {}).get('rho_search_revision') == 1:
    raise SystemExit('Already updated; no changes made.')
backup = path.with_name(path.stem + '.backup_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.ipynb')
shutil.copy2(path, backup)

def get(i):
    return ''.join(nb['cells'][i]['source'])

def put(i, s):
    nb['cells'][i]['source'] = s.splitlines(keepends=True)

put(0, '''# Adaptec4 — rho schedules and extrapolation search

Full experiment: three independent branches start from the same projected DRL checkpoint.
Gamma overlap stays 5000; rho schedules are 0.01/0.01/0.01,
0.03/0.01/0.003, and 0.01/0.003/0.001 for 50/100/50 iterations.
After each QP, evaluate alpha in {0, 0.25, 0.5, 1, 2} on the full objective.
Prior baseline/improved artifacts are fallback incumbents, not branch starting points.
Each branch saves its history and best placement immediately. Final selection minimizes F;
HPWL is reported separately, and simultaneous improvement is explicitly checked.
Maximum budget: 600 QPs; early convergence advances to the next rho stage.
These are experimental settings, not a guarantee of improved quality.
''')
s = get(1)
a = s.index('# Baseline-matched')
b = s.index('# Active-set', a)
s = s[:a] + '''RHO_BRANCHES = {
    "control": (0.01, 0.01, 0.01),
    "moderate": (0.03, 0.01, 0.003),
    "aggressive": (0.01, 0.003, 0.001),
}
STAGE_ITERATIONS = (50, 100, 50)
ACCELERATION_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)
RHO = 0.01
STAGE_SCHEDULE = []  # built separately for each branch
MAX_DCA_ITER = sum(STAGE_ITERATIONS)
RHO_MIN, RHO_MAX = 0.001, 0.03
RHO_INCREASE, RHO_DECREASE, RHO_MAX_RETRIES = 1.0, 1.0, 0
MIN_STAGE_ITERATIONS = 20
STABLE_ITERATIONS = 8

''' + s[b:]
s = s.replace('REL_STEP_TOL: float = 1e-5', 'REL_STEP_TOL: float = 1e-6')
s = s.replace('EARLY_STOP_RTOL: float = 1e-5', 'EARLY_STOP_RTOL: float = 1e-6')
s = s[:s.index('print("Cell 1:')] + '''print("Adaptec4 full rho-schedule experiment", RHO_BRANCHES)
print("Stage budgets:", STAGE_ITERATIONS, "Acceleration alphas:", ACCELERATION_ALPHAS)
'''
put(1, s)
s = get(2)
a, b = s.index('exp_name ='), s.index('exp_dir =')
s = s[:a] + 'exp_name = "rho_schedules_drl_alpha_search"\n' + s[b:]
s = s.replace('"stage_schedule": STAGE_SCHEDULE,', '"rho_branches": RHO_BRANCHES,\n    "stage_iterations": STAGE_ITERATIONS,\n    "acceleration_alphas": ACCELERATION_ALPHAS,\n    "minimum_stage_iterations": MIN_STAGE_ITERATIONS,\n    "stable_iterations": STABLE_ITERATIONS,')
s = s.replace('"use_controlled_nonmonotone_acceleration": USE_ACCELERATION,', '"use_extrapolation_search": USE_ACCELERATION,')
put(2, s)
s = get(12)
s = s[s.index('@dataclass\nclass DCAResult:'):s.index('def run_baseline_first_pdca(')]
s += '''def run_baseline_first_pdca(z0, db, stage_schedule, cplex_models_folder):
    started = time.time()
    zk = project_to_hard_boundary(z0, db)
    best_z = zk.copy()
    best_f = float(objective_metrics(zk, db, EVAL_GAMMA_OVERLAP)["eq6_penalized_objective"])
    history, stage_statuses = [], []
    failed = False
    for stage_id, stage in enumerate(stage_schedule, 1):
        rho = float(stage["rho_init"])
        assert RHO_MIN <= rho <= RHO_MAX
        assert stage["gamma_overlap"] == EVAL_GAMMA_OVERLAP
        stable = 0
        stage_status = "iteration_budget"
        for local_iter in range(int(stage["max_iter"])):
            tick = time.time()
            before = objective_metrics(zk, db, EVAL_GAMMA_OVERLAP)
            f_before = float(before["eq6_penalized_objective"])
            qp, info = solve_with_active_refinement(
                zk, db, violating_pair_indices(zk, db), EVAL_GAMMA_OVERLAP,
                rho, len(history), cplex_models_folder)
            if qp is None:
                stage_status = "solver_failed:" + info.get("status", "unknown")
                failed = True
                break
            direction = qp - zk
            options = []
            alphas = ACCELERATION_ALPHAS if USE_ACCELERATION else (0.0,)
            for alpha in alphas:
                z = project_to_hard_boundary(qp + alpha * direction, db)
                m = objective_metrics(z, db, EVAL_GAMMA_OVERLAP)
                options.append((float(m["eq6_penalized_objective"]), z, m, float(alpha), "qp" if alpha == 0 else "accelerated"))
            candidate = min(options, key=lambda item: item[0])
            if candidate[0] > f_before:
                for factor in LINE_SEARCH_FACTORS:
                    z = project_to_hard_boundary(zk + factor * direction, db)
                    m = objective_metrics(z, db, EVAL_GAMMA_OVERLAP)
                    options.append((float(m["eq6_penalized_objective"]), z, m, float(factor - 1), "backtracking"))
                candidate = min(options, key=lambda item: item[0])
            if candidate[0] > f_before:
                stage_status = "no_descent_advance_rho"
                break
            f_next, z_next, metrics, alpha, kind = candidate
            step = float(np.linalg.norm(z_next - zk)) / max(sqrt(db.num_macros) * math.hypot(db.canvas_width, db.canvas_height), 1.0)
            rel_improvement = (f_before - f_next) / max(abs(f_before), 1.0)
            stable = stable + 1 if step <= REL_STEP_TOL and rel_improvement <= EARLY_STOP_RTOL else 0
            zk = z_next
            if f_next < best_f:
                best_f, best_z = f_next, zk.copy()
            history.append({
                "global_iteration": len(history) + 1, "stage_id": stage_id,
                "stage_iteration": local_iter + 1, "rho": rho,
                "gamma_overlap_stage": EVAL_GAMMA_OVERLAP,
                "F_stage_before": f_before, "F_ref_after": f_next,
                "HPWL_after": metrics["eq5_hpwl"],
                "overlap_hat_sum": metrics["eq8_overlap_hat_sum"],
                "candidate_kind": kind, "extrapolation_alpha": alpha,
                "relative_step": step, "relative_improvement": rel_improvement,
                "candidate_evaluations": len(options),
                "active_pairs": info.get("active_pairs", 0),
                "cplex_status": info.get("status", "unknown"),
                "total_runtime_sec": time.time() - tick,
            })
            log_msg(f"Iter {len(history)} stage={stage_id} rho={rho} alpha={alpha} F/1e5={f_next/1e5:.6f} HPWL/1e5={metrics['eq5_hpwl']/1e5:.6f}")
            if local_iter + 1 >= MIN_STAGE_ITERATIONS and stable >= STABLE_ITERATIONS:
                stage_status = "joint_convergence"
                break
        stage_statuses.append(stage_status)
        log_msg(f"Stage {stage_id} ended: {stage_status}")
        if failed:
            break
    return DCAResult(best_z, zk, history, ";".join(stage_statuses),
                     time.time() - started,
                     bool(stage_statuses) and all(s == "joint_convergence" for s in stage_statuses),
                     len(history), best_f)
'''
put(12, s)
s = get(13)
a, b = s.index('    baseline_root ='), s.index('    for baseline_path')
s = s[:a] + '''    experiment_root = PROJECT_ROOT / "experiments" / BENCHMARK
    baseline_paths = []
    for family in ("activeset_gb_100_gov_5000_rho_1e-02_iter300_acc_q3",
                   "baseline_first_gov_5000_rho_1e-02_nonmono_q3",
                   "rho_schedules_drl_alpha_search"):
        baseline_paths.extend(sorted((experiment_root / family).glob("run_*/placements/z_star_paper_centers.npy")))
''' + s[b:]
s = s[:s.index('DCA_RES = run_baseline_first_pdca(')] + '''# Every branch starts from DRL; the old incumbent only protects final output.
Z_DRL_START = project_to_hard_boundary(Z0_PAPER_CENTERS, DB)
np.save(placements_dir / "z_branch_start_drl.npy", Z_DRL_START)
ALL_HISTORY, BRANCH_SUMMARY = [], []
Z_PDCA = Z_RUN_START.copy()
winner_f = float(METRICS_START["eq6_penalized_objective"])
WINNER_SOURCE = START_SOURCE
total_time = 0.0
for branch_name, rhos in RHO_BRANCHES.items():
    schedule = [dict(name=f"{branch_name}_{i+1}", gamma_overlap=EVAL_GAMMA_OVERLAP,
                     rho_init=rho, max_iter=budget)
                for i, (rho, budget) in enumerate(zip(rhos, STAGE_ITERATIONS))]
    folder = cplex_models_dir / branch_name
    folder.mkdir(parents=True, exist_ok=True)
    result = run_baseline_first_pdca(Z_DRL_START.copy(), DB, schedule, folder)
    total_time += result.total_time_sec
    metrics = objective_metrics(result.z_star, DB, EVAL_GAMMA_OVERLAP)
    for row in result.history:
        row["branch"] = branch_name
        row["branch_iteration"] = row["global_iteration"]
        row["global_iteration"] = len(ALL_HISTORY) + 1
        ALL_HISTORY.append(row)
    np.save(placements_dir / f"z_{branch_name}_best.npy", result.z_star)
    summary = dict(branch=branch_name, rho_schedule=list(rhos),
                   iterations=result.iterations_run, status=result.final_status,
                   F_ref=metrics["eq6_penalized_objective"], HPWL=metrics["eq5_hpwl"],
                   overlap_sum=metrics["eq8_overlap_hat_sum"],
                   improves_both=bool(metrics["eq6_penalized_objective"] < METRICS_START["eq6_penalized_objective"]
                                      and metrics["eq5_hpwl"] < 6697549.485130764))
    BRANCH_SUMMARY.append(summary)
    if summary["F_ref"] < winner_f:
        winner_f, Z_PDCA, WINNER_SOURCE = summary["F_ref"], result.z_star.copy(), branch_name
    # Durable checkpoints after every branch, including if a later solve fails.
    np.save(placements_dir / "z_star_paper_centers.npy", Z_PDCA)
    (metrics_dir / "branch_summary.json").write_text(json.dumps(BRANCH_SUMMARY, indent=2), encoding="utf-8")
    (metrics_dir / "branch_history.json").write_text(json.dumps(ALL_HISTORY, indent=2), encoding="utf-8")
    log_msg("BRANCH RESULT: " + json.dumps(summary))
DCA_RES = DCAResult(Z_PDCA.copy(), result.z_last, ALL_HISTORY,
                    "branches_finished; see branch_summary.json", total_time,
                    False, len(ALL_HISTORY), winner_f)
log_msg(f"Final incumbent source: {WINNER_SOURCE}")
np.save(placements_dir / "z_best_reference_pdca_centers.npy", Z_PDCA)
'''
put(13, s)
put(16, get(16).replace('baseline_first_pdca_history.csv', 'rho_schedule_history.csv'))
s = get(17)
a,b = s.index('    iterations ='), s.index('    for axis in axs.flat:')
s = s[:a] + '''    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    for branch in RHO_BRANCHES:
        rows = [r for r in DCA_RES.history if r["branch"] == branch]
        x = [r["branch_iteration"] for r in rows]
        for axis, key, scale, title in (
            (axs[0,0], "F_ref_after", 1e5, "F_ref / 1e5"),
            (axs[0,1], "HPWL_after", 1e5, "HPWL / 1e5"),
            (axs[1,0], "overlap_hat_sum", 1, "Overlap sum"),
            (axs[1,1], "rho", 1, "Rho schedule")):
            axis.plot(x, [r[key]/scale for r in rows], label=branch)
            axis.set_title(title)
            axis.legend()
''' + s[b:]
s = s.replace('"Global iteration"', '"Iteration within branch"')
put(17, s)
put(18, get(18) + '\nlog_msg(f"Selected result source: {WINNER_SOURCE}")\nfor branch in BRANCH_SUMMARY:\n    log_msg(json.dumps(branch))\n')
nb.setdefault('metadata', {})['rho_search_revision'] = 1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), f'cell_{i}', 'exec')
        cell['outputs'], cell['execution_count'] = [], None
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Updated:', path)
print('Backup:', backup)

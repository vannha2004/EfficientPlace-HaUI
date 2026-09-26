"""Upgrade the exact user-selected backup notebook, preserving its old contents."""
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'cplex_resolver/adaptec4_drl_pdca_cplex_improved.backup_20260924_200320.ipynb'
nb = json.loads(TARGET.read_text(encoding='utf-8'))
if nb.get('metadata', {}).get('long_search_revision'):
    raise SystemExit('Already upgraded; not overwriting user changes.')

def get(i):
    return ''.join(nb['cells'][i]['source'])

def put(i, source):
    nb['cells'][i]['source'] = source.lstrip('\n').splitlines(keepends=True)

put(0, '''# Adaptec4 — extended rho/gamma search with resumable checkpoints

Run All starts a FULL search: 8 independent trials (DRL and saved incumbent ×
4 parameter schedules), then reference-objective polishing of the best 3 trial results.
Maximum: 2,760 QPs; convergence can reduce this. Expect hours to tens of hours,
depending on CPLEX and hardware. This is an experimental search, not a quality guarantee.

All saved candidates are scored with gamma_overlap=5000, regardless of training gamma.
The final placement minimizes this reference F including the prior incumbent.
A separate placement minimizes HPWL among candidates whose F does not exceed the
starting incumbent. HPWL improvement is not implied by an improvement in F.

Each accepted iteration is checkpointed atomically. To resume, paste the printed
experiment directory into RESUME_RUN_DIR in Cell 1 and Run All. Keep search parameters
unchanged when resuming; incompatible configuration/data are rejected.
The existing input datasets and old experiment results are not modified.
''')
s = get(1)
start, end = s.index('# Baseline-matched'), s.index('BENCHMARK:')
s = s[:start] + '''# FULL search configuration. Budget = 8*300 + 3*120 = 2760 QPs.
RESUME_RUN_DIR = None  # e.g. r"D:\\...\\run_20260924_210000"
SEARCH_SEED = 20260924  # CPLEX seed; no random movement of macros
RHO = 0.01  # used by the initial decomposition audit only
STAGE_BUDGETS = (80, 120, 100)
SEARCH_PROFILES = {
    "control": ((5000., .01), (5000., .01), (5000., .01)),
    "rho_decay": ((5000., .03), (5000., .01), (5000., .003)),
    "hpwl_explore": ((3500., .003), (5000., .001), (5000., .01)),
    "overlap_explore": ((8000., .01), (5000., .003), (5000., .001)),
}
POLISH_TOP_K = 3
POLISH_BUDGETS = (60, 60)
POLISH_RHOS = (.001, .0003)
ACCELERATION_ALPHAS = (0., .25, .5, 1., 2., 4.)
LINE_SEARCH_FACTORS = (.5, .25, .125, .0625, .015625)
ACTIVE_SET_MARGIN = 1.0
ACTIVE_SET_MAX_REFINEMENTS = 1
ACTIVE_VIOLATION_TOL = 1e-9
REL_STEP_TOL = 1e-7
REL_OBJECTIVE_TOL = 1e-7
MIN_STAGE_ITERATIONS = 30
STABLE_ITERATIONS = 10
CPLEX_THREADS = 4
CPLEX_WORKMEM_MB = 8192
CPLEX_SOLVE_TIME_LIMIT = 300  # seconds per QP; audited feasible points may be used
RUN_TOPOLOGY_LNS = False

''' + s[end:]
s = s[:s.index('print("Cell 1:')] + '''print("FULL extended search: 8 trials + up to 3 polish runs; maximum 2760 QPs")
print("Rho/gamma profiles:", SEARCH_PROFILES)
'''
put(1, s)
put(2, r'''
import hashlib
import io

timestamp = time.strftime("%Y%m%d_%H%M%S")
exp_config = dict(version=1, benchmark=BENCHMARK, gamma_boundary=GAMMA_BOUNDARY,
    eval_gamma=EVAL_GAMMA_OVERLAP, profiles=SEARCH_PROFILES, budgets=STAGE_BUDGETS,
    polish_k=POLISH_TOP_K, polish_budgets=POLISH_BUDGETS, polish_rhos=POLISH_RHOS,
    alphas=ACCELERATION_ALPHAS, backtracking=LINE_SEARCH_FACTORS,
    active_margin=ACTIVE_SET_MARGIN, refinements=ACTIVE_SET_MAX_REFINEMENTS,
    relative_step=REL_STEP_TOL, relative_objective=REL_OBJECTIVE_TOL,
    minimum_iterations=MIN_STAGE_ITERATIONS, stable_iterations=STABLE_ITERATIONS,
    threads=CPLEX_THREADS, seed=SEARCH_SEED, qp_time_limit=CPLEX_SOLVE_TIME_LIMIT)
exp_config = json.loads(json.dumps(exp_config))
exp_dir = (Path(RESUME_RUN_DIR) if RESUME_RUN_DIR else
    PROJECT_ROOT / "experiments" / BENCHMARK / "extended_rho_gamma_search" / f"run_{timestamp}_{time.time_ns()%1000000:06d}")
if RESUME_RUN_DIR:
    old_config = json.loads((exp_dir / "config/experiment_config.json").read_text(encoding="utf-8"))
    if old_config != exp_config:
        raise ValueError("Resume configuration differs. Restore previous settings or start a new run.")
config_dir, logs_dir, metrics_dir, placements_dir, plots_dir, cplex_models_dir = [
    exp_dir / name for name in ("config", "logs", "metrics", "placements", "plots", "cplex_models")]
for directory in (config_dir, logs_dir, metrics_dir, placements_dir, plots_dir, cplex_models_dir):
    directory.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "run.log"
def log_msg(message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with log_file_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)
def atomic_npy(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temp, path)
atomic_json(config_dir / "experiment_config.json", exp_config)
log_msg(f"Experiment / RESUME_RUN_DIR: {exp_dir}")
''')
s = get(11).replace('solution = mdl.solve(log_output=False)',
    'mdl.parameters.timelimit = CPLEX_SOLVE_TIME_LIMIT\n    mdl.parameters.randomseed = SEARCH_SEED\n    solution = mdl.solve(log_output=False)')
put(11, s)
# Keep the existing audited full-pair metrics and active-set solver helpers.
s = get(12)
s = s[s.index('def project_to_hard_boundary'):s.index('def run_baseline_first_pdca')]
a, b = s.index('def descent_tolerance'), s.index('def solve_with_active_refinement')
s = s[:a] + 'def descent_tolerance(value):\n    return max(1e-5, 1e-10 * abs(value))\n\n\n' + s[b:]
s += r'''

def score(z):
    return objective_metrics(z, DB, EVAL_GAMMA_OVERLAP)

def eval_stage(metrics, gamma):
    return float(metrics["eq5_hpwl"] + GAMMA_BOUNDARY * metrics["eq7_boundary_penalty_sum"]
                 + gamma * metrics["eq8_overlap_hat_sum"])

def data_fingerprint():
    digest = hashlib.sha256()
    digest.update(np.asarray(Z0_PAPER_CENTERS, dtype=np.float64).tobytes())
    digest.update(str((DB.canvas_width, DB.canvas_height)).encode())
    for macro in DB.macros:
        digest.update(str((macro.name, macro.width, macro.height)).encode())
    for net in DB.nets:
        digest.update(np.asarray(net.macro_ids, dtype=np.int64).tobytes())
        digest.update(b";")
    return digest.hexdigest()

def optimize_trial(name, start, schedule):
    folder = exp_dir / "trials" / name
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "state.npz"
    signature = hashlib.sha256((json.dumps(schedule) + DATA_SIGNATURE).encode()
                               + np.asarray(start, dtype=np.float64).tobytes()).hexdigest()
    z = project_to_hard_boundary(start, DB)
    best = z.copy()
    best_m = score(best)
    dual = Z_RUN_START.copy()
    dual_m = score(dual)
    meta = dict(signature=signature, stage=0, iteration=0, stable=0,
                history=[], statuses=[], done=False)
    if checkpoint.is_file():
        with np.load(checkpoint, allow_pickle=False) as saved:
            meta = json.loads(str(saved["meta"].item()))
            if meta["signature"] != signature:
                raise ValueError(f"Checkpoint data/schedule mismatch: {name}")
            z, best, dual = [saved[key].copy() for key in ("current", "best", "dual")]
        best_m, dual_m = score(best), score(dual)
        log_msg(f"RESUME {name}: stage={meta['stage']} iteration={meta['iteration']} done={meta['done']}")

    def save():
        temp = checkpoint.with_suffix(".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, current=z, best=best, dual=dual,
                                meta=np.asarray(json.dumps(meta, allow_nan=False)))
        os.replace(temp, checkpoint)
        atomic_npy(folder / "best.npy", best)
        atomic_npy(folder / "best_hpwl_under_F_cap.npy", dual)

    for si in range(meta["stage"], len(schedule)):
        gamma, rho, budget = schedule[si]
        first_iteration = meta["iteration"] if si == meta["stage"] else 0
        status = "budget_exhausted"
        for k in range(first_iteration, int(budget)):
            before = score(z)
            f_before = eval_stage(before, gamma)
            tick = time.time()
            qp, info = solve_with_active_refinement(z, DB, violating_pair_indices(z, DB),
                gamma, rho, len(meta["history"]), folder)
            if qp is None:
                # Preserve the current iteration for retry on explicit resume.
                save()
                raise RuntimeError(f"{name}: CPLEX returned no solution ({info['status']}). Resume from {exp_dir}")
            direction = qp - z
            candidates = []
            def consider(point, alpha):
                nonlocal best, best_m, dual, dual_m
                point = project_to_hard_boundary(point, DB)
                m = score(point)
                if not np.isfinite(m["eq6_penalized_objective"]):
                    return
                # Archive even evaluated candidates not selected by the training gamma.
                if m["eq6_penalized_objective"] < best_m["eq6_penalized_objective"]:
                    best, best_m = point.copy(), m
                if m["eq6_penalized_objective"] <= F_CAP and m["eq5_hpwl"] < dual_m["eq5_hpwl"]:
                    dual, dual_m = point.copy(), m
                candidates.append((eval_stage(m, gamma), alpha, point, m))
            for alpha in ACCELERATION_ALPHAS:
                consider(qp + float(alpha) * direction, float(alpha))
            if not candidates or min(item[0] for item in candidates) > f_before:
                for factor in LINE_SEARCH_FACTORS:
                    consider(z + float(factor) * direction, float(factor) - 1.)
            chosen = min(candidates, key=lambda item: item[0]) if candidates else None
            if chosen is None or chosen[0] > f_before:
                status = "no_descent_advance_stage"
                break
            f_next, alpha, z_next, m_next = chosen
            step = float(np.linalg.norm(z_next-z)) / max(
                math.sqrt(DB.num_macros)*math.hypot(DB.canvas_width, DB.canvas_height), 1.)
            rel_f = (f_before-f_next)/max(abs(f_before), 1.)
            stable = meta["stable"] + 1 if step <= REL_STEP_TOL and rel_f <= REL_OBJECTIVE_TOL else 0
            z = z_next
            meta.update(stage=si, iteration=k+1, stable=stable)
            meta["history"].append(dict(trial=name, iteration=len(meta["history"])+1,
                stage=si+1, stage_iteration=k+1, rho=rho, gamma=gamma, alpha=alpha,
                F_ref=float(m_next["eq6_penalized_objective"]),
                HPWL=float(m_next["eq5_hpwl"]), overlap=float(m_next["eq8_overlap_hat_sum"]),
                best_F_ref=float(best_m["eq6_penalized_objective"]),
                relative_step=step, relative_improvement=rel_f,
                candidate_evaluations=len(candidates), solver_status=info["status"],
                seconds=time.time()-tick))
            save()
            log_msg(f"{name} stage={si+1} iter={k+1}/{budget} rho={rho} gamma={gamma} alpha={alpha} F_ref/1e5={m_next['eq6_penalized_objective']/1e5:.6f} best={best_m['eq6_penalized_objective']/1e5:.6f}")
            if k+1 >= MIN_STAGE_ITERATIONS and stable >= STABLE_ITERATIONS:
                status = "joint_convergence"
                break
        meta["statuses"].append(status)
        meta.update(stage=si+1, iteration=0, stable=0)
        # Return from a tilted gamma using its last iterate, not its best reference point.
        save()
        log_msg(f"{name}: stage {si+1} {status}")
    meta["done"] = True
    save()
    return dict(name=name, z=best, metrics=best_m, dual=dual,
                history=meta["history"], statuses=meta["statuses"])
'''
put(12, s)
put(13, r'''
# Load an immutable start snapshot on resume; new runs search prior compatible results.
DATA_SIGNATURE = data_fingerprint()
snapshot = config_dir / "start_snapshot.json"
if snapshot.is_file():
    snapshot_data = json.loads(snapshot.read_text(encoding="utf-8"))
    if snapshot_data["data_signature"] != DATA_SIGNATURE:
        raise ValueError("Dataset/checkpoint changed; cannot resume this experiment.")
    Z_RUN_START = np.load(placements_dir / "z_start_incumbent_centers.npy", allow_pickle=False)
    START_SOURCE = snapshot_data["source"]
else:
    Z_RUN_START = project_to_hard_boundary(Z0_PAPER_CENTERS, DB)
    START_SOURCE = "projected_drl"
    for prior in sorted((PROJECT_ROOT / "experiments" / BENCHMARK).glob("*/run_*/placements/z_star_paper_centers.npy")):
        if prior.parent == placements_dir:
            continue
        try:
            z = np.load(prior, allow_pickle=False)
            if z.shape != Z_RUN_START.shape or not np.isfinite(z).all():
                continue
            z = project_to_hard_boundary(z, DB)
            if score(z)["eq6_penalized_objective"] < score(Z_RUN_START)["eq6_penalized_objective"]:
                Z_RUN_START, START_SOURCE = z.copy(), str(prior)
        except (ValueError, OSError) as exc:
            log_msg(f"Skip {prior}: {exc}")
    atomic_npy(placements_dir / "z_start_incumbent_centers.npy", Z_RUN_START)
    atomic_json(snapshot, dict(source=START_SOURCE, data_signature=DATA_SIGNATURE))
METRICS_START = score(Z_RUN_START)
F_CAP = float(METRICS_START["eq6_penalized_objective"])
atomic_json(metrics_dir / "metrics_start.json", dict(METRICS_START, source=START_SOURCE))
log_msg(f"Incumbent F_ref/1e5={F_CAP/1e5:.6f}; source={START_SOURCE}")
RESULTS = []
Z_PDCA, Z_HPWL = Z_RUN_START.copy(), Z_RUN_START.copy()
WINNER_SOURCE = START_SOURCE

def register_result(result):
    global Z_PDCA, Z_HPWL, WINNER_SOURCE
    RESULTS.append(result)
    if result["metrics"]["eq6_penalized_objective"] < score(Z_PDCA)["eq6_penalized_objective"]:
        Z_PDCA, WINNER_SOURCE = result["z"].copy(), result["name"]
    if score(result["dual"])["eq5_hpwl"] < score(Z_HPWL)["eq5_hpwl"]:
        Z_HPWL = result["dual"].copy()
    atomic_npy(placements_dir / "z_star_paper_centers.npy", Z_PDCA)
    atomic_npy(placements_dir / "z_hpwl_under_F_cap.npy", Z_HPWL)
    summary = [dict(trial=r["name"], F_ref=r["metrics"]["eq6_penalized_objective"],
                    HPWL=r["metrics"]["eq5_hpwl"], overlap=r["metrics"]["eq8_overlap_hat_sum"],
                    iterations=len(r["history"]), statuses=r["statuses"],
                    improves_both=bool(r["metrics"]["eq6_penalized_objective"] < F_CAP
                        and r["metrics"]["eq5_hpwl"] < METRICS_START["eq5_hpwl"])) for r in RESULTS]
    atomic_json(metrics_dir / "trial_summary.json", summary)
    atomic_json(metrics_dir / "metrics_after.json", score(Z_PDCA))

starts = {"incumbent": Z_RUN_START, "drl": project_to_hard_boundary(Z0_PAPER_CENTERS, DB)}
for start_name, start in starts.items():
    for profile, pairs in SEARCH_PROFILES.items():
        schedule = [(float(gamma), float(rho), int(budget))
                    for (gamma, rho), budget in zip(pairs, STAGE_BUDGETS)]
        register_result(optimize_trial(f"{start_name}_{profile}", start.copy(), schedule))

# Freeze ranking before adding polishing results, ensuring stable resume ordering.
ranked = []
for candidate in sorted(RESULTS, key=lambda r: (r["metrics"]["eq6_penalized_objective"], r["name"])):
    if len(ranked) >= POLISH_TOP_K:
        break
    if not any(np.allclose(candidate["z"], old["z"], rtol=0, atol=1e-6) for old in ranked):
        ranked.append(candidate)
for item in ranked:
    schedule = [(EVAL_GAMMA_OVERLAP, rho, budget) for rho, budget in zip(POLISH_RHOS, POLISH_BUDGETS)]
    register_result(optimize_trial("polish_" + item["name"], item["z"].copy(), schedule))
Z_STAR = Z_PDCA.copy()
TOPOLOGY_LNS_HISTORY = []
log_msg(f"Selected best-F placement: {WINNER_SOURCE}")
''')
put(14, '# Search completed in Cell 13. Topology LNS is disabled for this parameter experiment.\n')
s = get(15).replace('improved_drl_pdca_lns.pl', 'extended_search_best_F.pl')
s += '''\nhpwl_pl_path = placements_dir / f"{BENCHMARK}.best_HPWL_under_F_cap.pl"
write_bookshelf_pl_from_paper_centers(Z_HPWL, DB, bench_dir / f"{BENCHMARK}.pl", hpwl_pl_path)
'''
put(15, s)
put(16, r'''
METRICS_AFTER = score(Z_STAR)
METRICS_HPWL = score(Z_HPWL)
atomic_json(metrics_dir / "metrics_after.json", METRICS_AFTER)
atomic_json(metrics_dir / "metrics_hpwl_under_F_cap.json", METRICS_HPWL)
csv_path = metrics_dir / "extended_search_history.csv"
all_history = [row for result in RESULTS for row in result["history"]]
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    if all_history:
        writer = csv.DictWriter(handle, fieldnames=list(all_history[0]))
        writer.writeheader()
        writer.writerows(all_history)
for label, metrics in [("Start incumbent", METRICS_START), ("Best F", METRICS_AFTER),
                       ("Best HPWL with F <= start", METRICS_HPWL)]:
    print(f"{label}: F/1e5={metrics['eq6_penalized_objective']/1e5:.6f}, HPWL/1e5={metrics['eq5_hpwl']/1e5:.6f}, overlap={metrics['eq8_overlap_hat_sum']:.6f}")
''')
put(17, r'''
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for result in RESULTS:
    rows = result["history"]
    axes[0].plot([r["iteration"] for r in rows], [r["best_F_ref"]/1e5 for r in rows], label=result["name"])
    axes[1].scatter(result["metrics"]["eq5_hpwl"]/1e5, result["metrics"]["eq8_overlap_hat_sum"], label=result["name"])
axes[0].axhline(F_CAP/1e5, color="black", linestyle="--", label="prior incumbent")
axes[0].set(xlabel="Iteration within trial", ylabel="Best reference F / 1e5", yscale="log")
axes[1].set(xlabel="HPWL / 1e5", ylabel="Overlap sum")
for axis in axes:
    axis.grid(True)
    axis.legend(fontsize=6)
plt.tight_layout()
plt.savefig(plots_dir / "extended_search_comparison.png", dpi=160)
plt.close(fig)
''')
put(18, r'''
assert METRICS_AFTER["eq6_penalized_objective"] <= F_CAP + descent_tolerance(F_CAP)
assert METRICS_HPWL["eq6_penalized_objective"] <= F_CAP + descent_tolerance(F_CAP)
assert METRICS_HPWL["eq5_hpwl"] <= METRICS_START["eq5_hpwl"] + 1e-5
assert METRICS_AFTER["eq4_boundary_violating_macros"] == 0
assert METRICS_HPWL["eq4_boundary_violating_macros"] == 0
assert final_pl_path.is_file() and hpwl_pl_path.is_file() and csv_path.is_file()
log_msg(f"Completed {len(RESULTS)} trials; accepted iterations={len(all_history)}")
log_msg(f"Best F source: {WINNER_SOURCE}; F/1e5={METRICS_AFTER['eq6_penalized_objective']/1e5:.6f}")
log_msg(f"Best HPWL under starting F cap: {METRICS_HPWL['eq5_hpwl']/1e5:.6f}")
log_msg(f"Best-F PL: {final_pl_path}")
log_msg(f"HPWL-focused PL: {hpwl_pl_path}")
log_msg(f"Results and resume directory: {exp_dir}")
print("DONE: reference-objective and HPWL safeguards verified.")
''')
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), f'cell_{i}', 'exec')
        cell['execution_count'], cell['outputs'] = None, []
nb.setdefault('metadata', {})['long_search_revision'] = 1
nb['metadata']['improvement_notes'] = '8 rho/gamma trials + top-3 polish; resumable; best F and HPWL-cap archive'
backup = TARGET.with_name(TARGET.stem + '.before_long_search_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.ipynb')
shutil.copy2(TARGET, backup)
TARGET.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print('Updated:', TARGET)
print('Preserved:', backup)

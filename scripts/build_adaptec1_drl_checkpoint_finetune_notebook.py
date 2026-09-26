from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "cplex_resolver" / "adaptec1_drl_checkpoint_finetune.ipynb"


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
# adaptec1 — DRL checkpoint audit, warm-start và fine-tune cho CPLEX/LNS-MIP

Mục tiêu duy nhất dùng để chọn placement là:

\[
F(x)=\mathrm{pin\mbox{-}level\ HPWL}(x)/10^5.
\]

Placement chỉ được đưa sang candidate pool khi legalization nhanh cho
`overlap_pairs = 0`, `boundary_violating_macros = 0` và `F(x) < 16.2531734`.
Checkpoint gốc không bị ghi đè; mọi output mới nằm dưới
`cplex_resolver/experiments/adaptec1/drl_checkpoint_finetune/`.

Nguồn chuẩn đã audit là file người dùng cung cấp cho Kaggle
`scriptVersionId=340973617`: `08-08-drl.ipynb`, SHA-256
`9AEA2702CE7B85E053592E6ECFC8B91EB3427458C8390C9041E22C163887F1D1`.
Notebook này không tự động tải hay thay bằng version Kaggle mới hơn.
'''
    ),
    code_cell(
        r'''
# Cell 1 - Locate the exact local repository and load the auditable core
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def locate_core_dir() -> Path:
    cwd = Path.cwd().resolve()
    candidates = []
    for base in [cwd, *cwd.parents]:
        candidates.extend((base / "cplex_resolver", base / "EfficientPlace-HaUI-main" / "cplex_resolver"))
    for candidate in candidates:
        if (candidate / "drl_finetune_core.py").is_file():
            return candidate
    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Cannot find drl_finetune_core.py. Checked:\n{checked}")


CORE_DIR = locate_core_dir()
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from drl_finetune_core import (
    ACCEPT_LEGAL_F,
    TORCH_AVAILABLE,
    TORCH_IMPORT_ERROR,
    TrainingConfig,
    build_db,
    full_metrics,
    grid_xy_to_local_centers,
    inspect_checkpoint_compatibility,
    locate_repo_root,
    run_finetuning,
    run_numpy_smoke,
)

REPO_ROOT = locate_repo_root(CORE_DIR)
CPLEX_ROOT = REPO_ROOT / "cplex_resolver"
DB = build_db(REPO_ROOT)
print("REPO_ROOT:", REPO_ROOT)
print("PyTorch available:", TORCH_AVAILABLE, "" if TORCH_AVAILABLE else TORCH_IMPORT_ERROR)
print("macros:", DB.num_macros, "macro-order SHA256:", DB.macro_order_sha256)
'''
    ),
    markdown_cell(
        r'''
## Audit nguồn Kaggle 340973617 và nguyên nhân checkpoint hiện tại yếu

### Preprocessing và adaptec1

- Cell hiển thị **4** đọc `.nodes/.nets/.pl/.scl`; 543 terminal được coi là macro và tất cả
  đều được RL đặt, kể cả 543 dòng `/FIXED`. Macro giữ đúng thứ tự terminal trong `.nodes`.
- Canvas lấy từ `.scl`: origin `(459,459)`, góc trên `(11151,11139)`.
- Net dùng cho graph/reward chỉ giữ net có ít nhất hai macro. Parser vẫn lưu pin lặp và
  pin offsets, nhưng offsets chỉ dùng khi đánh giá sau train, không dùng trong reward.

### Observation, action, model và training loop

- Grid `32×32`; occupancy một kênh được bilinear-upsample lên `84×84` và nhân 255.
- Action rời rạc 1024 ô. `divmod(action,32)` cho `(x,y)`; tensor occupancy được index
  `[x,y]` (x-major). Ô trùng được dịch sang ô trống gần nhất theo Manhattan shell.
- Policy: CNN `1→32→64→32→496`; GCN `2→16→32→16`; ghép 512 chiều rồi actor/critic.
  Intermediate reward dùng RND.
- Cell hiển thị **12** định nghĩa `train_seed`; cell **13** gọi nó. Seed mặc định `0`
  (`0..4` chỉ chạy khi bật multi-seed), 800 PPO updates, 4 episodes/update,
  543 actions/episode = 2.172 transitions/update. Mỗi update có 4 PPO epochs × 4
  minibatches = 16 optimizer steps; tổng 3.200 episodes và 12.800 PPO minibatch steps.
  RND predictor được update ở 542 bước không-terminal mỗi episode, tức 1.734.400 lần.
  `seed_everything` đặt Python, NumPy, Torch CPU/CUDA và bật cuDNN deterministic.

### Reward, checkpoint, best-placement và export

- Terminal reward thực tế là
  `0.2 * (-Top32_RUDY_mean - 0.1*(grid_source_wirelength - 19264))`.
  Đây là macro-center/grid proxy; không có pin offsets, macro size, physical boundary,
  overlap area hoặc đúng `F(x)`.
- `/kaggle/working/deeppr_table2/adaptec1/seed_0/{best.pt,last.pt}` chứa policy/RND
  weights, PPO/RND optimizer states, update, seed, `best_reward` và `best_grid`.
- `best.pt` được chọn theo terminal proxy reward của sampled episode. Sau train, code nạp
  weights rồi chạy policy deterministic để sinh placement mới; nó không xuất chính
  `best_grid`. `placement_grid.npy`, `.pl`, `metrics.json` vì vậy là deterministic rollout
  của best-weight snapshot, còn pin-HPWL chỉ được đo hậu kiểm.
- Log nguồn cho thấy `best_grid` theo reward chỉ đạt pin-HPWL khoảng `26.774×10^5`, trong
  khi deterministic reroll xuất ra `20.648095×10^5`; đây là bằng chứng hai tiêu chí/placement
  không đồng nhất, không phải bằng chứng reward đã tối ưu đúng `F(x)`.
- Export biến grid thành **lower-left global** theo
  `x=x_min+grid_x/32*canvas_width`, `y=y_min+grid_y/32*canvas_height`, rồi thay toạ độ
  theo tên macro trong `.pl`. Pipeline mới kiểm tra lại phép biến đổi hai chiều này.
- Workspace không có `best.pt/last.pt`; CPLEX local đang đọc đúng
  `cplex_resolver/results/adaptec1/placement_grid.npy`. `config.json` và `metrics.json`
  chỉ là metadata. Do đó không thể giả vờ resume weights Kaggle; notebook dùng placement
  này làm warm-start và chỉ resume weights khi checkpoint mới có payload tương thích.

Các điểm giới hạn trong code local được truy vết cụ thể:

- `src/place_db.py::read_pl_file` suy một canvas vuông bằng `min(max_height,max_width)` và
  không mang origin `.scl`; `rank_macros` lại đổi thứ tự so với artifact Kaggle.
- `src/environment.py::Environment.step/calc_wiremask` dùng pin offsets nhưng làm tròn pin
  về grid; reward không phải pin-HPWL vật lý chính xác. `calc_position_mask` có hard mask
  hữu ích nhưng dựa trên canvas vuông nói trên.
- `src/agent.py::Actor.get_distr` gán `action_distr = greedy_distr`, nên policy chỉ được
  lấy mẫu trong các ô có wire-mask nhỏ nhất; logits học được hầu như chỉ phá hoà. Pipeline
  mới mask tất cả vị trí bất hợp lệ nhưng để actor chọn trong toàn bộ tập vị trí hợp lệ;
  greedy chỉ dùng cho 415 macro đuôi.
- `main.py::Trainer.train/run_episode` chỉ lưu `actor_net.pth`, `critic_net.pth` và pickle
  action; optimizer, scheduler, RNG và solution-pool không được lưu nên không resume đúng.

| Thành phần | Kaggle DRL 340973617 | Code EfficientPlace local | Nhất quán? |
|---|---|---|---|
| Grid size | 32 | 256 | Không |
| Seed | mặc định 0; tuỳ chọn 0..4 | 0 | Một seed trùng |
| Macro ordering | thứ tự terminal trong `.nodes` | rank area, rồi tổng area của net liên quan | Không |
| Coordinate convention | grid `(x,y)` là lower-left | action `//grid=x`, `%grid=y`, lower-left | Có về action |
| Canvas origin | `.scl`, global origin `(459,459)` khi export | code gốc suy square side từ `.pl`, bỏ origin | Không |
| Pin offsets | chỉ dùng hậu kiểm | dùng trong wire mask nhưng lượng tử hoá theo grid | Một phần |
| Reward | grid wirelength + Top-32 RUDY + RND | incremental grid HPWL | Không; cả hai chưa đúng `F(x)` tuyệt đối |
| Best-checkpoint criterion | sampled proxy reward; export deterministic reroll | solution-pool approximate HPWL | Không |
| Checkpoint format | weights + optimizer + `best_grid` | actor/critic weights riêng + placement pickle; không optimizer | Không |

Các giới hạn được sửa trong pipeline mới: pin-HPWL vật lý dùng offsets ngay trong reward;
canvas chữ nhật và origin từ `.scl`; action mask dùng kích thước macro cho cả boundary và
overlap; mapping theo tên khi đổi macro order; reward/advantage normalization; penalty
continuation; curriculum 128→256; behavior-cloning warm start từ placement grid-32;
checkpoint đầy đủ; top-K đa dạng; và lựa chọn theo `F(x)` sau legalization nhanh.
'''
    ),
    code_cell(
        r'''
# Cell 2 - Trace the exact local Kaggle artifact and reproduce its metrics
DRL_RESULTS = CPLEX_ROOT / "results" / "adaptec1"
BASELINE_GRID_PATH = DRL_RESULTS / "placement_grid.npy"
BASELINE_CONFIG_PATH = DRL_RESULTS / "config.json"
BASELINE_METRICS_PATH = DRL_RESULTS / "metrics.json"

baseline_grid = np.load(BASELINE_GRID_PATH)
baseline_centers = grid_xy_to_local_centers(baseline_grid, DB, 32)
baseline_metrics_recomputed = full_metrics(baseline_centers, DB)
baseline_config = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
baseline_metrics_saved = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))

binary_checkpoints = sorted(
    [*REPO_ROOT.rglob("best.pt"), *REPO_ROOT.rglob("last.pt"), *REPO_ROOT.rglob("*.pth")]
)
print("Kaggle-derived file used by CPLEX:", BASELINE_GRID_PATH)
print("Binary checkpoints present:", [str(path) for path in binary_checkpoints])
print("Saved F(x):", baseline_metrics_saved["hpwl_pin"] / 1e5)
print("Recomputed full_metrics:", json.dumps(baseline_metrics_recomputed, indent=2))
assert abs(baseline_metrics_recomputed["F_x"] - 20.6480950) < 1e-9
assert baseline_metrics_recomputed["overlap_pairs"] == 1086
assert baseline_metrics_recomputed["boundary_violating_macros"] == 5
'''
    ),
    code_cell(
        r'''
# Cell 3 - RAM-bounded NumPy smoke test and .pl coordinate round-trip
SMOKE = run_numpy_smoke(REPO_ROOT)
print(json.dumps(SMOKE, indent=2))
'''
    ),
    code_cell(
        r'''
# Cell 4 - Fine-tuning configuration; seeds run strictly one after another
RUN_TRAINING = os.getenv("EFFICIENTPLACE_RUN_FINETUNE", "0").lower() in {"1", "true", "yes", "on"}

CONFIG = TrainingConfig(
    seeds=(0, 1, 2, 3, 4),
    curriculum=((128, 100), (256, 400)),
    num_macros_to_place=128,
    episodes_per_loop=5,
    ppo_epochs=10,
    batch_size=128,
    behavior_clone_epochs=3,
    checkpoint_interval=25,
    top_k=5,
    early_stopping_patience=100,
    resume=True,
    accept_legal_F=ACCEPT_LEGAL_F,
)

# Optional exact-weight resume path. The current workspace intentionally has no Kaggle best.pt.
KAGGLE_WEIGHT_CHECKPOINT = DRL_RESULTS / "best.pt"
compatibility = inspect_checkpoint_compatibility(KAGGLE_WEIGHT_CHECKPOINT, DB, CONFIG)
print("Weight checkpoint compatibility:", json.dumps(compatibility, indent=2))
print("RUN_TRAINING:", RUN_TRAINING)
print(json.dumps(CONFIG.__dict__, indent=2, default=list))
'''
    ),
    code_cell(
        r'''
# Cell 5 - Resume/fine-tune and emit periodic full-state checkpoints + top-K placements
FINETUNE_RESULT = None
if RUN_TRAINING:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "RUN_TRAINING=True but this kernel has no PyTorch training stack: "
            f"{TORCH_IMPORT_ERROR}"
        )
    FINETUNE_RESULT = run_finetuning(REPO_ROOT, CONFIG)
    print(json.dumps(FINETUNE_RESULT, indent=2))
else:
    print(
        "Training is disabled for this kernel. Set EFFICIENTPLACE_RUN_FINETUNE=1 "
        "in a PyTorch EfficientPlace environment. NumPy smoke/audit above still ran."
    )
'''
    ),
    code_cell(
        r'''
# Cell 6 - Acceptance report; never claim an unmeasured improvement
FINETUNE_ROOT = (
    CPLEX_ROOT / "experiments" / "adaptec1" / "drl_checkpoint_finetune"
)
records = []
for metrics_path in sorted(FINETUNE_ROOT.glob("seed_*/top_k/rank_*/metrics.json")):
    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    record["metrics_path"] = str(metrics_path)
    records.append(record)

accepted = [record for record in records if record.get("accepted_for_cplex", False)]
accepted.sort(key=lambda record: record["legalized_metrics"]["F_x"])

print("| Candidate | F(x) raw | Overlap | Boundary | F(x) sau legalize | F(x) sau LNS |")
print("|---|---:|---:|---:|---:|---:|")
print("| DRL cũ | 20.6480950 | 1086 | 5 | 16.2531734 | 14.4531515 |")
if accepted:
    best = accepted[0]
    raw = best["raw_metrics"]
    legal = best["legalized_metrics"]
    print(
        f"| DRL fine-tune tốt nhất | {raw['F_x']:.7f} | {raw['overlap_pairs']} | "
        f"{raw['boundary_violating_macros']} | {legal['F_x']:.7f} | chưa chạy quality-max |"
    )
    print("Accepted candidate:", best["metrics_path"])
else:
    print("| DRL fine-tune tốt nhất | chưa có số liệu | — | — | — | — |")
    print("Không có checkpoint mới nào đạt tiêu chí; candidate pool sẽ không nhận warm smoke artifact.")
'''
    ),
    markdown_cell(
        r'''
## Quy tắc sử dụng với quality-max

Notebook `adaptec1_quality_max_hybrid.ipynb` quét `seed_*/top_k/rank_*/metrics.json`
và chỉ thêm candidate có `accepted_for_cplex=true`. Full-pair audit dùng NumPy theo chunk;
CPLEX tiếp tục chỉ dựng active-set/critical-pair constraints, không dựng
`all pairs × 4 directions`.

Sau khi có accepted fine-tune candidate, chạy quality-max để lấy số `F(x) sau LNS`.
Chỉ coi checkpoint mới vượt baseline khi output cuối vẫn có overlap/boundary bằng 0 và
`F(x) < 14.4531515`; nếu không, giữ nguyên checkpoint grid-32 và các run quality-max cũ.
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
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

DESTINATION.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(DESTINATION)

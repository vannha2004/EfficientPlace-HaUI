from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cplex_resolver" / "adaptec3_drl_pdca_cplex_improved.ipynb"
DESTINATION = ROOT / "cplex_resolver" / "adaptec4_drl_pdca_cplex_improved.ipynb"


def replace_source(cell: dict, old: str, new: str) -> None:
    source = "".join(cell.get("source", []))
    source = source.replace(old, new)
    cell["source"] = source.splitlines(keepends=True)


with SOURCE.open("r", encoding="utf-8") as handle:
    notebook = json.load(handle)

notebook = copy.deepcopy(notebook)

# Benchmark-wide labels and literal paths.
for cell in notebook["cells"]:
    if "source" not in cell:
        continue
    replace_source(cell, "ADAPTEC3", "ADAPTEC4")
    replace_source(cell, "Adaptec3", "Adaptec4")
    replace_source(cell, "adaptec3", "adaptec4")
    replace_source(cell, "LNS-MIQP", "LNS-MILP")

# Adaptec4 has 1,329 macros and grid_size=40. Keep the inherited one-QP policy
# and full-pair audits after every candidate.
config_cell = notebook["cells"][1]
replace_source(config_cell, "TOPOLOGY_LNS_CRITICAL_PAIRS: int = 64", "TOPOLOGY_LNS_CRITICAL_PAIRS: int = 48")
replace_source(config_cell, "GRID_SIZE: int = 32", "GRID_SIZE: int = 40")

# Always route the DRL artifact through BENCHMARK rather than another literal.
validation_cell = notebook["cells"][3]
replace_source(
    validation_cell,
    'drl_results_dir = PROJECT_ROOT / "results" / "adaptec4"',
    'drl_results_dir = PROJECT_ROOT / "results" / BENCHMARK',
)

# Remove an unused duplicate dataclass inherited from the original notebook.
database_cell = notebook["cells"][4]
replace_source(
    database_cell,
    """@dataclass
class Macro:
    name: str
    width: float
    height: float
    is_macro: bool
    global_id: int

""",
    "",
)

# The three horizontal targets belonged only to the Adaptec3 experiment.
plot_cell = notebook["cells"][17]
for line in (
    '    axs[0, 0].axhline(62.84, color="black", linestyle="--", label="target 62.84")\n',
    '    axs[0, 0].legend()\n',
    '    axs[0, 1].axhline(50.0, color="black", linestyle="--", label="target 50")\n',
    '    axs[0, 1].legend()\n',
    '    axs[1, 0].axhline(256.8, color="black", linestyle="--", label="target budget 256.8")\n',
    '    axs[1, 0].legend()\n',
):
    replace_source(plot_cell, line, "")

# Add an explicit scale warning to the introduction.
intro = "".join(notebook["cells"][0]["source"])
intro += (
    "\nAdaptec4 có 1.329 macro (882.456 cặp). Bản này dùng đúng một "
    "active-set QP mỗi vòng, carry các cặp vi phạm mới sang vòng sau và giữ topology LNS tắt trong "
    "lần chạy PDCA đầu tiên.\n"
)
notebook["cells"][0]["source"] = intro.splitlines(keepends=True)

# Clear every old execution artifact and mark provenance accurately.
for cell in notebook["cells"]:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook.setdefault("metadata", {})["improved_from"] = "adaptec4_drl_pdca_cplex.ipynb"
notebook["metadata"]["benchmark"] = "adaptec4"
notebook["metadata"]["adaptation_notes"] = (
    "grid_size 40, dynamic results path, one active-set QP, 48 optional LNS pairs, "
    "Adaptec3-only chart targets removed"
)

with DESTINATION.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(notebook, handle, indent=1, ensure_ascii=False)
    handle.write("\n")

print(DESTINATION)

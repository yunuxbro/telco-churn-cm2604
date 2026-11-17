"""
Build report/appendix_source_code.md: every line of project source code as TEXT.

The brief says the appendix must contain all source code as text (screenshots are
rejected), so re-run this after any code change:   python tools/make_appendix.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "appendix_source_code.md"

parts = ["# Appendix A – Source Code\n",
         "Repository: https://github.com/yunuxbro/telco-churn-cm2604\n"]

for py in [ROOT / "src" / "data_prep.py", ROOT / "tools" / "make_appendix.py"]:
    parts.append(f"\n## {py.relative_to(ROOT).as_posix()}\n\n```python\n{py.read_text(encoding='utf-8').rstrip()}\n```\n")

for nb_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    parts.append(f"\n## {nb_path.relative_to(ROOT).as_posix()} (code cells)\n")
    n = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            n += 1
            src = "".join(cell["source"]).rstrip()
            parts.append(f"\n**Cell {n}**\n\n```python\n{src}\n```\n")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text("".join(parts), encoding="utf-8")
print("Wrote", OUT)

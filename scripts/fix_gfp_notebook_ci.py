#!/usr/bin/env python3
"""Fix gfp_molecular_forge.ipynb: add pio.renderers.default='notebook' for non-Colab/CI envs."""
import json
from pathlib import Path

NOTEBOOKS = [
    Path(__file__).parent.parent / 'examples/interactive_tutorials/gfp_molecular_forge.ipynb',
    Path(__file__).parent.parent / 'docs/tutorials/gfp_molecular_forge.ipynb',
]

FIX = "    import plotly.io as pio\n    pio.renderers.default = 'notebook'  # Prevent browser-open hang in CI\n"

for nb_path in NOTEBOOKS:
    if not nb_path.exists():
        print(f"Skip (not found): {nb_path}")
        continue

    nb = json.loads(nb_path.read_text(encoding='utf-8'))

    # Find and patch the environment-detection cell (Cell 1)
    patched = False
    for cell in nb['cells']:
        src = ''.join(cell['source'])
        if "IN_COLAB = 'google.colab' in sys.modules" in src and "pio.renderers.default = 'notebook'" not in src:
            # Add renderer setting in the else branch
            new_src = src.replace(
                "else:\n    print('💻 Running in local Jupyter environment')\n    sys.path.append(os.path.abspath('../../'))",
                "else:\n    print('💻 Running in local Jupyter / CI environment')\n    sys.path.append(os.path.abspath('../../'))\n" + FIX
            )
            if new_src == src:
                print(f"WARN: pattern not matched in {nb_path.name}, trying alternate")
                # Try to insert it at the end of the else block
                lines = src.split('\n')
                new_lines = []
                in_else = False
                for line in lines:
                    new_lines.append(line)
                    if line.strip().startswith("else:"):
                        in_else = True
                    if in_else and "sys.path.append" in line:
                        new_lines.append("    import plotly.io as pio")
                        new_lines.append("    pio.renderers.default = 'notebook'  # Prevent browser-open hang in CI")
                        in_else = False
                new_src = '\n'.join(new_lines)
            cell['source'] = [new_src]
            patched = True
            break

    if patched:
        nb_path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
        print(f"Fixed: {nb_path.name}")
    else:
        print(f"Already fixed or pattern not found: {nb_path.name}")

# Verify
for nb_path in NOTEBOOKS:
    if nb_path.exists():
        content = nb_path.read_text(encoding='utf-8')
        has_fix = "pio.renderers.default = 'notebook'" in content
        print(f"  {nb_path.name}: renderer fix={'PRESENT' if has_fix else 'MISSING'}")

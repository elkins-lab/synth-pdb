import json

with open("examples/interactive_tutorials/idp_ensemble_validation.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "view_single = py3Dmol.view" in src:
            cell["source"] = [
                "view = py3Dmol.view(width=800, height=300, viewergrid=(1,2), linked=False)\n",
                "view.addModel(single_pdb, 'pdb', viewer=(0,0))\n",
                "view.setStyle({'line': {'color': 'blue', 'linewidth': 3}}, viewer=(0,0))\n",
                "view.zoomTo(viewer=(0,0))\n",
                "\n",
                "for pdb_data in ensemble_pdbs[:20]: # Show first 20 subsets for visual clarity\n",
                "    view.addModel(pdb_data, 'pdb', viewer=(0,1))\n",
                "view.setStyle({'line': {'color': 'spectrum', 'linewidth': 2}}, viewer=(0,1))\n",
                "view.zoomTo(viewer=(0,1))\n",
                "\n",
                "print(\"Single Structure (Left) vs. Dynamic Ensemble (Right)\")\n"
            ]
        elif "out1 = widgets.Output()" in src:
            cell["source"] = [
                "# Render interactive widgets globally to avoid Jupyter context stripping\n",
                "view.show()\n"
            ]

with open("examples/interactive_tutorials/idp_ensemble_validation.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("IDP Notebook patched successfully")

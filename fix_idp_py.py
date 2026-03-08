with open("examples/interactive_tutorials/idp_ensemble_validation.py", "r") as f:
    src = f.read()

old_str = """view_single = py3Dmol.view(width=400, height=300)
view_single.addModel(single_pdb, 'pdb')
view_single.setStyle({'line': {'color': 'blue', 'linewidth': 3}})
view_single.zoomTo()

view_ensemble = py3Dmol.view(width=400, height=300)
for pdb_data in ensemble_pdbs[:20]: # Show first 20 subsets for visual clarity
    view_ensemble.addModel(pdb_data, 'pdb')
view_ensemble.setStyle({'line': {'color': 'spectrum', 'linewidth': 2}})
view_ensemble.zoomTo()

print("Single Structure (Left) vs. Dynamic Ensemble (Right)")"""

new_str = """view = py3Dmol.view(width=800, height=300, viewergrid=(1,2), linked=False)
view.addModel(single_pdb, 'pdb', viewer=(0,0))
view.setStyle({'line': {'color': 'blue', 'linewidth': 3}}, viewer=(0,0))
view.zoomTo(viewer=(0,0))

for pdb_data in ensemble_pdbs[:20]: # Show first 20 subsets for visual clarity
    view.addModel(pdb_data, 'pdb', viewer=(0,1))
view.setStyle({'line': {'color': 'spectrum', 'linewidth': 2}}, viewer=(0,1))
view.zoomTo(viewer=(0,1))

print("Single Structure (Left) vs. Dynamic Ensemble (Right)")"""

src = src.replace(old_str, new_str)

old_out = """out1 = widgets.Output()
out2 = widgets.Output()
with out1: view_single.show()
with out2: view_ensemble.show()
display(widgets.HBox([out1, out2]))"""

new_out = """# Render interactive widgets globally to avoid Jupyter context stripping
view.show()"""

src = src.replace(old_out, new_out)

with open("examples/interactive_tutorials/idp_ensemble_validation.py", "w") as f:
    f.write(src)
print("Updated .py correctly.")

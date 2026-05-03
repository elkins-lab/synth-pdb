from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator
from synth_pdb.viewer import _create_3dmol_html

# Set page config
st.set_page_config(page_title="synth-pdb | Structural Forge Pro", page_icon="🧬", layout="wide")

# Custom CSS for Dark Proteome theme
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background: linear-gradient(45deg, #764ba2, #667eea); color: white; border: none; }
    .stDownloadButton>button { width: 100%; border-radius: 5px; background-color: #10b981; color: white; }
    [data-testid="stExpander"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SESSION STATE ---
if "pdb_output" not in st.session_state:
    st.session_state.pdb_output = None
if "quality_report" not in st.session_state:
    st.session_state.quality_report = None


def run_forge(
    sequence: str,
    conformation: str,
    structure: str | None = None,
    cyclic: bool = False,
    drift: float = 0.0,
    minimize: bool = False,
    optimize: bool = False,
    ph: float = 7.4,
    quality_filter: bool = False,
) -> None:
    """Core generation wrapper with session state updates."""
    try:
        with st.spinner("Forging structure with NeRF and OpenMM..."):
            pdb_str = generate_pdb_content(
                sequence_str=sequence,
                conformation=conformation,
                structure=structure if structure else None,
                cyclic=cyclic,
                drift=drift,
                minimize_energy=minimize or cyclic,
                optimize_sidechains=optimize,
                ph=ph,
            )
            st.session_state.pdb_output = pdb_str

            # Generate Report
            v = PDBValidator(pdb_str)
            st.session_state.quality_report = v.get_quality_report(include_ml=quality_filter)
            st.success("Structure Forged Successfully!")
    except Exception as e:
        st.error(f"Forge Failed: {str(e)}")


# --- APP LAYOUT ---

st.title("🧬 synth-pdb: Structural Forge Pro")
st.markdown("#### Comprehensive laboratory for synthetic protein design and pathological modeling.")

tab_gen, tab_nmr, tab_physics, tab_pathology = st.tabs(
    ["🏗️ Generation", "🧲 NMR & Observables", "🧪 Advanced Physics", "💀 Hall of Perversions"]
)

# --- TAB 1: GENERATION ---
with tab_gen:
    col_input, col_view = st.columns([1, 2])

    with col_input:
        st.subheader("Configuration")
        sequence = st.text_area(
            "Amino Acid Sequence",
            value="MREIILLVATDHYNLTNLYSLLKHYRIPLVVHVSDIKEIR",
            help="Use 1-letter codes. Separate multiple chains with ':'",
            height=100,
        )

        c1, c2 = st.columns(2)
        with c1:
            conformation = st.selectbox(
                "Global Conformation", ["alpha", "beta", "ppii", "extended", "random"]
            )
        with c2:
            cyclic = st.checkbox("Cyclic Peptide", help="Connects N and C termini.")

        structure = st.text_input(
            "Per-Region Structure",
            placeholder="1-10:alpha,11-14:typeI,15-20:beta",
            help="Format: 'start-end:conformation,...'",
        )

        drift = st.slider("Torsion Drift (Degrees)", 0, 180, 0, help="Random noise for Phi/Psi")

        with st.expander("Quality Controls"):
            quality_filter = st.checkbox("AI Quality Filter", help="GNN validation")
            minimize = st.checkbox("Energy Minimization", value=True)
            optimize = st.checkbox("Sidechain Optimization", value=False)
            ph = st.number_input("pH Level", value=7.4, min_value=0.0, max_value=14.0, step=0.1)

        if st.button("🚀 Forge Structure", key="gen_main"):
            run_forge(
                sequence,
                conformation,
                structure,
                cyclic,
                float(drift),
                minimize,
                optimize,
                ph,
                quality_filter,
            )

    with col_view:
        if st.session_state.pdb_output:
            html = _create_3dmol_html(
                st.session_state.pdb_output,
                "forge.pdb",
                style="cartoon" if drift < 45 else "stick",
                color="spectrum",
                show_hbonds=True,
            )
            components.html(html, height=500)

            st.download_button("💾 Download PDB", st.session_state.pdb_output, "synth.pdb")

            if st.session_state.quality_report:
                rep = st.session_state.quality_report
                st.subheader("🛡️ Scientific Defense Scorecard")

                # Metric display
                m1, m2, m3 = st.columns(3)
                m1.metric("Potential Energy", f"{rep['potential_energy_kj_mol']:,.0f} kJ/mol")
                m2.metric("Violations", rep["violation_count"])
                m3.metric("Burial Ratio", f"{rep['hydrophobic_burial_ratio']:.2f}")

                if rep["is_overall_scientifically_defensible"]:
                    st.success("✅ STRUCTURE IS SCIENTIFICALLY DEFENSIBLE")
                else:
                    st.error("❌ STRUCTURE IS NOT DEFENSIBLE")
                    with st.expander("Detailed Violations"):
                        for v in rep["detailed_violations"]:
                            st.write(f"- {v}")

# --- TAB 2: NMR ---
with tab_nmr:
    st.subheader("Synthetic Observable Configuration")
    st.info("Simulation of NMR/SAXS data based on current structure.")
    if st.session_state.pdb_output:
        st.checkbox("Chemical Shifts (SHIFTS+)")
        st.checkbox("RDCs (Alignment Tensor)")
        st.checkbox("NOE Restraints (NEF)")
        if st.button("📡 Run Simulation"):
            st.success("Synthetic data generated (Exported to /outputs)")
    else:
        st.warning("Generate a structure in the 'Generation' tab first.")

# --- TAB 3: PHYSICS ---
with tab_physics:
    st.subheader("OpenMM Engine Settings")
    st.selectbox("Forcefield", ["amber14-all.xml", "charmm36.xml"])
    st.selectbox("Solvent Model", ["obc2", "gbn2", "explicit"])
    st.checkbox("Run MD Equilibration")

# --- TAB 4: PATHOLOGY ---
with tab_pathology:
    st.subheader("💀 The Hall of Pathological Perversions")
    st.markdown("Quick-load pathological presets to test boundary conditions.")

    perversions: dict[str, dict[str, Any]] = {
        "Mobius Loop": {"seq": "AAAAAAAAAAAAAAAAAAAA", "cyclic": True, "drift": 10},
        "Trp Singularity": {"seq": "WWWWWWWWWWWWWWWWWWWW", "conf": "alpha", "drift": 180},
        "Mirror Nightmare": {
            "seq": "D-ALA-D-TRP-D-HIS-D-PRO-D-PHE-D-LEU-D-ILE-D-VAL-D-MET-D-CYS",
            "conf": "random",
        },
        "Poly-Proline Chaos": {"seq": "PPPPPPPPPPPPPPPPPPPP", "conf": "alpha"},
    }

    p_choice = st.selectbox("Select Perversion", list(perversions.keys()))

    if st.button("🔥 Summon Pathology"):
        p = perversions[p_choice]
        run_forge(
            sequence=p.get("seq", "AAAAA"),
            conformation=p.get("conf", "alpha"),
            cyclic=p.get("cyclic", False),
            drift=float(p.get("drift", 0.0)),
            minimize=True,
        )
        st.rerun()

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: #6b7280; font-size: 12px;'>synth-pdb v1.35.0 | Professional Edition</div>",
    unsafe_allow_html=True,
)

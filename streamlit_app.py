import streamlit as st
import streamlit.components.v1 as components

from synth_pdb.generator import generate_pdb_content
from synth_pdb.viewer import _create_3dmol_html

# Set page config for a professional look
st.set_page_config(
    page_title="synth-pdb | Pathological Protein Designer", page_icon="🧬", layout="wide"
)

# Custom CSS to match the "Dark Proteome" theme
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #764ba2;
        color: white;
    }
    .stDownloadButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #10b981;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧬 synth-pdb: The Hall of Pathological Perversions")
st.markdown("### Generate, visualize, and corrupt synthetic protein structures in real-time.")

# Sidebar for Configuration
with st.sidebar:
    st.header("🛠️ Generation Parameters")

    sequence = st.text_area(
        "Amino Acid Sequence", value="MREIILLVATDHYNLTNLYSLLKHYRIPLVVHVSDIKEIR", height=100
    )

    col1, col2 = st.columns(2)
    with col1:
        conformation = st.selectbox(
            "Conformation", ["alpha", "beta", "ppii", "extended", "random"], index=0
        )
    with col2:
        cyclic = st.checkbox("Cyclic Peptide", value=False)

    drift = st.slider(
        "Torsion Drift (Degrees)",
        0,
        180,
        0,
        help="Adds random noise to Phi/Psi angles. 180 = Total Chaos.",
    )

    st.divider()
    st.header("🔋 Physics & Refinement")
    minimize = st.checkbox("Energy Minimization (OpenMM)", value=True)
    optimize = st.checkbox("Sidechain Optimization", value=False)
    ph = st.number_input("pH Level", value=7.4, min_value=0.0, max_value=14.0, step=0.1)

    st.divider()
    generate_btn = st.button("🚀 Generate Structure")

# Main Display Area
if generate_btn:
    try:
        with st.spinner("Executing NeRF geometry and physics-based refinement..."):
            pdb_str = generate_pdb_content(
                sequence_str=sequence,
                conformation=conformation,
                drift=drift,
                minimize_energy=minimize,
                optimize_sidechains=optimize,
                cyclic=cyclic,
                ph=ph,
            )

        # Success message and stats
        st.success("Structure Generated Successfully!")

        # Create columns for the viewer and actions
        view_col, data_col = st.columns([3, 1])

        with view_col:
            st.markdown("#### 🧊 3D Interactive View")
            # Reuse our existing 3Dmol.js integration
            html_content = _create_3dmol_html(
                pdb_str,
                "synthetic.pdb",
                style="cartoon" if not drift > 45 else "stick",
                color="spectrum",
                show_hbonds=True,
            )
            components.html(html_content, height=600, scrolling=False)

        with data_col:
            st.markdown("#### 📥 Actions")
            st.download_button(
                label="💾 Download PDB",
                data=pdb_str,
                file_name="synthetic_perversion.pdb",
                mime="text/plain",
            )

            st.markdown("#### 📝 Sequence Metadata")
            st.info(
                f"**Length:** {len(sequence)} residues\n\n**Mode:** {conformation}\n\n**Drift:** {drift}°"
            )

            if drift > 90:
                st.warning("⚠️ High drift detected. Expect significant steric violations.")
            if cyclic and not minimize:
                st.error("❗ Cyclic peptides require minimization for ring closure.")

    except Exception as e:
        st.error(f"Generation Failed: {str(e)}")
        st.exception(e)

else:
    # Initial landing state
    st.info("👈 Adjust parameters in the sidebar and click 'Generate Structure' to begin.")

    # Showcase a few "Dark Proteome" facts
    col1, col2, col3 = st.columns(3)
    col1.metric("Atomic Density", "Variable", "Pathological")
    col2.metric("Chirality", "L-Isomer", "Mirror-ready")
    col3.metric("Entropy", "Low", "Increasing")

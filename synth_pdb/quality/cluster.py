"""
Protein structure clustering and representative selection.

This module provides tools for grouping protein conformations into clusters
based on their structural similarity (RMSD) and selecting representative
'medoid' structures for each group.

SCIENTIFIC RATIONALE:
--------------------
Structural ensembles (e.g., from NMR or MD simulations) often contain
redundant conformations. Clustering allows for:
1. Identifying distinct conformational states.
2. Reducing ensemble size while maintaining diversity.
3. Quantifying conformational landscape occupancy.

We use K-Means clustering on superimposed C-alpha coordinates, which is a
standard and computationally efficient proxy for full-atom RMSD clustering.
"""

import glob
import logging
import os
import shutil

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np

logger = logging.getLogger(__name__)


def cluster_structures(
    input_pattern: str, n_clusters: int, output_dir: str, random_seed: int = 42
) -> None:
    """Clusters PDB structures and saves the medoid for each cluster.

    Args:
        input_pattern: Glob pattern for input PDB files (e.g., "decoys/*.pdb").
        n_clusters: Number of clusters to form.
        output_dir: Directory to save representative medoid structures.
        random_seed: Seed for K-Means initialisation.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        logger.error("Clustering requires scikit-learn. Install with `pip install synth-pdb[ai]`.")
        return

    # 1. Discover files
    file_list = sorted(glob.glob(input_pattern))
    if not file_list:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return

    if len(file_list) < n_clusters:
        logger.warning(
            f"Fewer structures ({len(file_list)}) than requested clusters ({n_clusters}). "
            "Reducing n_clusters to match file count."
        )
        n_clusters = len(file_list)

    # 2. Structural Alignment and Feature Extraction
    # We use the first structure as the alignment reference.
    try:
        ref_file = pdb.PDBFile.read(file_list[0])
        ref_struct = ref_file.get_structure(model=1)
        ref_ca = ref_struct[ref_struct.atom_name == "CA"]
    except Exception as e:
        logger.error(f"Failed to read reference structure {file_list[0]}: {e}")
        return

    feature_vectors = []
    valid_files = []

    logger.info(f"Extracting features from {len(file_list)} structures...")
    for f in file_list:
        try:
            s = pdb.PDBFile.read(f).get_structure(model=1)
            ca = s[s.atom_name == "CA"]

            if len(ca) != len(ref_ca):
                logger.warning(f"Skipping {f}: Residue count mismatch ({len(ca)} vs {len(ref_ca)})")
                continue

            # Superimpose CA to reference
            fitted, _ = struc.superimpose(ref_ca, ca)
            # Use flattened XYZ coordinates as feature vector
            feature_vectors.append(fitted.coord.flatten())
            valid_files.append(f)
        except Exception as e:
            logger.warning(f"Error processing {f}: {e}")
            continue

    if not feature_vectors:
        logger.error("No valid structures found for clustering.")
        return

    x_features = np.array(feature_vectors)

    # 3. K-Means Clustering
    logger.info(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init="auto")
    labels = kmeans.fit_predict(x_features)
    centroids = kmeans.cluster_centers_

    # 4. Medoid Selection and Export
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Exporting medoids to {output_dir}...")
    for i in range(n_clusters):
        # Find indices of structures assigned to this cluster
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue

        # Calculate distances to the cluster centroid
        dists = np.linalg.norm(x_features[cluster_indices] - centroids[i], axis=1)
        # The medoid is the structure closest to the centroid
        medoid_idx = cluster_indices[np.argmin(dists)]

        src_path = valid_files[medoid_idx]
        dst_path = os.path.join(output_dir, f"cluster_{i}_medoid.pdb")

        try:
            shutil.copy(src_path, dst_path)
            logger.info(f"  Cluster {i} (size={len(cluster_indices)}): Medoid is {src_path}")
        except Exception as e:
            logger.error(f"  Failed to export medoid for cluster {i}: {e}")

    logger.info("Clustering complete.")

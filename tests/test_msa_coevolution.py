import numpy as np
from sklearn.metrics import mutual_info_score

from synth_pdb.msa import generate_msa


def calculate_mutual_information_matrix(msa_array):
    """Given an MSA of shape (N_sequences, L_residues) containing integer labels,
    computes the L x L Mutual Information matrix.
    """
    L = msa_array.shape[1]
    mi_matrix = np.zeros((L, L))

    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            mi_matrix[i, j] = mutual_info_score(msa_array[:, i], msa_array[:, j])

    return mi_matrix


def test_msa_mutual_information_recovery():
    """The ultimate proof of Co-evolution.
    Generate a deep synthetic MSA constrained by a strict contact map.
    The Mutual Information between contacting pairs in the generated MSA
    must be statistically significantly higher than non-contacting pairs.
    """
    # 10-residue sequence
    base_sequence = "G" * 10
    L = len(base_sequence)

    # Define a clean contact map: Only (0, 9) and (2, 7) are in contact
    cmap = np.zeros((L, L), dtype=bool)
    true_contacts = [(0, 9), (2, 7)]
    for i, j in true_contacts:
        cmap[i, j] = True
        cmap[j, i] = True

    # Generate an MSA of 2000 sequences
    # A low enough temperature to enforce contacts, but high enough to cause drift
    msa_sequences = generate_msa(
        base_sequence, cmap, num_sequences=2000, temperature=1.5, steps_between_samples=100
    )

    assert len(msa_sequences) == 2000

    # Convert string MSA to integer array for sklearn mutual_info_score
    aa_to_int = {k: v for v, k in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    msa_array = np.zeros((2000, L), dtype=int)
    for row_idx, seq in enumerate(msa_sequences):
        for col_idx, char in enumerate(seq):
            msa_array[row_idx, col_idx] = aa_to_int.get(char, 0)

    # Calculate Mutual Information
    mi_matrix = calculate_mutual_information_matrix(msa_array)

    # Extract MI for true contacts
    contact_mis = []
    for i, j in true_contacts:
        contact_mis.append(mi_matrix[i, j])

    # Extract MI for random non-contacts (e.g., 0, 1) or (2, 5)
    non_contact_mis = []
    for i in range(L):
        for j in range(i + 1, L):
            if (i, j) not in true_contacts and (j, i) not in true_contacts:
                non_contact_mis.append(mi_matrix[i, j])

    avg_contact_mi = np.mean(contact_mis)
    avg_non_contact_mi = np.mean(non_contact_mis)

    # The statistical proof:
    # Mutual Information must be definitively higher for pairs forced to co-evolve
    # to avoid steric clashes in the Potts Energy Model.
    # 2.5x higher MI is a very strong statistical proof of correlation.
    assert avg_contact_mi > (avg_non_contact_mi * 2.5), (
        f"Co-evolution failed to imprint. "
        f"Contact MI: {avg_contact_mi:.4f}, Non-Contact MI: {avg_non_contact_mi:.4f}"
    )


import numpy as np


def export_constraints(contact_map: np.ndarray, sequence: str, fmt: str = "casp", separation_cutoff: int = 0, threshold: float = 8.0) -> str:
    """
    Export a Contact Map or Distance Matrix to text format for AI modeling.
    
    Parameters:
    -----------
    contact_map : np.ndarray
        NxN matrix. Values can be Binary (0/1), Probabilities (0.0-1.0), 
        or raw distances (Angstroms).
    sequence : str
        The protein sequence (required for CASP header).
    fmt : str
        "casp" (CASP RR format) or "csv" (Simple list).
    separation_cutoff : int
        Minimum sequence separation |i-j| to include. 
        Default 0 includes neighbors.
    threshold : float
        Distance cutoff for including a pair in the export.
        
    Returns:
    --------
    content : str
        The textual content of the file.
    """
    n_res = contact_map.shape[0]
    lines = []

    # Heuristic: Is this a distance matrix or a probability map?
    # If values are > 1.0, it's almost certainly distances.
    is_distance_matrix = np.any(contact_map > 1.0)

    if fmt == "casp":
        # CASP RR Format: i j d_minor d_major prob
        lines.append(sequence)

        for i in range(n_res):
            for j in range(i + 1 + separation_cutoff, n_res):
                val = contact_map[i, j]

                if is_distance_matrix:
                    # Input is raw distances
                    if val <= threshold:
                        res_i, res_j = i + 1, j + 1
                        # Use actual distance as the upper bound for the bin
                        lines.append(f"{res_i} {res_j} 0.0 {val:.1f} 1.00000")
                else:
                    # Input is binary or probabilities
                    if val > 0.0:
                        res_i, res_j = i + 1, j + 1
                        lines.append(f"{res_i} {res_j} 0.0 {threshold:.1f} {val:.5f}")

    elif fmt == "csv":
        # CSV Format: Res1,Res2,Distance_or_Prob
        lines.append("Res1,Res2,Value")
        for i in range(n_res):
            for j in range(i + 1 + separation_cutoff, n_res):
                val = contact_map[i, j]
                if is_distance_matrix:
                    if val <= threshold:
                        lines.append(f"{i+1},{j+1},{val:.5f}")
                else:
                    if val > 0.0:
                        lines.append(f"{i+1},{j+1},{val:.5f}")

    else:
        raise ValueError(f"Unknown format: {fmt}")

    return "\n".join(lines)

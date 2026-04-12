"""
merge_filter_rank.py
====================
Post-processing pipeline for Con-CDVAE generated crystal structures.

Steps
-----
1. Load one or more generated .pt files (output of gen_crystal.py).
2. Load CGCNN prediction CSVs for formation_energy and band_gap.
3. Merge predictions with structure data (crystal index alignment).
4. Filter: keep only structures that contain Lithium (atomic number 3).
5. Rank: composite score = w_e * formation_energy + w_bg * |band_gap - target_bg|
   (lower is better → more negative energy, band gap close to target).
6. Save ranked results to CSV and optionally save filtered .pt file.

Usage
-----
python scripts/merge_filter_rank.py \\
    --pt_files   src/model/mp20_format/eval_gen_FM0.pt src/model/mp20_format/eval_gen_FM1.pt \\
    --fm_csv     src/model/mp20_format/cgcnn_result/test_results_fm_pred.csv \\
    --bg_csv     src/model/mp20_format/cgcnn_result/test_results_bg_pred.csv \\
    --out_csv    scripts/ranked_li_crystals.csv \\
    --target_bg  1.5 \\
    --w_energy   1.0 \\
    --w_bandgap  0.5 \\
    --top_k      20 \\
    --save_pt    scripts/top_li_crystals.pt
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

# ── periodic table: atomic number → symbol ──────────────────────────────────
ELEMENT_MAP = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La",
    58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd",
    65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
    72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt",
    79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 86: "Rn",
    87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92: "U", 93: "Np",
    94: "Pu",
}
LI_ATOMIC_NUM = 3


# ── helpers ──────────────────────────────────────────────────────────────────

def load_pt_structures(pt_path: str):
    """
    Load a .pt file produced by gen_crystal.py and return per-crystal records.

    The .pt file contains tensors with a leading batch dimension:
        frac_coords  : (n_batches, total_atoms_in_batch, 3)
        num_atoms    : (n_batches, n_crystals_in_batch)
        atom_types   : (n_batches, total_atoms_in_batch)   – atomic numbers
        lengths      : (n_batches, n_crystals_in_batch, 3)
        angles       : (n_batches, n_crystals_in_batch, 3)

    Returns a list of dicts, one per crystal:
        {source_file, global_idx, atom_types (list[int]),
         lengths (list[float]), angles (list[float]),
         frac_coords (np.ndarray)}
    """
    data = torch.load(pt_path, map_location="cpu")
    source_name = os.path.basename(pt_path)

    frac_coords_all = data["frac_coords"]   # list of tensors or nested
    num_atoms_all   = data["num_atoms"]
    atom_types_all  = data["atom_types"]
    lengths_all     = data["lengths"]
    angles_all      = data["angles"]

    records = []
    global_idx = 0

    for batch_idx in range(len(num_atoms_all)):
        num_atoms_batch = num_atoms_all[batch_idx]   # 1-D tensor
        atom_types_batch = atom_types_all[batch_idx]  # 1-D tensor
        lengths_batch    = lengths_all[batch_idx]     # (n_cryst, 3)
        angles_batch     = angles_all[batch_idx]      # (n_cryst, 3)
        frac_coords_batch = frac_coords_all[batch_idx]  # (total_atoms, 3)

        atom_offset = 0
        for cryst_idx, n_at in enumerate(num_atoms_batch.tolist()):
            n_at = int(n_at)
            atom_slice = atom_types_batch[atom_offset: atom_offset + n_at].tolist()
            frac_slice = frac_coords_batch[atom_offset: atom_offset + n_at].numpy()
            length_row = lengths_batch[cryst_idx].tolist()
            angle_row  = angles_batch[cryst_idx].tolist()

            records.append({
                "source_file": source_name,
                "global_idx":  global_idx,
                "batch_idx":   batch_idx,
                "cryst_in_batch": cryst_idx,
                "n_atoms":     n_at,
                "atom_types":  atom_slice,
                "lengths":     length_row,
                "angles":      angle_row,
                "frac_coords": frac_slice,
            })
            atom_offset += n_at
            global_idx  += 1

    return records


def load_cgcnn_csv(csv_path: str, value_col: str):
    """
    Load a CGCNN prediction CSV.

    The existing file looks like:
        <crystal_idx>,<target>,<predicted>
    (no header row).

    Returns a DataFrame with columns: crystal_idx, target, <value_col>.
    """
    if not os.path.exists(csv_path):
        print(f"[WARNING] CGCNN CSV not found: {csv_path}  →  column '{value_col}' will be NaN")
        return None

    # Try to detect header automatically
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] == 3:
        df.columns = ["crystal_idx", "target", value_col]
    elif df.shape[1] == 2:
        df.columns = ["crystal_idx", value_col]
    else:
        raise ValueError(f"Unexpected number of columns ({df.shape[1]}) in {csv_path}")

    df["crystal_idx"] = df["crystal_idx"].astype(int)
    df[value_col] = df[value_col].astype(float)
    return df


def infer_formula(atom_types: list[int]) -> str:
    """Build reduced formula string from a list of atomic numbers."""
    from collections import Counter
    counts = Counter(atom_types)
    parts = []
    for z in sorted(counts):
        sym = ELEMENT_MAP.get(z, f"X{z}")
        n = counts[z]
        parts.append(sym if n == 1 else f"{sym}{n}")
    return "".join(parts)


def contains_lithium(atom_types: list[int]) -> bool:
    return LI_ATOMIC_NUM in atom_types


# ── main pipeline ─────────────────────────────────────────────────────────────

def main(args):
    # ── 1. Load all .pt files ────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Loading generated crystal .pt files")
    all_records = []
    for pt_path in args.pt_files:
        recs = load_pt_structures(pt_path)
        print(f"  {os.path.basename(pt_path)}: {len(recs)} crystals")
        all_records.extend(recs)

    if not all_records:
        print("[ERROR] No crystal records found. Exiting.")
        sys.exit(1)

    df = pd.DataFrame([{
        "source_file":    r["source_file"],
        "global_idx":     r["global_idx"],
        "batch_idx":      r["batch_idx"],
        "cryst_in_batch": r["cryst_in_batch"],
        "n_atoms":        r["n_atoms"],
        "formula":        infer_formula(r["atom_types"]),
        "has_Li":         contains_lithium(r["atom_types"]),
        "a":              r["lengths"][0],
        "b":              r["lengths"][1],
        "c":              r["lengths"][2],
        "alpha":          r["angles"][0],
        "beta":           r["angles"][1],
        "gamma":          r["angles"][2],
    } for r in all_records])

    print(f"  Total crystals loaded: {len(df)}")

    # ── 2. Load CGCNN property predictions ──────────────────────────────────
    print("\nStep 2: Loading CGCNN property predictions")

    df_fm = load_cgcnn_csv(args.fm_csv, "formation_energy")
    df_bg = load_cgcnn_csv(args.bg_csv, "band_gap")

    # Merge formation energy
    if df_fm is not None:
        df = df.merge(
            df_fm[["crystal_idx", "formation_energy"]],
            left_on="global_idx", right_on="crystal_idx", how="left"
        ).drop(columns=["crystal_idx"])
        print(f"  Formation energy predictions merged: {df['formation_energy'].notna().sum()} / {len(df)}")
    else:
        df["formation_energy"] = float("nan")

    # Merge band gap
    if df_bg is not None:
        df = df.merge(
            df_bg[["crystal_idx", "band_gap"]],
            left_on="global_idx", right_on="crystal_idx", how="left"
        ).drop(columns=["crystal_idx"])
        print(f"  Band gap predictions merged: {df['band_gap'].notna().sum()} / {len(df)}")
    else:
        df["band_gap"] = float("nan")

    # ── 3. Filter: Li-containing structures ─────────────────────────────────
    print("\nStep 3: Filtering for Lithium-containing structures")
    df_li = df[df["has_Li"]].copy()
    print(f"  Li-containing: {len(df_li)} / {len(df)} total")

    if df_li.empty:
        print("[WARNING] No Li-containing structures found after filtering.")
        print("  The model may not have generated any Li structures.")
        print("  → To generate Li structures, add Li to your target composition in the input CSV.")
        # Still write an empty CSV so the pipeline doesn't crash
        df_li.to_csv(args.out_csv, index=False)
        print(f"\n[OUTPUT] Empty ranked CSV written to: {args.out_csv}")
        return

    # ── 4. Rank: composite score ─────────────────────────────────────────────
    print("\nStep 4: Computing composite ranking score")
    print(f"  target_band_gap = {args.target_bg} eV")
    print(f"  w_energy = {args.w_energy},  w_bandgap = {args.w_bandgap}")
    print(f"  score = w_energy × formation_energy + w_bandgap × |band_gap − target_bg|")
    print("  (lower score = better candidate)")

    # Handle missing values: use 0.0 if no CGCNN available
    fe_vals = df_li["formation_energy"].fillna(0.0)
    bg_vals = df_li["band_gap"].fillna(args.target_bg)  # neutral if missing

    df_li["score"] = (
        args.w_energy  * fe_vals +
        args.w_bandgap * (bg_vals - args.target_bg).abs()
    )

    df_li = df_li.sort_values("score", ascending=True).reset_index(drop=True)
    df_li["rank"] = df_li.index + 1

    # Top-k selection
    top_k = min(args.top_k, len(df_li))
    df_top = df_li.head(top_k).copy()

    print(f"\n  Top-{top_k} Li structures (lower score = better candidate):")
    display_cols = ["rank", "formula", "source_file", "n_atoms",
                    "formation_energy", "band_gap", "score"]
    available = [c for c in display_cols if c in df_top.columns]
    print(df_top[available].to_string(index=False))

    # ── 5. Save outputs ──────────────────────────────────────────────────────
    print(f"\nStep 5: Saving results")

    # Save full ranked CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df_li.to_csv(args.out_csv, index=False)
    print(f"  Full ranked Li CSV → {args.out_csv}")

    # Optionally save top-k structures back to .pt for further evaluation
    if args.save_pt:
        print(f"  Saving top-{top_k} structures to {args.save_pt} ...")
        top_global_indices = set(df_top["global_idx"].tolist())
        top_recs = [r for r in all_records if r["global_idx"] in top_global_indices]

        # Pack back into the format gen_crystal.py uses
        frac_coords_list, num_atoms_list, atom_types_list = [], [], []
        lengths_list, angles_list = [], []

        for r in top_recs:
            frac_coords_list.append(torch.tensor(r["frac_coords"], dtype=torch.float32))
            num_atoms_list.append(r["n_atoms"])
            atom_types_list.append(torch.tensor(r["atom_types"], dtype=torch.long))
            lengths_list.append(r["lengths"])
            angles_list.append(r["angles"])

        save_dict = {
            "frac_coords":  [torch.cat(frac_coords_list, dim=0)],
            "num_atoms":    [torch.tensor(num_atoms_list, dtype=torch.long)],
            "atom_types":   [torch.cat(atom_types_list, dim=0)],
            "lengths":      [torch.tensor(lengths_list, dtype=torch.float32)],
            "angles":       [torch.tensor(angles_list, dtype=torch.float32)],
            "meta": {
                "source": "merge_filter_rank.py",
                "top_k": top_k,
                "target_bg": args.target_bg,
                "w_energy": args.w_energy,
                "w_bandgap": args.w_bandgap,
            }
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_pt)), exist_ok=True)
        torch.save(save_dict, args.save_pt)
        print(f"  Top-{top_k} Li .pt saved → {args.save_pt}")

    print("\n✓ Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_model_dir = os.path.join(repo_root, "src", "model", "mp20_format")

    parser = argparse.ArgumentParser(
        description="Merge CGCNN predictions, filter Li-based structures, rank by energy+bandgap"
    )
    parser.add_argument(
        "--pt_files", nargs="+", required=True,
        help="Path(s) to generated .pt files from gen_crystal.py"
    )
    parser.add_argument(
        "--fm_csv", default=os.path.join(default_model_dir, "cgcnn_result", "test_results_fm_pred.csv"),
        help="CGCNN CSV with formation energy predictions (crystal_idx, target, predicted)"
    )
    parser.add_argument(
        "--bg_csv", default=os.path.join(default_model_dir, "cgcnn_result", "test_results_bg_pred.csv"),
        help="CGCNN CSV with band gap predictions (crystal_idx, target, predicted). "
             "If not present, band_gap column will be NaN."
    )
    parser.add_argument(
        "--out_csv", default=os.path.join(default_model_dir, "ranked_li_crystals.csv"),
        help="Output path for the ranked CSV"
    )
    parser.add_argument(
        "--target_bg", type=float, default=1.5,
        help="Target band gap (eV) for ranking. Candidates closest to this value score better."
    )
    parser.add_argument(
        "--w_energy", type=float, default=1.0,
        help="Weight for formation energy in the composite score"
    )
    parser.add_argument(
        "--w_bandgap", type=float, default=0.5,
        help="Weight for |band_gap - target_bg| in the composite score"
    )
    parser.add_argument(
        "--top_k", type=int, default=20,
        help="Number of top structures to report and optionally save"
    )
    parser.add_argument(
        "--save_pt", default=None,
        help="If set, save top-k Li structures to this .pt file for further evaluation"
    )

    args = parser.parse_args()
    main(args)

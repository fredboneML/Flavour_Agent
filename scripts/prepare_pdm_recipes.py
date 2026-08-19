"""Convert the new all-fruit PDM export into a drop-in CSV for scoring_ms_assessment.ipynb.

The MS scoring pipeline (scripts/ms_scoring.py:load_recipes) maps recipe ingredients to
cluster weights by CAS number. The new export
`PDM_Rezepturen_Gesamt_nurP_18_08_2026.xlsx` has no CAS column: its ingredient code lives
in `Ident.1` (R-codes). This script adds `CAS-Nr.` via an Ident->CAS lookup and emits a CSV
with exactly the columns load_recipes expects, so the notebook re-runs unchanged.

Ident->CAS map is unioned from two sources (primary wins on conflict):
  1. data/gold/CAS Nummern.csv        (reference table, header at row 13)
  2. old strawberry CSV               (fills a few gaps)

Usage:
    python3 scripts/prepare_pdm_recipes.py
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "gold"

NEW_XLSX = DATA / "PDM_Rezepturen_Gesamt_nurP_18_08_2026.xlsx"
CAS_REF_CSV = DATA / "CAS Nummern.csv"
OLD_CSV = DATA / "Third_Trial_Set_PDM Erdbeere Gesamt 8-5-2026.csv"
OUT_CSV = DATA / "PDM_Rezepturen_Gesamt_nurP_18_08_2026.csv"

# Columns load_recipes() requires (scripts/ms_scoring.py)
OUT_COLS = ["Rez.-Nr.", "Ident", "CAS-Nr.", "Name", "Totalmenge"]


def build_ident_to_cas() -> dict:
    """Union Ident->CAS map from the reference table (primary) and the old CSV (fallback)."""
    ident_to_cas: dict = {}

    # Fallback source first, so the primary source overwrites on conflict.
    old = pd.read_csv(OLD_CSV, dtype=str)
    for ident, cas in zip(old["Ident"], old["CAS-Nr."]):
        if pd.notna(ident) and pd.notna(cas):
            ident_to_cas[str(ident).strip()] = str(cas).strip()

    # Primary source: CAS Nummern.csv, real header at row 13.
    ref = pd.read_csv(CAS_REF_CSV, dtype=str, sep=None, engine="python", header=13)
    ref.columns = [str(c).strip() for c in ref.columns]
    ident_col = next(c for c in ref.columns if c.startswith("Ident"))
    cas_col = next(c for c in ref.columns if "CAS" in c)
    for ident, cas in zip(ref[ident_col], ref[cas_col]):
        if pd.notna(ident) and pd.notna(cas):
            ident_to_cas[str(ident).strip()] = str(cas).strip()

    return ident_to_cas


def main() -> None:
    ident_to_cas = build_ident_to_cas()
    print(f"Ident->CAS map size: {len(ident_to_cas)}")

    df = pd.read_excel(NEW_XLSX, sheet_name="Rezept", header=14, dtype=str)
    print(f"Loaded new export: {df.shape[0]} rows, {df['Rez.-Nr.'].nunique()} recipes")

    ident = df["Ident.1"].astype(str).str.strip()
    out = pd.DataFrame(
        {
            "Rez.-Nr.": df["Rez.-Nr."],
            "Ident": ident,
            "CAS-Nr.": ident.map(ident_to_cas),
            "Name": df["Name"],
            "Totalmenge": df["Totalmenge"],
        }
    )[OUT_COLS]

    # Coverage stats
    total_rows = len(out)
    mapped_rows = out["CAS-Nr."].notna().sum()
    uniq_idents = ident.dropna().nunique()
    mapped_idents = sum(1 for i in ident.dropna().unique() if i in ident_to_cas)
    print(
        f"CAS coverage: {mapped_rows}/{total_rows} rows "
        f"({100 * mapped_rows / total_rows:.1f}%), "
        f"{mapped_idents}/{uniq_idents} idents "
        f"({100 * mapped_idents / uniq_idents:.1f}%)"
    )

    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()

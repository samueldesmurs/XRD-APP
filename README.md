# Thesis Analysis Code (XRD & Utilities)

This repository contains Python tools developed during my Master's thesis to **visualize, filter, and analyze XRD scans** and to support related characterization workflows.

## Contents

- `APP_XRD_patched_v9.py`  
  Interactive **Tkinter + Matplotlib** application to:
  - load `.xy` XRD scans recursively
  - optionally merge experimental metadata from `metadata.xlsx`
  - filter/group scans, color by condition, and export plots
  - optionally apply **ITO peak offset correction** using `ITO.txt`
  - (optional) assist phase/peak review and export tables for Origin/Excel

- Example inputs (optional):
  - `metadata.xlsx` — experiment table (sheet `FILES`), with a `path` (recommended) or `filename` column
  - `ITO.txt` — ITO reference pattern (2 columns or VESTA-like peak table)
  - `TEST FILE.xy` — sample `.xy` file

## Quick start

### 1) Create a clean environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the XRD app
Run from the folder that contains your `.xy` files (and optionally `metadata.xlsx` / `ITO.txt`):
```bash
python APP_XRD_patched_v9.py
```

## Input file conventions

### `.xy` scans
Plain text with two columns:
- 2θ (degrees)
- intensity (a.u.)

Lines starting with `#`, `;` or `//` are ignored.

### `metadata.xlsx` (optional but supported)
- File name: `metadata.xlsx`
- Default sheet: `FILES`
- Required column (recommended): `path` (relative path to each `.xy` file)
  - Alternative: `filename` if you prefer matching by name only
- Recommended column: `sample_id`
- You may include units in headers, e.g. `FAI [mg/mL]`, `annealing [min]`

## Reproducibility (recommended)

- Pin the exact version used for the thesis using a Git tag (e.g. `v1.0.0`) and/or a commit hash.
- If archiving with Zenodo, cite the release DOI in the thesis.

## License

See `LICENSE`.

## Contact

If you use this code for academic work, please cite the archived release (Zenodo DOI) when available.

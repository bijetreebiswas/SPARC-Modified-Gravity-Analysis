# SPARC Modified Gravity Analysis

This project extends the work of Dutta & Islam (2020) — *Testing Modified Gravity Theories with the Radial Acceleration Relation in the Milky Way* — to the full SPARC sample of 175 disk galaxies. It computes observed, baryonic, and predicted accelerations under Newtonian gravity, MOND, and Weyl conformal gravity, reproducing the Radial Acceleration Relation (RAR) and Halo Acceleration Relation (HAR) for the entire sample.

## Features

- Reads SPARC galaxy catalog (`SPARC_Lelli2016c.mrt`) and rotation curve files (`*_rotmod.dat`).
- Computes observed centripetal acceleration \(a_{\text{obs}}\) and baryonic Newtonian acceleration \(a_{\text{bar}}\).
- Calculates MOND acceleration using the standard interpolating function.
- Optionally computes Weyl gravity acceleration (disk contribution + global terms) using bulge‑disk decomposition parameters (`decomp.dat`).
- Generates three key plots:
  1. **Radial Acceleration Relation (RAR):** \(a_{\text{obs}}\) vs \(a_{\text{bar}}\) with Newtonian, MOND, and Weyl predictions.
  2. **Halo Acceleration Relation (HAR):** \(|a_{\text{obs}} - a_{\text{bar}}|\) vs \(a_{\text{bar}}\) with MOND and Weyl predictions.
  3. **Residual histogram** for the MOND fit.
- Saves the collected data to a CSV file for further analysis.

## Data

All data are from the SPARC (Spitzer Photometry and Accurate Rotation Curves) database. You will need:

- `SPARC_Lelli2016c.mrt` – Galaxy sample table.
- `*_rotmod.dat` – Rotation curve files (one per galaxy).
- `decomp.dat` – (Optional) Bulge‑disk decomposition parameters from the `BulgeDiskDecomp.zip` archive. Required for Weyl gravity calculations. If missing, the code will still run but omit Weyl points.

Place all files in the same directory and update the `data_path` variable in the script.

## Requirements

- Python 3.6+
- numpy
- matplotlib
- pandas
- scipy


# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:27:36 2026

@author: bijet
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.special import iv, kv

# ==================================================
# 1. Set the data folder
# ==================================================
data_path = r"C:\Users\bijet\OneDrive\Desktop\SPARC data"
os.chdir(data_path)
print("Working directory:", os.getcwd())

# ==================================================
# 2. Read galaxy names from SPARC_Lelli2016c.mrt
# ==================================================
def read_galaxy_names():
    """Extract galaxy names from the fixed‑width galaxy table."""
    galaxies = []
    # List of common prefixes for SPARC galaxy names
    prefixes = ('NGC', 'UGC', 'IC', 'PGC', 'ESO', 'DDO', 'F', 'UGCA', 'KK98', 'CamB', 'D512', 'D564', 'D631')
    with open("SPARC_Lelli2016c.mrt", "r") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # If the line starts with one of the prefixes, it's a galaxy name
            if any(line_stripped.startswith(p) for p in prefixes):
                # The first token is the galaxy name
                name = line_stripped.split()[0]
                galaxies.append(name)
    return galaxies

galaxies = read_galaxy_names()
print(f"Found {len(galaxies)} galaxies")
print("First 10:", galaxies[:10])

# ==================================================
# 3. Constants
# ==================================================
a0 = 1.21e-10          # m/s²
c = 3.0e8              # m/s
beta_star = 1.48e3     # m
gamma_star = 5.42e-39  # m^-1
gamma0 = 3.06e-28      # m^-1
kappa = 9.54e-50       # m^-2
Msun = 1.989e30        # kg
pc = 3.086e16          # m
kpc = 1000 * pc

def mond_acceleration(a_bar):
    """Standard MOND acceleration (Eq. 3.7)"""
    x = a_bar / a0
    return np.where(x > 0,
                    a_bar / np.sqrt(2) * np.sqrt(1 + np.sqrt(1 + (2*a0/a_bar)**2)),
                    0.0)

# ==================================================
# 4. Read bulge-disk decomposition (decomp.dat)
# ==================================================
def read_decomp():
    """Read bulge-disk decomposition file if available."""
    decomp = {}
    if not os.path.exists("decomp.dat"):
        print("decomp.dat not found – Weyl will be skipped.")
        return decomp
    with open("decomp.dat", "r") as f:
        lines = f.readlines()
    # Find the header line that starts with 'Galaxy'
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('Galaxy'):
            start = i + 1
            break
    for line in lines[start:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0]
        try:
            R_disk = float(parts[1])        # kpc
            S0_disk = float(parts[2])       # M☉/pc²
            decomp[name] = {'R_disk_kpc': R_disk, 'S0_disk_msun_pc2': S0_disk}
        except:
            continue
    print(f"Loaded {len(decomp)} disk parameters from decomp.dat")
    return decomp

decomp = read_decomp()

# ==================================================
# 5. Weyl gravity functions
# ==================================================
def weyl_disk_v2(r_m, R0_m, Sigma0_kg_m2):
    """Weyl disk contribution to v² (m²/s²)"""
    x = r_m / (2.0 * R0_m)
    N = 2.0 * np.pi * Sigma0_kg_m2 * R0_m**2
    I0 = iv(0, x)
    K0 = kv(0, x)
    I1 = iv(1, x)
    K1 = kv(1, x)
    term1 = N * beta_star * c**2 * r_m**2 / (2.0 * R0_m**3) * (I0*K0 - I1*K1)
    term2 = N * gamma_star * c**2 * r_m**2 / (2.0 * R0_m) * I1*K1
    return term1 + term2

def weyl_acceleration(r_kpc, name):
    """Return Weyl acceleration (m/s²) for given galaxy name."""
    if name not in decomp:
        return None
    R0_kpc = decomp[name]['R_disk_kpc']
    Sigma0_msun_pc2 = decomp[name]['S0_disk_msun_pc2']
    # Convert to SI
    R0_m = R0_kpc * kpc
    Sigma0_kg_m2 = Sigma0_msun_pc2 * Msun / (pc**2)
    r_m = r_kpc * kpc
    v2_local = weyl_disk_v2(r_m, R0_m, Sigma0_kg_m2)
    v2_global = gamma0 * c**2 * r_m / 2.0 - kappa * c**2 * r_m**2
    v2_tot = v2_local + v2_global
    return v2_tot / r_m

# ==================================================
# 6. Loop over all galaxies and compute accelerations
# ==================================================
all_r = []
all_a_obs = []
all_a_bar = []
all_a_mond = []
all_a_weyl = []

for name in galaxies:
    rot_file = f"{name}_rotmod.dat"
    if not os.path.exists(rot_file):
        continue   # skip missing files
    try:
        data = np.loadtxt(rot_file, comments='#')
        # Columns: 0:r(kpc), 1:Vobs, 2:errV, 3:Vgas, 4:Vdisk, 5:Vbul
        r_kpc = data[:,0]
        v_obs = data[:,1]
        v_gas = data[:,3]
        v_disk = data[:,4]
        v_bul = data[:,5]

        # Convert to SI
        r_m = r_kpc * kpc
        v_obs_m = v_obs * 1000.0
        a_obs = v_obs_m**2 / r_m

        v_new_km = np.sqrt(v_disk**2 + v_bul**2 + v_gas**2)
        v_new_m = v_new_km * 1000.0
        a_bar = v_new_m**2 / r_m

        a_mond = mond_acceleration(a_bar)

        all_r.append(r_kpc)
        all_a_obs.append(a_obs)
        all_a_bar.append(a_bar)
        all_a_mond.append(a_mond)

        # Weyl (if decomposition data available)
        a_weyl = weyl_acceleration(r_kpc, name)
        if a_weyl is not None:
            all_a_weyl.append(a_weyl)
        else:
            # Pad with NaN to keep list lengths consistent
            all_a_weyl.append(np.full_like(a_obs, np.nan))

    except Exception as e:
        print(f"Error processing {name}: {e}")
        continue

# Flatten lists
r_all = np.concatenate(all_r)
a_obs_all = np.concatenate(all_a_obs)
a_bar_all = np.concatenate(all_a_bar)
a_mond_all = np.concatenate(all_a_mond)
if all_a_weyl:
    a_weyl_all = np.concatenate(all_a_weyl)
    has_weyl = True
else:
    a_weyl_all = None
    has_weyl = False

print(f"Collected {len(a_obs_all)} data points from {len(all_r)} galaxies.")
if has_weyl:
    print(f"Weyl data points: {np.sum(~np.isnan(a_weyl_all))}")

# ==================================================
# 7. Plot 1: Radial Acceleration Relation (RAR)
# ==================================================
plt.figure(figsize=(6,5))
plt.loglog(a_bar_all, a_obs_all, 'k.', markersize=1, alpha=0.3, label='Data')
a_bar_grid = np.logspace(-12, -8, 100)
a_mond_grid = mond_acceleration(a_bar_grid)
plt.loglog(a_bar_grid, a_mond_grid, 'r-', linewidth=2, label='MOND')
plt.plot(a_bar_grid, a_bar_grid, 'b--', linewidth=1, label='Newtonian')
if has_weyl:
    mask = ~np.isnan(a_weyl_all)
    plt.loglog(a_bar_all[mask], a_weyl_all[mask], 'g.', markersize=1, alpha=0.5, label='Weyl (disk only)')
plt.xlabel(r'$a_{\mathrm{bar}}$ (m/s$^2$)')
plt.ylabel(r'$a_{\mathrm{obs}}$ (m/s$^2$)')
plt.xlim(1e-12, 1e-8)
plt.ylim(1e-12, 1e-8)
plt.legend()
plt.title('Radial Acceleration Relation (SPARC)')
plt.grid(True, alpha=0.3)
plt.savefig('RAR_all_galaxies.png', dpi=150)
plt.show()

# ==================================================
# 8. Plot 2: Halo Acceleration Relation (HAR)
# ==================================================
a_halo = a_obs_all - a_bar_all
plt.figure(figsize=(6,5))
plt.loglog(a_bar_all, np.abs(a_halo), 'k.', markersize=1, alpha=0.3, label='Data')
a_mond_halo = a_mond_grid - a_bar_grid
plt.loglog(a_bar_grid, np.abs(a_mond_halo), 'r-', linewidth=2, label='MOND')
if has_weyl:
    a_halo_weyl = a_weyl_all - a_bar_all
    mask = ~np.isnan(a_halo_weyl)
    plt.loglog(a_bar_all[mask], np.abs(a_halo_weyl[mask]), 'g.', markersize=1, alpha=0.5, label='Weyl (disk only)')
plt.xlabel(r'$a_{\mathrm{bar}}$ (m/s$^2$)')
plt.ylabel(r'$|a_{\mathrm{halo}}|$ (m/s$^2$)')
plt.xlim(1e-12, 1e-8)
plt.ylim(1e-13, 1e-9)
plt.legend()
plt.title('Halo Acceleration Relation (SPARC)')
plt.grid(True, alpha=0.3)
plt.savefig('HAR_all_galaxies.png', dpi=150)
plt.show()

# ==================================================
# 9. Plot 3: Residual histogram for MOND
# ==================================================
res_mond = ((a_obs_all - a_mond_all)**2 / a_obs_all**2)
plt.figure()
plt.hist(res_mond, bins=50, log=True, alpha=0.7, label='MOND')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residuals (MOND)')
plt.savefig('residuals_mond.png', dpi=150)
plt.show()

# ==================================================
# 10. Save results to CSV
# ==================================================
df = pd.DataFrame({
    'r_kpc': r_all,
    'a_obs': a_obs_all,
    'a_bar': a_bar_all,
    'a_mond': a_mond_all
})
if has_weyl:
    df['a_weyl'] = a_weyl_all
df.to_csv('SPARC_RAR_results.csv', index=False)
print("Saved results to SPARC_RAR_results.csv")
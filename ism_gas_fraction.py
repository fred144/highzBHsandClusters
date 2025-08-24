# %% need to see what the results of varying the normalization kappa in the KS law 
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.size": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)

file = "./runs/smhm_2phase_redshift_scan_KS_kap0p1_rd_0p05.h5"
f = h5py.File(file, "r")
smhm_normalized = f["SMHM"]  # smhm is already normalized by baryon fractions
mhalo_obs = f["Mhalo_obs"]
mstar = f["Mstar_obs"]
mism = f["MISM_obs"]
zs = f["redshifts"][:]
print(f.keys())
print(f["redshifts"][:])
#%%
fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)

# loop through redshift and plot mgas/mstar vs mstar
for i, z in enumerate(zs):
    ax.plot(mstar[i], mism[i] / mstar[i],  label=f"z={z:.2f}", marker="o")


# make a line passing through (8.7, 0.322and (10.35, -0.5) and extend it to the left and right
ax.plot(
    [10**8.7, 10**10.35],
    [10**0.2, 10**-0.4],
    ls="-",
    color="k",
    label=r"Calette",
)

ax.legend(ncols=3, frameon=False, bbox_to_anchor=(0,1.1), loc="lower left")

ax.set(xscale="log", yscale="log", xlabel=r"$M_\star$ [M$_\odot$]", ylabel=r"$M_{\rm {ISM}}/M_\star$")
plt.show()
# %%

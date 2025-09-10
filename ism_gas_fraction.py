# %% need to see what the results of varying the normalization kappa in the KS law
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cmasher as cmr

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
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)

file = "./runs/smhm_2phase_redshift_scan_KS_kap0p1_rd_0p02_wSFRandZ.h5"
f = h5py.File(file, "r")
smhm_normalized = f["SMHM"]  # smhm is already normalized by baryon fractions
mhalo_obs = f["Mhalo_obs"]
mstar = f["Mstar_obs"]
mism = f["MISM_obs"]
zs = f["redshifts"][:]
sfr = f["SFR_obs"]
mmetal_cgm = f["MMetals_obs"]    
print(f.keys())
print(f["redshifts"][:])
# %%
fig, ax = plt.subplots(2, 1, figsize=(4.5, 6.5), dpi=300, sharex=True)
plt.subplots_adjust(hspace=0.05)
# get a colormap from cmasher
cmap = plt.get_cmap("Set1")
colors = cmap(np.linspace(0, 1, len(zs)))

# loop through redshift and plot mgas/mstar vs mstar
for i, z in enumerate(zs):
    ax[0].plot(
        mstar[i],
        mism[i] / mstar[i],
        label=f"{z:.1f}",
        marker="o",
        color=colors[i],
        markeredgecolor="k",
    )
    t_depletion = mism[i] / sfr[i]
    sSFR_Gyr = sfr[i] / mstar[i] # Gyr^-1
    sSFR_yr = sSFR_Gyr * 1e-9   
    ax[1].plot(
        mstar[i],
        sSFR_yr,
        label=f"{z:.1f}",
        marker="o",
        color=colors[i],
        markeredgecolor="k",
    )


# make a line passing through (8.7, 0.322and (10.35, -0.5) and extend it to the left and right
ax[0].plot(
    [10**8.7, 10**10.35],
    [10**0.2, 10**-0.4],
    ls="-",
    color="k",
    label=r"Calette",
)

ax[0].legend(
    ncols=4,
    frameon=False,
    bbox_to_anchor=(-0.05, 1.011),
    loc="lower left",
    title=r"redshift $z$",
    fontsize=10,
)

ax[0].set(xscale="log", yscale="log", ylabel=r"$M_{\rm {ISM}}/M_\star$")
ax[1].set(
    xscale="log",
    yscale="log",
    xlabel=r"$M_\star$ [M$_\odot$]",
    ylabel=r"sSFR [yr$^{-1}$]",
)
plt.savefig(
    "./figures/ism_gas_fractions_2phase.png", dpi=200, bbox_inches="tight", pad_inches=0.05
)
plt.show()
# %%

z_sun = 0.0134 
fig2, ax2 = plt.subplots(figsize=(4.5, 3.5), dpi=300)
for i, z in enumerate(zs):
    metallicity = (mmetal_cgm[i] / mism[i]) / z_sun
    ax2.plot(
        mhalo_obs[i],
        metallicity,
        label=f"{z:.1f}",
        marker="o",
        color=colors[i],
        markeredgecolor="k",
    )

ax2.set(
    xscale="log",
    yscale="log",
    xlabel=r"$M_\star$ [M$_\odot$]",
    ylabel=r"CGM Metallicity [$Z_\odot$]",
)
ax2.legend(
    ncols=4,
    frameon=False,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    title=r"redshift $z$",
    fontsize=10,
)
plt.savefig(
    "./figures/cgm_metallicity_2phase.png", dpi=200, bbox_inches="tight", pad_inches=0.05
)
plt.show()

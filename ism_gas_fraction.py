# %% need to see what the results of varying the normalization kappa in the KS law
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cmasher as cmr
import astropy.units as u
from astropy import constants as consts
from astropy import cosmology

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

# standard flat cosmology
H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7

Ob0 = 0.0490
f_b = Ob0 / Omegam0
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)


# calculate corresponding virial temperatures
def virial_T(mhalo, Rvir):
    """Halo fp

    Args:
        mhalo (_type_): halo mass
        Rvir (_type_): halo virial radius

    Returns:
        _type_: virial temperture
    """
    G = consts.G
    kb = consts.k_B
    mp = consts.m_p
    return ((2 / 5) * ((G * mhalo * mp) / (Rvir * kb))).to(u.K)


def virial_radius(z, mhalo, Delc=200):
    """
    Halo virial radius, classical 200 top-hat overdensity.
    Supports z as a 1D array and mhalo as a 2D array (shape: [len(z), N]).
    Args:
        z (array-like): Redshift array of shape (len(z),)
        mhalo (array-like): Halo mass array of shape (len(z), N)
        Delc (float): Overdensity parameter (default: 200)
    Returns:
        ndarray: Virial radius array of shape (len(z), N)
    """
    z = np.asarray(z)

    # Broadcast critical density to shape (len(z), 1)
    rhoc = LCDM.critical_density(z)[:, np.newaxis]
    rvir = (mhalo / (rhoc * (4 / 3) * np.pi * Delc)) ** (1 / 3)

    return rvir.to(u.kpc)


def circular_velocity(mhalo, Rvir):
    """circular velocity of the halo"""
    G = consts.G
    return np.sqrt(G * mhalo / Rvir).to(u.km / u.s)


def vcirc_from_virial_T(Tvir, mu=0.59):
    kb = consts.k_B
    mp = consts.m_p
    return np.sqrt((2 * kb * Tvir) / (mu * mp)).to(u.km / u.s)


file = "./runs/smhm_2phase_redshift_scan_KS_kap0p1_rd_0p02_wSFRandZ.h5"
f = h5py.File(file, "r")
smhm_normalized = f["SMHM"]  # smhm is already normalized by baryon fractions
mhalo_obs = f["Mhalo_obs"]
mstar = f["Mstar_obs"]
mism = f["MISM_obs"]
zs = f["redshifts"][:]
sfr = f["SFR_obs"]
mmetal_cgm = f["MMetals_obs"]
rvirs = virial_radius(zs, mhalo_obs * u.Msun)  # virial radius in kpc
halo_Tvirs = virial_T(mhalo_obs * u.Msun, rvirs)  # virial temperature in K
print(f.keys())
print(f["redshifts"][:])
# %%
fig, ax = plt.subplots(2, 1, figsize=(4.5, 6.5), dpi=300, sharex=True)
plt.subplots_adjust(hspace=0.05)
# get a colormap from cmasher
cmap = plt.get_cmap("Set1")
colors = cmap(np.linspace(0, 1, len(zs)))
Tvir_max = 1e6 * u.K
# loop through redshift and plot mgas/mstar vs mstar
for i, z in enumerate(zs):
    Tvirs = halo_Tvirs[i]
    mask = Tvirs < Tvir_max

    ax[0].plot(
        mstar[i],
        mism[i] / mstar[i],
       
        color=colors[i],
    )
    ax[0].scatter(
        mstar[i][mask],
        mism[i][mask] / mstar[i][mask],
        s=30,
        color=colors[i],
        edgecolor="k",
        zorder=3,
        label=f"{z:.1f}",
    )

    t_depletion = mism[i] / sfr[i]
    sSFR_Gyr = sfr[i] / mstar[i]  # Gyr^-1
    sSFR_yr = sSFR_Gyr * 1e-9
    ax[1].plot(
        mstar[i],
        sSFR_yr,
        color=colors[i],
    )
    ax[1].scatter(
        mstar[i][mask], sSFR_yr[mask], s=30, color=colors[i], edgecolor="k", zorder=3
    )


# make a line passing through (8.7, 0.322and (10.35, -0.5) and extend it to the left and right

# Calculate the slope and intercept in log-log space
x1, y1 = 8.7, 0.2
x2, y2 = 10.35, -0.4

# Generate x values spanning the plot range
x_vals = np.linspace(6, 12, 100)
# Compute corresponding y values for the line in log-log space
slope = (y2 - y1) / (x2 - x1)
y_vals = y1 + slope * (x_vals - x1)

ax[0].plot(
    10**x_vals,
    10**y_vals,
    ls="-",
    color="k",
    lw="4",
    alpha=0.5,
    zorder=1,
)
# put a text label near this line 
ax[0].text(0.85, 0.22, r"Calette+18, $z\sim 0$", transform=ax[0].transAxes, fontsize=8, rotation=-35, va="center", ha="right")

ax[0].legend(
    ncols=4,
    frameon=False,
    bbox_to_anchor=(-0.05, 1.011),
    loc="lower left",
    title=r"redshift $z$",
    fontsize=10,
)

ax[0].set(xscale="log", yscale="log", ylabel=r"$M_{\rm {ISM}}/M_\star$", ylim=(0.1, 8))
ax[1].set(
    xscale="log",
    yscale="log",
    xlabel=r"$M_\star$ [M$_\odot$]",
    ylabel=r"sSFR [yr$^{-1}$]",
    xlim=(8e6, 8e11),
)
plt.savefig(
    "./figures/ism_gas_fractions_2phase.png",
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.05,
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
    "./figures/cgm_metallicity_2phase.png",
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()

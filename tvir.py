# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const

# get harikane data
data = np.loadtxt("./data/harikane_2022.txt", delimiter=",")
log_mvir = data[:, 0]
log_sfr_per_DM = data[:, 1]
redshift = data[:, 2]


def tvir_osterbrock(mvir, zred, h=0.6711, mu=1):
    """https://ned.ipac.caltech.edu/level5/Sept15/Johnson/Johnson1.html#:~:text=As%20can%20be%20inferred%20from,with%20such%20a%20virial%20temperature.

    Args:
        mvir (_type_): _description_
        redhisft (_type_): _description_
        h (float, optional): _description_. Defaults to 0.6711.
        mu (int, optional): _description_. Defaults to 1.
    """

    Tvir_K = 4e4 * (mu / 1.2) * (mvir / (1e8 * h**-1)) ** (2 / 3) * ((1 + zred) / 10)
    return Tvir_K

def tvir_ohandhaiman(mvir, zred, h=0.6711, mu=1):
    return 1e4* (mvir / 5e7)**(2/3) * ((1 + zred) / 10)

def virial_radius(mvir, zred, h=0.6711):
    return (800 / h) * (mvir / (1e8 * h**-1)) ** (1 / 3) * ((1 + zred) / 10)**-1

    
f_baryon = 0.16
redshift_bins = [1.7, 2.2, 3.0, 4.0, 5.0, 6.0, 7.0]
colors = plt.cm.rainbow(np.linspace(0, 1, len(redshift_bins)))
mvir = 10**log_mvir
sfr_per_DM = 10**log_sfr_per_DM

fig, ax = plt.subplots(2, 2, figsize=(5, 5), dpi=300)
ax=ax.ravel()
plt.subplots_adjust(wspace=0.45, hspace=0.35)
for i, z in enumerate(redshift_bins):
    mask = redshift == z
    vir_temp = mvir[mask] ** (2 / 3) * (1 + z)
    vir_temp_os = tvir_ohandhaiman(mvir[mask], redshift[mask])
    r_vir_pc = virial_radius(mvir[mask], redshift[mask])
    r_vir_m = r_vir_pc * u.pc.to(u.m)
    mvir_kg = mvir[mask] * u.Msun.to(u.kg)
    sigma_max = np.sqrt(const.G.value * mvir_kg / r_vir_m) # m/s
    sigma_max_kms = sigma_max / 1e3
    
    m_bh = 1.9 *  1e8 * (sigma_max_kms / 200 )*5.1
    
    t_vir_alt = (3/2) * sigma_max
    
    # potential
    phi = mvir[mask] * vir_temp_os * f_baryon
    
    ax[0].scatter(mvir[mask], sfr_per_DM[mask] / f_baryon, label=f"z={z}", color=colors[i], edgecolors="black")
    ax[1].scatter(vir_temp_os, sfr_per_DM[mask]/ f_baryon, label=f"z={z}", color=colors[i], edgecolors="black")
    ax[2].scatter(sigma_max_kms, sfr_per_DM[mask]/ f_baryon, label=f"z={z}", color=colors[i], edgecolors="black")
    ax[3].scatter(m_bh, sfr_per_DM[mask]/ f_baryon, label=f"z={z}", color=colors[i], edgecolors="black")

ax[0].legend(ncols=3, loc="upper left", bbox_to_anchor=(0.0, 1.5))
ax[0].set(
    xscale="log",
    yscale="log",
    ylim=(0.01, 1),
    xlabel=r"$M_{\rm vir} \, [M_{\odot}]$",
    ylabel=r"$\rm SFR /  f_b \dot{M}_{\rm h} $",
)
ax[1].set(
    xscale="log",
    yscale="log",
    ylim=(0.01, 1),
    xlabel=r"$T_{\rm vir}$ [K]",
    ylabel=r"$\rm SFR / f_b \dot{M}_{\rm h} $",
)
ax[2].set(
    
    yscale="log",
    ylim=(0.01, 1),
    xlabel=r"$\sigma$ [km/s]",
    ylabel=r"$\rm SFR / f_b  \dot{M}_{\rm h} $",
)
ax[3].set(yscale="log", xscale="log", ylim=(0.01, 1), xlabel=r"$M_{\rm BH} \, [M_{\odot}]$", ylabel=r"$\rm SFR / f_b \dot{M}_{\rm h} $", xlim=(2e8, 1e10))

plt.savefig("sfr_over_DMaccretion_vs_Tvir.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
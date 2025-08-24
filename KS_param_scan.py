# %% scan through the KS_n and KS_kappa_s of the KS law for the 2 phase model

import numpy as np
import matplotlib.pyplot as plt
from cgm_sf_regulator import CGMRegulator
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import astropy.units as u


mhalo_z0 = 1e12 * u.Msun
t_span = (0.2, 13.3)  # gyrs
n_samples = 10
n_fixed = 1.5  # keep n fixed for now
kappa_s_fixed = 1
disk_scale_length_fixed = 0.02  # of rvir


# %%
# Store results for each kappa_s
timescales = {
    "tcool_real": [],
    "t_dynamical": [],
    "t_dep_effect": [],
    "t_ejection": [],
    "sim_time": [],
}
sf_statistics = {
    "SFR": [],
    "SFE": [],
    "m_star": [],
    "m_ism": []
}

kappa_s_to_try = np.logspace(-1, 1, n_samples)
for kappa_s in kappa_s_to_try:
    model = CGMRegulator(mhalo_z0, t_span, KS_kappa_s=kappa_s, KS_n=n_fixed)
    model.run_halo()
    results = model.get_results()
    derived = model.get_derived_quantities()
    timescales["tcool_real"].append(derived["tcool_real"])
    timescales["t_dynamical"].append(derived["t_dynamical"])
    timescales["t_dep_effect"].append(results["m_star"] / derived["dot_m_sfr"])
    timescales["t_ejection"].append(derived["t_ejection"])
    timescales["sim_time"].append(derived["sim_time"])

    sf_statistics["SFR"].append(derived["dot_m_sfr"])
    sf_statistics["SFE"].append(derived["halo_sfe"])
    sf_statistics["m_star"].append(results["m_star"])
    sf_statistics["m_ism"].append(results["m_ism"])


# %%

# Scan for KS_n values
n_to_try = np.linspace(1.0, 2.0, n_samples)
timescales_n = {
    "tcool_real": [],
    "t_dynamical": [],
    "t_dep_effect": [],
    "t_ejection": [],
    "sim_time": [],
}
sf_statistics_n = {
    "SFR": [],
    "SFE": [],
    "m_star": [],
    "m_ism": []
}

for n_val in n_to_try:
    model_n = CGMRegulator(mhalo_z0, t_span, KS_kappa_s=kappa_s_fixed, KS_n=n_val)
    model_n.run_halo()
    results_n = model_n.get_results()
    derived_n = model_n.get_derived_quantities()
    timescales_n["tcool_real"].append(derived_n["tcool_real"])
    timescales_n["t_dynamical"].append(derived_n["t_dynamical"])
    timescales_n["t_dep_effect"].append(results_n["m_star"] / derived_n["dot_m_sfr"])
    timescales_n["t_ejection"].append(derived_n["t_ejection"])
    timescales_n["sim_time"].append(derived_n["sim_time"])
    sf_statistics_n["SFR"].append(derived_n["dot_m_sfr"])
    sf_statistics_n["SFE"].append(derived_n["halo_sfe"])
    sf_statistics_n["m_star"].append(results_n["m_star"])
    sf_statistics_n["m_ism"].append(results_n["m_ism"])

# %% now for disk
r_disk_to_try = np.logspace(np.log10(0.01), np.log10(0.2), n_samples)
r_disk_to_try = np.sort(np.append(r_disk_to_try, 0.02))  # center at 0.02
timescales_disk = {
    "tcool_real": [],
    "t_dynamical": [],
    "t_dep_effect": [],
    "t_ejection": [],
    "sim_time": [],
}
sf_statistics_disk = {
    "SFR": [],
    "SFE": [],
    "m_star": [],
    "m_ism": []
}

for r_disk in r_disk_to_try:
    model_disk = CGMRegulator(
        mhalo_z0,
        t_span,
        KS_kappa_s=kappa_s_fixed,
        KS_n=n_fixed,
        disk_scale_length=r_disk,
    )
    model_disk.run_halo()
    results_disk = model_disk.get_results()
    derived_disk = model_disk.get_derived_quantities()
    timescales_disk["tcool_real"].append(derived_disk["tcool_real"])
    timescales_disk["t_dynamical"].append(derived_disk["t_dynamical"])
    timescales_disk["t_dep_effect"].append(
        results_disk["m_star"] / derived_disk["dot_m_sfr"]
    )
    timescales_disk["t_ejection"].append(derived_disk["t_ejection"])
    timescales_disk["sim_time"].append(derived_disk["sim_time"])
    sf_statistics_disk["SFR"].append(derived_disk["dot_m_sfr"])
    sf_statistics_disk["SFE"].append(derived_disk["halo_sfe"])
    sf_statistics_disk["m_star"].append(results_disk["m_star"])
    sf_statistics_disk["m_ism"].append(results_disk["m_ism"])

# %%
# Plotting
fig, ax = plt.subplots(3, 3, figsize=(10, 8), sharex=True, dpi=300, sharey="row")
plt.subplots_adjust(hspace=0, wspace=0)

# --- Left column: kappa_s scan ---
cmap_kappa = plt.get_cmap("coolwarm")
norm_kappa = mpl.colors.LogNorm(vmin=kappa_s_to_try.min(), vmax=kappa_s_to_try.max())
sm_kappa = mpl.cm.ScalarMappable(norm=norm_kappa, cmap=cmap_kappa)

for i, kappa_s in enumerate(kappa_s_to_try):
    color = cmap_kappa(norm_kappa(kappa_s))
    ax[0, 0].plot(timescales["sim_time"][i], timescales["tcool_real"][i], color=color)
    ax[1, 0].plot(timescales["sim_time"][i], timescales["t_dep_effect"][i], color=color)
    ax[2, 0].plot(timescales["sim_time"][i], timescales["t_ejection"][i], color=color)

ax[0, 0].set_ylabel(r"$t_{\mathrm{cool}}$ [Gyr]")
ax[1, 0].set_ylabel(r"$t_{\mathrm{dep, eff}}$ [Gyr]")
ax[2, 0].set_ylabel(r"$t_{\mathrm{eject}}$ [Gyr]")
ax[2, 0].set_xlabel("Time [Gyr]")

# Add colorbar for kappa_s
cax_kappa = ax[0, 0].inset_axes([0.05, 1.1, 0.9, 0.05], transform=ax[0, 0].transAxes)
cb_kappa = fig.colorbar(sm_kappa, cax=cax_kappa, orientation="horizontal")
cax_kappa.set_title(r"$\kappa_s$")
cax_kappa.xaxis.set_ticks_position("top")

# --- Middle column: n scan ---
cmap_n = plt.get_cmap("coolwarm")
norm_n = mpl.colors.Normalize(vmin=n_to_try.min(), vmax=n_to_try.max())
sm_n = mpl.cm.ScalarMappable(norm=norm_n, cmap=cmap_n)

for i, n_val in enumerate(n_to_try):
    color = cmap_n(norm_n(n_val))
    ax[0, 1].plot(
        timescales_n["sim_time"][i], timescales_n["tcool_real"][i], color=color
    )
    ax[1, 1].plot(
        timescales_n["sim_time"][i], timescales_n["t_dep_effect"][i], color=color
    )
    ax[2, 1].plot(
        timescales_n["sim_time"][i], timescales_n["t_ejection"][i], color=color
    )


ax[2, 1].set_xlabel("Time [Gyr]")

# Add colorbar for n
cax_n = ax[0, 1].inset_axes([0.05, 1.1, 0.9, 0.05], transform=ax[0, 1].transAxes)
cb_n = fig.colorbar(sm_n, cax=cax_n, orientation="horizontal")
cax_n.set_title(r"$n$")
cax_n.xaxis.set_ticks_position("top")

# --- Right column: r_disk scan ---
cmap_disk = plt.get_cmap("coolwarm")
norm_disk = mpl.colors.LogNorm(vmin=r_disk_to_try.min(), vmax=r_disk_to_try.max())
sm_disk = mpl.cm.ScalarMappable(norm=norm_disk, cmap=cmap_disk)

for i, r_disk in enumerate(r_disk_to_try):
    color = cmap_disk(norm_disk(r_disk))
    ax[0, 2].plot(
        timescales_disk["sim_time"][i], timescales_disk["tcool_real"][i], color=color
    )
    ax[1, 2].plot(
        timescales_disk["sim_time"][i], timescales_disk["t_dep_effect"][i], color=color
    )
    ax[2, 2].plot(
        timescales_disk["sim_time"][i], timescales_disk["t_ejection"][i], color=color
    )


ax[2, 2].set_xlabel("Time [Gyr]")

# Add colorbar for r_disk
cax_disk = ax[0, 2].inset_axes([0.05, 1.1, 0.9, 0.05], transform=ax[0, 2].transAxes)
cb_disk = fig.colorbar(sm_disk, cax=cax_disk, orientation="horizontal")
cax_disk.set_title(r"$r_{\mathrm{disk}}/r_{\mathrm{vir}}$")
cax_disk.xaxis.set_ticks_position("top")

# Common formatting and free fall time reference
for i, a in enumerate(ax.flatten()):
    a.set(yscale="log", ylim=(1e-3, 2), xlim=(0.15, 0.5))
    label = "free fall time"
    if i  == 0:
        a.plot(
            timescales_n["sim_time"][0],
            timescales_n["t_dynamical"][0],
            color="grey",
            linestyle="--",
            lw=2,
            label=label,
        )
    else:
        a.plot(
            timescales_n["sim_time"][0],
            timescales_n["t_dynamical"][0],
            color="grey",
            linestyle="--",
            lw=2,
        )

ax[0, 0].legend(frameon=False)
plt.savefig(
    "./figures/KS_param_scan.png", dpi=300, bbox_inches="tight", pad_inches=0.05
)
plt.show()


#%%
# Plot SFE, SFR, m_gas, m_star for each parameter scan
fig2, ax2 = plt.subplots(4, 3, figsize=(10, 10), sharex=True, dpi=300, sharey="row")
plt.subplots_adjust(hspace=0.0, wspace=0.0)

# --- Left column: kappa_s scan ---
for i, kappa_s in enumerate(kappa_s_to_try):
    color = cmap_kappa(norm_kappa(kappa_s))
    ax2[0, 0].plot(timescales["sim_time"][i], sf_statistics["SFE"][i], color=color)
    ax2[1, 0].plot(timescales["sim_time"][i], sf_statistics["SFR"][i], color=color)
    ax2[2, 0].plot(timescales["sim_time"][i], sf_statistics["m_ism"][i], color=color)
    ax2[3, 0].plot(timescales["sim_time"][i], sf_statistics["m_star"][i], color=color)

ax2[0, 0].set_ylabel("SFE")
ax2[0, 0].set_ylim(2e-3, None)
ax2[1, 0].set_ylabel("SFR [Msun/Gyr]")
ax2[1, 0].set_ylim(1e7, None)
ax2[2, 0].set_ylabel(r"$M_{\mathrm{gas}}$ [Msun]")
ax2[2, 0].set_ylim(5e5, 5e9)
ax2[3, 0].set_ylabel(r"$M_{\star}$ [Msun]")
ax2[3, 0].set_ylim(5e5, 5e9)
ax2[3, 0].set_xlabel("Time [Gyr]")

cax2_kappa = ax2[0, 0].inset_axes([0.05, 1.1, 0.9, 0.05], transform=ax2[0, 0].transAxes)
fig2.colorbar(sm_kappa, cax=cax2_kappa, orientation="horizontal")
cax2_kappa.set_title(r"$\kappa_s$")
cax2_kappa.xaxis.set_ticks_position("top")

# --- Middle column: n scan ---
for i, n_val in enumerate(n_to_try):
    color = cmap_n(norm_n(n_val))
    ax2[0, 1].plot(timescales_n["sim_time"][i], sf_statistics_n["SFE"][i], color=color)
    ax2[1, 1].plot(timescales_n["sim_time"][i], sf_statistics_n["SFR"][i], color=color)
    ax2[2, 1].plot(timescales_n["sim_time"][i], sf_statistics_n["m_ism"][i], color=color)
    ax2[3, 1].plot(timescales_n["sim_time"][i], sf_statistics_n["m_star"][i], color=color)

ax2[3, 1].set_xlabel("Time [Gyr]")

cax2_n = ax2[0, 1].inset_axes([0.05, 1.1, 0.9, 0.05], transform=ax2[0, 1].transAxes)
fig2.colorbar(sm_n, cax=cax2_n, orientation="horizontal")
cax2_n.set_title(r"$n$")
cax2_n.xaxis.set_ticks_position("top")

# --- Right column: r_disk scan ---
for i, r_disk in enumerate(r_disk_to_try):
    color = cmap_disk(norm_disk(r_disk))
    ax2[0, 2].plot(timescales_disk["sim_time"][i], sf_statistics_disk["SFE"][i], color=color)
    ax2[1, 2].plot(timescales_disk["sim_time"][i], sf_statistics_disk["SFR"][i], color=color)
    ax2[2, 2].plot(timescales_disk["sim_time"][i], sf_statistics_disk["m_ism"][i], color=color)
    ax2[3, 2].plot(timescales_disk["sim_time"][i], sf_statistics_disk["m_star"][i], color=color)

ax2[3, 2].set_xlabel("Time [Gyr]")

cax2_disk = ax2[0, 2].inset_axes([0.05, 1.1, 0.9, 0.05], transform=ax2[0, 2].transAxes)
fig2.colorbar(sm_disk, cax=cax2_disk, orientation="horizontal")
cax2_disk.set_title(r"$r_{\mathrm{disk}}/r_{\mathrm{vir}}$")
cax2_disk.xaxis.set_ticks_position("top")

# Formatting
for i in range(4):
    for j in range(3):
        ax2[i, j].set(xlim=(t_span[0], 2), yscale="log")


plt.savefig("./figures/KS_param_scan_SFE_SFR_mgas_mstar.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.show()
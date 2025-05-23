#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import solve_ivp
from astropy import cosmology
import scipy
import cmasher as cmr

import astropy.constants as consts
import astropy.units as u


# standard flat cosmology
H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7
Ob0 = 0.0490
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)



def halo_infall_dekel(z, mhalo):
    """halo mass inflows, from # d M_{halo} / dt (Dekel et al 2009)
    also, Carr 2023 Eq 4

    Args:
        z (_type_): _description_
        mhalo (_type_): _description_

    Returns:
        _type_: \dot{M}_{halo}
    """
    mdot = (
        0.47
        * mhalo
        * (mhalo / (1e12 * u.solMass)) ** (0.15)
        * ((1 + z) / 3) ** (2.25)
        * u.Gyr ** (-1)
    ).to(u.solMass / u.Gyr)
    try:
        if mdot <= 0:
            print("____", mhalo)
    except:
        pass

    return mdot


def halo_infall_fakhouri(z, mhalo):
    """Fakhouri + 2011

    Args:
        z (_type_): _description_
        mhalo (_type_): _description_
    """
    mean_mdot = (
        46.1
        * (u.solMass / u.yr)
        * (mhalo / (1e12 * u.solMass)) ** 1.1
        * (1.0 + 1.11 * z)
        * ((Omegam0 * (1 + z) ** 3) + Omegade0) ** 0.5
    )
    mean_mdot = mean_mdot.to(u.solMass / u.Gyr)
    return mean_mdot

def halo_infall_yung25( z,  mhalo):
    mhalo = mhalo.value
    a = 1 / (1 + z)
    log_beta_z = 2.673 - 2.075* a + 0.891*a**2
    beta_z = 10**log_beta_z
    alpha_z = 0.948 + 0.694 * a - 0.565*a**2
    expansion_rate_z = np.sqrt(Omegam0 * (1 + z)**3 + Omegade0)
    mdot = beta_z * ((mhalo / 1e12) * expansion_rate_z)**alpha_z
    return mdot * (u.solMass / u.Gyr) * 1e9
 
fig, ax = plt.subplots(figsize=(5, 4), dpi=300) 
halo_masses_theory = np.geomspace(1e7, 1e13, 100) * u.solMass
redshifts_theory = np.linspace(30, 2, 5) 
for i, z in enumerate(redshifts_theory):
    mdot_fakouri = halo_infall_fakhouri(z, halo_masses_theory).value
    mdot_yung = halo_infall_yung25( z, halo_masses_theory)
    mdot_dekel = halo_infall_dekel(z, halo_masses_theory).value
    
    
    ax.plot(
        halo_masses_theory,
        mdot_dekel/  halo_masses_theory,
        color="C0",
        alpha=0.5,
        label=f"Dekel, z={z:.1f}",
    )
    ax.plot(
        halo_masses_theory,
        mdot_fakouri/  halo_masses_theory,
        color="C1",
        alpha=0.5,
        label=f"Fakhouri, z={z:.1f}",
    )
    ax.plot(
        halo_masses_theory,
        mdot_yung /  halo_masses_theory, # to Gyr
        color="C2",
        alpha=0.5,
        label=f"Yung, z={z:.1f}",
    )
    
ax.legend(loc= "upper left", fontsize=8, ncol=2, bbox_to_anchor=(1.05, 1))
ax.set(yscale="log", xscale="log", xlabel = "$M_{halo}$ [M$_\odot$]", ylabel = r"$\dot{M}_{halo} / M_{halo}$ [Gyr$^{-1}$]")
plt.show()    

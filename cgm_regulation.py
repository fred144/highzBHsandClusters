#%%

import matplotlib.pyplot as plt
import numpy as np

def cgm_dens_profile(r, r_0, rho_0, alpha):
    """https://arxiv.org/pdf/2106.00013.pdf
    """
    return rho_0 * (r / r_0)**-alpha

r = np.geomspace(1e-2,1,100)
fig,ax = plt.subplots(1,1,dpi=300, figsize=(5,5))
rho_0 = 1e4
alpha=1.4
r_0 = 0.25
ax.plot(r, cgm_dens_profile(r, r_0, rho_0, alpha))
ax.set(xlabel="r [kpc]", ylabel=r"$\rho$ [cm$^{-3}$]", xscale="log", yscale="log")
plt.show()
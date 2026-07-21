#%%
import numpy as np
from astropy import cosmology
import h5py
import astropy.units as u
from cgm_sf_regulator_baseline import CGMRegulatorBaseline
from cgm_sf_regulator import mhalo_at_z0_fakhouri
from cgm_sf_regulator import CGMRegulator

# standard flat cosmology
H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7
Ob0 = 0.0490
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)


# ============================================================
# MULTIPLE_RUNS: population-of-halos mode (2-phase model only)
# ============================================================
# When MULTIPLE_RUNS is True, every (redshift, halo mass) grid point in
# run_2phase_model_redshift_grid is integrated N_HALOS_TO_RUN times instead
# of once. Each realization starts at a different t_init drawn from
# START_TIMES (the same START_TIMES vector is used for every halo/redshift),
# rather than the single fixed t_init=0.1 Gyr used when MULTIPLE_RUNS is
# False. This produces a "population" of halo histories per grid point, so
# a median/mean/stdev (i.e. error bars) can be computed later -- see
# `population_stats` below -- instead of a single deterministic value.
#
# Toggling this flag changes the *default* behavior of
# run_2phase_model_redshift_grid (its multiple_runs/start_times arguments
# default to these module constants), but individual calls can still
# override it.
MULTIPLE_RUNS = True
N_HALOS_TO_RUN = 5
START_TIMES = np.linspace(0.05, 0.15, N_HALOS_TO_RUN)  # Gyr, len == N_HALOS_TO_RUN


def _resolve_start_times(multiple_runs, start_times):
    """Validate/normalize the t_init vector for one grid function call."""
    if not multiple_runs:
        return np.array([0.1])  # original single-run behavior, unchanged
    t_inits = np.atleast_1d(np.asarray(start_times, dtype=float))
    if len(t_inits) != N_HALOS_TO_RUN:
        raise ValueError(
            f"start_times has length {len(t_inits)}, expected "
            f"N_HALOS_TO_RUN={N_HALOS_TO_RUN}. Update N_HALOS_TO_RUN or "
            "pass a start_times vector of matching length."
        )
    return t_inits


def population_stats(data, axis=-1):
    """
    Collapse the realization axis produced by MULTIPLE_RUNS into summary
    statistics, e.g. for error bars on a grid-of-models comparison plot.

    Parameters
    ----------
    data : array-like
        An output array/dataset from a MULTIPLE_RUNS grid, with the
        realization axis last (as written).
    axis : int, optional
        Axis to collapse (default -1, the realization axis).

    Returns
    -------
    dict with "median", "mean", "std" arrays (realization axis removed).
    """
    data = np.asarray(data)
    return {
        "median": np.median(data, axis=axis),
        "mean": np.mean(data, axis=axis),
        "std": np.std(data, axis=axis),
    }


def run_baseline_model_redshift_grid(observe_at, mhalos, write_to_file=None):
    """
    Runs the baseline CGM regulator model over a grid of redshifts and halo masses.

    For each redshift in `observe_at`, and for each corresponding halo mass in `mhalos`,
    this function computes the stellar mass, halo mass, and ISM mass using the CGMRegulatorBaseline model.
    The stellar-to-halo mass ratio (SMHM) is calculated and stored for each redshift and halo mass.

    Optionally, results can be saved to an HDF5 file.

    Parameters
    ----------
    observe_at : array-like
        List or array of redshifts at which to observe the halos.
    mhalos : array-like
        2D array (or list of lists) of halo masses. Each sub-array corresponds to the halo masses at a given redshift.
    write_to_file : str or None, optional
        Path to an HDF5 file to write the results. If None, results are not saved to file.

    Returns
    -------
    smhm_out : list of numpy.ndarray
        List of arrays containing the stellar-to-halo mass ratio for each redshift and halo mass.
    zsims : list
        List of redshifts for which the model was run.

    Notes
    -----
    - The function prints progress and intermediate results to stdout.
    - The output file (if specified) contains datasets for redshifts, SMHM, input halo masses, observed halo masses,
      observed stellar masses, and observed ISM masses.
    """
    # declare a 2d array to store the results
    # each row is a different redshift
    # each column is a different halo mass
    smhm_out = []
    zsims = []
    mhalo_obs_out = []
    mstar_obs_out = []
    mism_obs_out = []
    
    sfr_obs_out = []
    m_metals_obs_out = []

    for zidx, z_obs in enumerate(observe_at):
        t_init = 0.1  # Gyr
        # get the final time given the observed redshift
        t_final = LCDM.age(z_obs).value  # Gyr
        print("** t_final = {:.2f}".format(t_final))
        t_span = (t_init, t_final)  # span of the integration

        mhalo_obs = []
        mstar_obs = []
        mism_obs = []

       
        sfr_obs = []
        m_metals_obs = []


        # now, get the mass of the halo at z0 which is what we put into the integrator
        for midx, mhalo in enumerate(mhalos[zidx]):
            # we want to observe mhalo at z_obs, so we have to know its z=0 value for the function below
            mhalo_z0 = mhalo_at_z0_fakhouri(mhalo, z_obs) * u.Msun
            print("")
            print(
                "* observing halo with mass {:.2e} at z = {:.2f}".format(mhalo, z_obs)
            )
            print("** mass of halo at z=0 is {:.2e}".format(mhalo_z0))
            gridmodel = CGMRegulatorBaseline(
                mhalo_z0,
                t_span,
                cooling_dynamic_time_norm=1,
                updated_halo_infall=False,
                updated_loadings=False,
                updated_SF_law=False,
                
            )

            run = gridmodel.run_halo()
            results = gridmodel.get_results()
            derived = gridmodel.get_derived_quantities()
            m_star = results["m_star"][-1]
          
            m_halo = results["m_halo"][-1]
            m_metals = results["m_metals"][-1]

            print("** final halo mass = {:.2e}".format(m_halo))

            mhalo_obs.append(m_halo)
            mstar_obs.append(m_star)
           
            mism_obs.append(results["m_gas"][-1])
            sfr_obs.append(derived["dot_m_sfr"][-1])
            m_metals_obs.append(m_metals)


        mhalo_obs = np.array(mhalo_obs)
        mstar_obs = np.array(mstar_obs)
        mism_obs = np.array(mism_obs)
    

        # now put it into the output arrays
        mhalo_obs_out.append(mhalo_obs)
        mstar_obs_out.append(mstar_obs)

        mism_obs_out.append(mism_obs)
        sfr_obs_out.append(sfr_obs)
        m_metals_obs_out.append(m_metals_obs)


        smhm = mstar_obs / (mhalo_obs * (Ob0 / Omegam0))

        smhm_out.append(smhm)
        zsims.append(z_obs)

        # save as we go through the redshifts
        print(smhm_out)

    if write_to_file is not None:
        # write to hdf5 file
        smhm_out = np.array(smhm_out)
        zsims = np.array(zsims)
        out_file = h5py.File(write_to_file, "w")
        out_file.create_dataset("redshifts", data=zsims)
        out_file.create_dataset("SMHM", data=smhm_out)
        out_file.create_dataset("Mhalo", data=mhalos)
        out_file.create_dataset("Mhalo_obs", data=mhalo_obs_out)
        # now do star mass, ism mass, cgm hot and cold masses
        out_file.create_dataset("Mstar_obs", data=mstar_obs_out)
        out_file.create_dataset("MISM_obs", data=mism_obs_out)

        out_file.create_dataset("SFR_obs", data=sfr_obs_out)
        out_file.create_dataset("MMetals_obs", data=m_metals_obs_out)

        out_file.close()

    return smhm_out, zsims


def run_2phase_model_redshift_grid(
    observe_at,
    mhalos,
    write_to_file=None,
    multiple_runs=MULTIPLE_RUNS,
    start_times=START_TIMES,
    **kwargs,
):
    """
    Runs the 2-phase CGM regulator model over a grid of redshifts and halo
    masses.

    If `multiple_runs` is False (default), behaves exactly as before: one
    deterministic integration per (redshift, halo mass) grid point, starting
    at t_init=0.1 Gyr. Every output array/dataset has shape (n_z, n_mhalo),
    matching the original format.

    If `multiple_runs` is True, each grid point is integrated once per entry
    in `start_times` (length must equal N_HALOS_TO_RUN), producing a
    population of halo histories per grid point. Every output array/dataset
    gains a trailing realization axis, i.e. shape (n_z, n_mhalo, n_runs).
    Use `population_stats` to collapse that axis into median/mean/stdev
    later (e.g. for error bars in an observational comparison plot).
    """
    t_inits = _resolve_start_times(multiple_runs, start_times)
    n_runs = len(t_inits)

    # declare a 2d (or, if multiple_runs, 3d) array to store the results
    # each row is a different redshift
    # each column is a different halo mass
    # (each entry along the trailing axis, if present, is a different
    #  realization / starting time)
    smhm_out = []
    zsims = []
    mhalo_obs_out = []
    mstar_obs_out = []
    mism_obs_out = []
    mcgm_hot_obs_out = []
    mcgm_cold_obs_out = []
    mbulge_obs_out = []
    sfr_obs_out = []
    m_metals_cgm_obs_out = []
    m_metals_ism_obs_out = []

    for zidx, z_obs in enumerate(observe_at):
        # get the final time given the observed redshift
        t_final = LCDM.age(z_obs).value  # Gyr
        print("** t_final = {:.2f}".format(t_final))

        mhalo_obs = []
        mstar_obs = []
        mbulge_obs = []
        mism_obs = []
        mcgm_hot_obs = []
        mcgm_cold_obs = []
        sfr_obs = []
        m_metals_cgm_obs = []
        m_metals_ism_obs = []
        # now, get the mass of the halo at z0 which is what we put into the integrator
        for midx, mhalo in enumerate(mhalos[zidx]):
            # we want to observe mhalo at z_obs, so we have to know its z=0 value for the function below
            mhalo_z0 = mhalo_at_z0_fakhouri(mhalo, z_obs) * u.Msun
            print("")
            print(
                "* observing halo with mass {:.2e} at z = {:.2f}".format(mhalo, z_obs)
            )
            print("** mass of halo at z=0 is {:.2e}".format(mhalo_z0))

            # one realization per t_init in t_inits (just [0.1] when
            # multiple_runs is False, reproducing the original behavior)
            mhalo_runs = []
            mstar_runs = []
            mbulge_runs = []
            mism_runs = []
            mcgm_hot_runs = []
            mcgm_cold_runs = []
            sfr_runs = []
            m_metals_cgm_runs = []
            m_metals_ism_runs = []

            for ridx, t_init in enumerate(t_inits):
                t_span = (t_init, t_final)  # span of the integration
                if multiple_runs:
                    print(
                        "** realization {}/{}: t_init = {:.3f} Gyr".format(
                            ridx + 1, n_runs, t_init
                        )
                    )
                gridmodel = CGMRegulator(
                    mhalo_z0, t_span, add_f_prevent_floor=1e-6, verbose=False, **kwargs
                )

                run = gridmodel.run_halo()
                results = gridmodel.get_results()
                derived = gridmodel.get_derived_quantities()
                m_halo = results["m_halo"][-1]

                print("** final halo mass = {:.2e}".format(m_halo))
                mhalo_runs.append(m_halo)
                mstar_runs.append(results["m_star"][-1])
                mbulge_runs.append(results["m_bulge"][-1])
                mism_runs.append(results["m_ism"][-1])
                mcgm_hot_runs.append(results["m_cgm_hot"][-1])
                mcgm_cold_runs.append(results["m_cgm_cold"][-1])
                sfr_runs.append(derived["dot_m_sfr"][-1])
                m_metals_cgm_runs.append(results["m_metals_cgm"][-1])
                m_metals_ism_runs.append(results["m_metals_ism"][-1])

            if multiple_runs:
                mhalo_obs.append(np.array(mhalo_runs))
                mstar_obs.append(np.array(mstar_runs))
                mbulge_obs.append(np.array(mbulge_runs))
                mism_obs.append(np.array(mism_runs))
                mcgm_hot_obs.append(np.array(mcgm_hot_runs))
                mcgm_cold_obs.append(np.array(mcgm_cold_runs))
                sfr_obs.append(np.array(sfr_runs))
                m_metals_cgm_obs.append(np.array(m_metals_cgm_runs))
                m_metals_ism_obs.append(np.array(m_metals_ism_runs))
            else:
                # single realization: keep the original scalar-per-halo shape
                mhalo_obs.append(mhalo_runs[0])
                mstar_obs.append(mstar_runs[0])
                mbulge_obs.append(mbulge_runs[0])
                mism_obs.append(mism_runs[0])
                mcgm_hot_obs.append(mcgm_hot_runs[0])
                mcgm_cold_obs.append(mcgm_cold_runs[0])
                sfr_obs.append(sfr_runs[0])
                m_metals_cgm_obs.append(m_metals_cgm_runs[0])
                m_metals_ism_obs.append(m_metals_ism_runs[0])

        mhalo_obs = np.array(mhalo_obs)
        mstar_obs = np.array(mstar_obs)
        mbulge_obs = np.array(mbulge_obs)
        mism_obs = np.array(mism_obs)
        mcgm_hot_obs = np.array(mcgm_hot_obs)
        mcgm_cold_obs = np.array(mcgm_cold_obs)
        # now put it into the output arrays
        mhalo_obs_out.append(mhalo_obs)
        mstar_obs_out.append(mstar_obs)
        mbulge_obs_out.append(mbulge_obs)
        mism_obs_out.append(mism_obs)
        mcgm_hot_obs_out.append(mcgm_hot_obs)
        mcgm_cold_obs_out.append(mcgm_cold_obs)
        sfr_obs_out.append(sfr_obs)
        m_metals_cgm_obs_out.append(m_metals_cgm_obs)
        m_metals_ism_obs_out.append(m_metals_ism_obs)

        smhm = mstar_obs / (mhalo_obs * (Ob0 / Omegam0))

        smhm_out.append(smhm)
        zsims.append(z_obs)

        # save as we go through the redshifts
        print(smhm_out)

    if write_to_file is not None:
        # write to hdf5 file
        smhm_out = np.array(smhm_out)
        zsims = np.array(zsims)
        out_file = h5py.File(write_to_file, "w")
        out_file.attrs["multiple_runs"] = multiple_runs
        out_file.attrs["n_realizations"] = n_runs
        out_file.create_dataset("start_times", data=t_inits)
        out_file.create_dataset("redshifts", data=zsims)
        out_file.create_dataset("SMHM", data=smhm_out)
        out_file.create_dataset("Mhalo", data=mhalos)
        out_file.create_dataset("Mhalo_obs", data=mhalo_obs_out)
        # now do star mass, ism mass, cgm hot and cold masses
        out_file.create_dataset("Mstar_obs", data=mstar_obs_out)
        out_file.create_dataset("MISM_obs", data=mism_obs_out)
        out_file.create_dataset("MCGM_hot_obs", data=mcgm_hot_obs_out)
        out_file.create_dataset("MCGM_cold_obs", data=mcgm_cold_obs_out)
        out_file.create_dataset("SFR_obs", data=sfr_obs_out)
        out_file.create_dataset("MMetals_cgm_obs", data=m_metals_cgm_obs_out)
        out_file.create_dataset("MMetals_ism_obs", data=m_metals_ism_obs_out)
        # also write the m_bulge
        out_file.create_dataset("MBulge_obs", data=mbulge_obs_out)

        out_file.close()

    return smhm_out, zsims


# %% smoke test: MULTIPLE_RUNS, using the same grid/params as
# smhm_vcirc_over_redshift_plotting.py's 2-phase call (Fig. 8)
#
# Reproduces that script's zobs/mhalos setup and KS parameters, but runs
# with multiple_runs=True to sanity-check shapes/timing before wiring this
# into any plotting script. Writes to a distinct file so it never clobbers
# the real cache used by smhm_vcirc_over_redshift_plotting.py.
if __name__ == "__main__":
    import os

    # -- same zbins_str / zbins_ctr construction as smhm_vcirc_over_redshift_plotting.py --
    zbins_str = [
        "0.2 < z < 0.5",
        "2.0 < z < 2.5",
        "2.5 < z < 3.0",
        "3.5 < z < 4.5",
        "5.5 < z < 6.5",
        "6.5 < z < 7.5",
        "7.5 < z < 8.5",
        "10.0 < z < 12.0",
    ]
    zbins_ctr = []
    for zb in zbins_str:
        z = zb.split("<")
        z = (float(z[0]) + float(z[2])) / 2
        zbins_ctr.append(z)
    zbins_ctr = zbins_ctr[::-1]
    zbins_ctr.append(0.01)
    zobs = zbins_ctr

    mass_bins = 16
    mhalos = np.geomspace(1e10, 1e13, mass_bins)
    mhalos = np.broadcast_to(mhalos, (len(zobs), mhalos.size)) * u.Msun

    # -- same 2-phase params --
    kappa_sfr = 0.02
    n_sfr = 1.8
    r_disk_sfr = 0.018
    eta_z = 0.6

    test_file = "./runs/TEST_multiple_runs_smhm_vcirc_2phase_grid.h5"

    print(f"MULTIPLE_RUNS={MULTIPLE_RUNS}, N_HALOS_TO_RUN={N_HALOS_TO_RUN}")
    print(f"START_TIMES={START_TIMES}")

    redshift_variation, zsims = run_2phase_model_redshift_grid(
        observe_at=zobs,
        mhalos=mhalos,
        write_to_file=test_file,
        multiple_runs=True,
        start_times=START_TIMES,
        disk_scale_length=r_disk_sfr,
        KS_n=n_sfr,
        KS_kappa_s=kappa_sfr,
        eta_z=eta_z,
    )

    with h5py.File(test_file, "r") as f:
        print("\ndatasets and shapes:")
        for key in f.keys():
            print(f"  {key}: {f[key].shape}")
        print("attrs:", dict(f.attrs))

        mstar_stats = population_stats(f["Mstar_obs"][:])
        print("\nMstar_obs population_stats shapes (realization axis collapsed):")
        for k, v in mstar_stats.items():
            print(f"  {k}: {v.shape}")

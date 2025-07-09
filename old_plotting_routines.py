
def plot_halo_diagnostics(results, derived_quant, title: str):
    t = derived_quant["sim_time"]
    z_red = derived_quant["z"]
    tvir = derived_quant["tvir"]
    dot_e_cgm_out = derived_quant["dot_e_cgm_out"]
    dot_e_cgm_cooling = derived_quant["dot_e_cgm_cooling"]
    dot_e_cgm_in = derived_quant["dot_e_cgm_in"]
    dot_e_ism_wind = derived_quant["dot_e_ism_wind"]
    dot_m_cgm_out = derived_quant["dot_m_cgm_out"]

    dot_m_cgm_hot = derived_quant["dot_m_cgm_hot"]  # hot gas cooling to cold gas
    dot_m_cgm_cold = derived_quant["dot_m_cgm_cold"]  # cold gas falling into ISM
    dot_m_cgm_in = derived_quant["dot_m_cgm_in"]
    dot_m_ism_wind = derived_quant["dot_m_ism_wind"]
    dot_m_sfr = derived_quant["dot_m_sfr"]
    dot_m_bulge = derived_quant["dot_m_bulge"]
    f_prevent = derived_quant["f_prevent"]
    f_star = derived_quant["f_star"]
    t_depletion = derived_quant["t_dep"]
    t_dynamical = derived_quant["t_dyn"]
    tcool_real = derived_quant["tcool_real"]  # purely cooling time based on CGM
    tcool_eff = derived_quant["tcool_eff"]  # added some fraction of dynamical time
    tcool_grav = derived_quant["tcool_grav"]  # added dynamical time
    t_ejection = derived_quant["t_ejection"]
    cooling_lambda = derived_quant["cooling_lambda"]

    adaptive_z = results["z"]
    t_adaptive = results["t"]
    mgas_t = results["m_ism"]

    mstar_t = results["m_star"]
    mcgm_t = results["m_cgm"]
    mcgm_hot_t = results["m_cgm_hot"]
    mcgm_cold_t = results["m_cgm_cold"]
    mmetals_t = results["m_metals"]
    mhalo_t = results["m_halo"]
    egy_t = results["egy_ism_wind"]
    egy_radloss_t = results["egy_radloss"]
    egy_eject_t = results["egy_eject"]
    egy_accrete_t = results["egy_accrete"]
    egy_cgm_t = results["egy_cgm"]
    metal_cgm_mass = results["metal_cgm_mass"]
    metal_cgm_mass_sol = results["metal_cgm_mass_sol"]

    colors = plt.get_cmap("Dark2")(np.linspace(0, 1, 8))

    fig, ax = plt.subplots(3, 3, figsize=(13, 8), dpi=300, sharex="row")
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.15, wspace=0.2)

    #   ax[0,0].plot(sol.t, gradients[0,:])
    # add minor tick marks
    ax[0].minorticks_on()
    ax[3].minorticks_on()
    ax[6].minorticks_on()

    x_axis = t
    x_axis_adaptive = t_adaptive

    ax[0].plot(x_axis, dot_e_ism_wind, label="gain from SF winds", color="b")
    ax[0].plot(x_axis, dot_e_cgm_cooling, label="CGM cooling loss", color="r")
    ax[0].plot(x_axis, dot_e_cgm_in, label=r"cosmic infall", color="g")
    ax[0].plot(x_axis, dot_e_cgm_out, label=r"CGM ejection", color="orange")

    ## make an inset for the first 0.2 Gyr of the simulation in the first panel
    inset0 = ax[0].inset_axes([0.05, 1.3, 1, 0.8])  # [left, bottom, width, height]
    x_axis_mask = x_axis < t.min() * 4
    inset0.plot(x_axis[x_axis_mask], dot_e_ism_wind[x_axis_mask], color="b")
    inset0.plot(x_axis[x_axis_mask], dot_e_cgm_cooling[x_axis_mask], color="r")
    inset0.plot(x_axis[x_axis_mask], dot_e_cgm_in[x_axis_mask], color="g")
    inset0.plot(x_axis[x_axis_mask], dot_e_cgm_out[x_axis_mask], color="orange")
    inset0.set(
        yscale="log",
        ylim=(
            dot_e_cgm_cooling[x_axis_mask].max() / 1e6,
            dot_e_cgm_cooling[x_axis_mask].max() * 20,
        ),
    )
    # make inset transparent
    inset0.patch.set_alpha(0.5)
    # mark_inset
    mark_inset(ax[0], inset0, loc1=3, loc2=3, fc="none", ec="k", lw=0.8, alpha=0.7)
    ax[0].set(
        yscale="log",
        ylim=(dot_e_cgm_cooling.max() / 1e14, dot_e_cgm_cooling.max() * 20),
        ylabel=r"$\dot{E} ~ [{\rm erg \ Gyr^{-1}}]$",
    )
    ax[0].legend(
        ncols=2,
        loc="lower right",
        frameon=False,
        fontsize=7,
    )

    # do the same for the second panel
    inset1 = ax[1].inset_axes([0.05, 1.3, 1, 0.8])  # [left, bottom, width, height]
    x_axis_mask = x_axis < t.min() * 4
    inset1.plot(x_axis[x_axis_mask], dot_m_ism_wind[x_axis_mask], color="b")
    inset1.plot(
        x_axis[x_axis_mask],
        dot_m_cgm_cold[x_axis_mask],
        color="tab:blue",
        ls="--",
        label="cool CGM",
    )
    inset1.plot(
        x_axis[x_axis_mask],
        dot_m_cgm_hot[x_axis_mask],
        color="tab:red",
        ls="--",
        label="hot CGM",
    )
    inset1.plot(x_axis[x_axis_mask], dot_m_cgm_in[x_axis_mask], color="g")
    inset1.plot(x_axis[x_axis_mask], dot_m_cgm_out[x_axis_mask], color="orange")
    inset1.set(
        yscale="log",
        ylim=(
            dot_m_cgm_in[x_axis_mask].max() / 1e4,
            dot_m_cgm_in[x_axis_mask].max() * 2,
        ),
    )
    # mark_inset
    mark_inset(ax[1], inset1, loc1=3, loc2=3, fc="none", ec="k", lw=0.8, alpha=0.7)
    inset1.patch.set_alpha(0.5)

    ### plots of relevant timescales
    # fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    inset2 = ax[2].inset_axes([0.05, 1.2, 0.95, 0.9])
    inset2.plot(t, t_depletion, label=r"$t_{\rm depletion}$", color=colors[0])
    inset2.plot(t, t_dynamical, label=r"$t_{\rm dynamical}$", color=colors[1])
    inset2.plot(t, tcool_real, label=r"$t_{\rm cool}$", color=colors[2])
    inset2.plot(
        t,
        tcool_grav,
        label=r"$ t_{\rm cool} + t_{\rm dynamical}$",
        color="tab:blue",
        ls="--",
    )

    inset2.plot(t, t_ejection, label=r"$t_{\rm ejection}$", color=colors[5], ls="--")
    inset2.set(xlabel="time [Gyr]", ylabel="timescales [Gyr]", yscale="log")
    inset2.legend(frameon=False, fontsize=9, loc="upper right", ncols=2)
    # place the x axis on the top
    inset2.xaxis.set_ticks_position("top")
    inset2.xaxis.set_label_position("top")
    # add minor ticks
    inset2.minorticks_on()

    run_text = (
        r"{:}"
        "\n"
        "ran from $z = {:.2f} - {:.2f}$ ({:.2f} - {:.2f} Gyr)"
        "\n"
        r"and $M_{{\rm halo}}(z = {:.1f}) = {:.0e} ~ M_\odot$, $M_{{\rm halo}}(z = {:.1f}) = {:.0e} ~ M_\odot$, $M_{{\rm halo}}(z = {:.1f}) = {:.0e} ~ M_\odot$".format(
            title,
            adaptive_z[0].value,
            adaptive_z[-1].value,
            t[0],
            t[-1],
            adaptive_z[0].value,
            mhalo_t[0],
            adaptive_z[-1].value,
            mhalo_t[-1],
            0,
            mhalo_at_z0(mhalo_t[-1] * u.Msun, adaptive_z[-1]),
        )
    )
    # put text on the corner left
    ax[6].text(
        0,
        -0.25,
        run_text,
        transform=ax[6].transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    ax[1].plot(x_axis, dot_m_ism_wind, label="ISM winds", color="b")
    # ax[1].plot(x_axis, dot_m_cgm_cool, label="cooling gas", color="r")
    ax[1].plot(
        x_axis,
        dot_m_cgm_hot,
        label="hot CGM",
        color="tab:red",
        ls="--",
    )
    ax[1].plot(
        x_axis,
        dot_m_cgm_cold,
        label="cool CGM",
        color="tab:blue",
        ls="--",
    )
    ax[1].plot(x_axis, dot_m_cgm_in, label=r"cosmic infall", color="g")
    ax[1].plot(x_axis, dot_m_cgm_out, label=r"CGM ejection", color="orange")
    ax[1].plot(x_axis, dot_m_sfr, label="SFR", color=colors[0], ls=":")
    ax[1].legend(
        ncols=2,
        # bbox_to_anchor=(0.5, 1.45),
        loc="lower right",
        frameon=False,
        # title=run_text,
        fontsize=7,
    )
    ax[1].set(
        ylim=(dot_m_cgm_in.max() / 1e6, dot_m_cgm_in.max() * 10),
        # xlim=(t[0] * 0.9, t[-1]),
        yscale="log",
        ylabel=r" $ \dot{M} ~ [{\rm M_{\odot} \ Gyr^{-1}}] $",
    )

    ax[2].plot(x_axis_adaptive, mstar_t, label=r"$M_{\star}$", color=colors[0])
    ax[2].plot(x_axis_adaptive, mgas_t, label=r"$M_{\rm ISM}$", color=colors[1])
    ax[2].plot(x_axis_adaptive, mcgm_t, label=r"$M_{\rm CGM}$", color="violet")
    ax[2].plot(
        x_axis_adaptive,
        mcgm_hot_t,
        label=r"$M_{\rm CGM, hot}$",
        color="tab:red",
        ls="--",
    )
    ax[2].plot(
        x_axis_adaptive,
        mcgm_cold_t,
        label=r"$M_{\rm CGM, cold}$",
        color="tab:blue",
        ls="--",
    )
    ax[2].plot(x_axis_adaptive, mhalo_t, label=r"$M_{\rm halo}$", color="k")
    ax[2].legend(ncols=2, frameon=False)
    ax[2].set_ylabel(r"M$_{\odot}$")
    ax[2].set_ylim(1e6, mhalo_t[-1] * 4)
    ax[2].set_yscale("log")

    # plot the energy terms
    ax[3].plot(x_axis, egy_t, label=r"$E_{\rm ISM, winds}$", color=colors[0])
    ax[3].plot(x_axis, egy_radloss_t, label=r"$E_{\rm CGM, cooling}$", color=colors[3])
    ax[3].plot(x_axis, egy_accrete_t, label=r"$E_{\rm CGM, accrete}$", color=colors[1])
    ax[3].plot(x_axis, egy_eject_t, label=r"$E_{\rm eject}$", color=colors[2])

    ax[3].plot(x_axis, egy_cgm_t, label=r"$E_{\rm CGM}$", color="k")
    ax[3].set(yscale="log", ylabel=r"$E ~ [{\rm erg}]$", ylim=(1e52, egy_t[-1] * 10))
    ax[3].legend(ncols=2, frameon=False, loc="lower right", fontsize=9)
    ax[4].plot(x_axis, f_star)
    ax[4].set_yscale("log")
    ax[4].set_ylabel(r"halo scale $f_\star [M_\star / M_{\rm halo} f_{b}]$")
    # ax[4].set_ylim(1e-3, 1)

    ax[5].plot(x_axis, f_prevent)
    ax[5].set_ylim(0, 1.25)
    ax[5].set_ylabel(r"$f_{\rm prevent}$")

    mu = 0.6 * consts.m_p
    kb = consts.k_B
    Teff = ((egy_cgm_t * u.erg) / (mcgm_cold_t * u.solMass) * (mu / kb)).to(u.K)
    # print(Teff)
    ax[6].plot(x_axis_adaptive, Teff, label=r"CGM $T_{\rm eff}$")
    ax[6].plot(x_axis, tvir, label="Tvir")
    ax[6].set_ylabel("T [K]")
    ax[6].set_yscale("log")

    # ax[6].set_ylim(1e4, 1e12)
    ax[6].legend(ncols=2, frameon=False)

    # plot the metals
    ax[7].plot(x_axis, metal_cgm_mass_sol, label="CGM metallicity")
    ax[7].set_ylabel(r"CGM metallicity $[Z_{\odot}]$")
    ax[7].set_yscale("log")
    ax[7].set_xlabel("t [Gyr]")

    # plot the cooling function
    ax[8].plot(x_axis, np.log10(cooling_lambda))
    ax[8].set(
        ylabel=r"log $\Lambda [{\rm erg \:cm^3 \:s^{-1}}]$",
        ylim=(np.log10(cooling_lambda).max() - 5, np.log10(cooling_lambda).max() + 1),
    )

    # make a twin redshift axis for the top row, using z
    # get the current x axis labels of the first row and their
    x_axis_tick_labels = ax[0].get_xticks()
    x_axis_tick_labels = np.array(x_axis_tick_labels)[2:-1]
    # prepend the lowest time value to the beginning of the array
    x_axis_tick_labels = np.insert(x_axis_tick_labels, 0, t[0])

    print(x_axis_tick_labels)

    for i in range(3):
        ax2 = ax[i].twiny()
        # make sure the ranges are the same
        ax2.plot(x_axis, dot_e_cgm_out, color="k", alpha=0)
        ax2.set_xlim(t[0] * 0.9, t[-1])
        t_ticks = x_axis_tick_labels  # [8,  5,4, 3, 2, 1,  0.001]
        # z_ticks = np.geomspace(np.max(adaptive_z).value, np.min(adaptive_z).value, 5)
        # t_ticks = LCDM.age(z_ticks).value  # Convert redshift to corresponding time
        z_ticks = cosmology.z_at_value(LCDM.age, t_ticks * u.Gyr).value
        ax2.set_xticks(x_axis_tick_labels)
        ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
        ax2.set_xlim(t[0] * 0.9, t[-1])
        ax2.set_xlabel(r"$z$")

    plt.show()
# Compute, report and deposit the kondo-thermo record: impurity entropy and susceptibility of the
# symmetric Anderson model as U/Γ is raised, and the Kondo scale read off the flow.
#
#     julia --project=<env> examples/kondo-thermo/run.jl
#
# `<env>` needs WilsonNRG, ParamIO, DataVault, SweepRunner, Pinax, Plots and Archeion. The record
# goes to the registry the binding names (QAtlasHub/archeion-demo, cloned beside this repository).
# BINDING and REMOTE override the binding file and `publish`'s `remote` (`pr`, `push`, `local`),
# for a rehearsal against a scratch registry.

using ParamIO, DataVault, SweepRunner, Pinax, Plots, Archeion
using WilsonNRG
ENV["GKSwstype"] = "100"
gr()

const REPO = dirname(dirname(@__DIR__))
const STUDY = @__DIR__
const CONFIG = joinpath(STUDY, "config.toml")
const BINDING = get(
    ENV, "BINDING", joinpath(REPO, ".registry", "bindings", "kondo-thermo.toml")
)
const REMOTE = Symbol(get(ENV, "REMOTE", "pr"))
const TITLE = "Kondo screening in the symmetric Anderson model: NRG thermodynamics across U/Γ"

# Wilson's definition of the Kondo temperature: T_K·χ_imp(T_K) = 0.0701 (Wilson 1975).
const WILSON_TCHI = 0.0701

# The same scheme for every point; only U changes. EnergyCut, not KeepN: a fixed KeepN
# under-resolves the impurity-doubled run (test/gates/test_thermo.jl).
const ALG = NRGAlgorithm(;
    discretization=WilsonLog(2.0), symmetry=U1U1(), truncation=EnergyCut(6.0), nsites=18
)

function work(key)
    r = key.params["model.u_over_gamma"]
    Γ = key.params["model.gamma"]
    U = r * Γ
    th = thermodynamics(AndersonModel(; U, εd=-U / 2, Γ, D=1.0), ALG; betabar=1.0)
    return Dict{String,Any}(
        "u_over_gamma" => r,
        "gamma" => Γ,
        "T" => collect(th.T),
        "S_imp" => collect(th.S_imp),
        "Tchi_imp" => collect(th.Tχ_imp),
    )
end

# Haldane's Kondo scale for the symmetric model, valid for U ≫ Γ (flat band, half-width 1).
haldane_tk(U, Γ) = sqrt(U * Γ / 2) * exp(-π * U / (8Γ) + π * Γ / (2U))

# The temperature at which T·χ_imp last falls through 0.0701, interpolated in log T; `nothing` if
# the flow never reaches it (the chain is too short). At U = 0 there is no local moment, and the
# crossing is the free orbital emptying out, not a Kondo scale: `nothing` there too.
wilson_tk(x) = x["u_over_gamma"] > 0 ? wilson_tk(x["T"], x["Tchi_imp"]) : nothing
function wilson_tk(T, tchi)
    order = sortperm(T)                       # low T first
    T, tchi = T[order], tchi[order]
    i = findfirst(>=(WILSON_TCHI), tchi)
    (i === nothing || i == 1) && return nothing
    a, b = log(T[i - 1]), log(T[i])
    f = (WILSON_TCHI - tchi[i - 1]) / (tchi[i] - tchi[i - 1])
    return exp(a + f * (b - a))
end

function recipe(pairs)
    d = sort([p[2] for p in pairs]; by=x -> x["u_over_gamma"])
    label(x) = "U/Γ = $(x["u_over_gamma"])"

    fchi = plot(;
        xscale=:log10, xlabel="T / D", ylabel="T χ_imp", size=(640, 360), legend=:topleft
    )
    fs = plot(;
        xscale=:log10, xlabel="T / D", ylabel="S_imp", size=(640, 360), legend=:topleft
    )
    for x in d
        plot!(fchi, x["T"], x["Tchi_imp"]; marker=:circle, ms=2.5, label=label(x))
        plot!(fs, x["T"], x["S_imp"]; marker=:circle, ms=2.5, label=label(x))
    end
    hline!(fchi, [1 / 4, 1 / 8]; ls=:dash, lw=1, color=:black, label="1/4, 1/8")
    hline!(fchi, [WILSON_TCHI]; ls=:dot, lw=1, color=:gray, label="0.0701 (T_K)")
    hline!(fs, [log(2), log(4)]; ls=:dash, lw=1, color=:black, label="ln 2, ln 4")

    rows = map(d) do x
        Γ, r = x["gamma"], x["u_over_gamma"]
        tw = wilson_tk(x)
        th = r > 0 ? haldane_tk(r * Γ, Γ) : nothing
        fmt(v) = v === nothing ? "—" : string(round(v; sigdigits=3))
        ratio = (tw === nothing || th === nothing) ? "—" : fmt(tw / th)
        [
            string(r),
            fmt(tw),
            fmt(th),
            ratio,
            fmt(x["Tchi_imp"][argmin(x["T"])]),
            fmt(x["S_imp"][argmin(x["T"])]),
        ]
    end

    fcol = plot(;
        xscale=:log10, xlabel="T / T_K (Wilson)", ylabel="T χ_imp", size=(640, 360)
    )
    for x in d
        tw = wilson_tk(x)
        tw === nothing && continue
        plot!(fcol, x["T"] ./ tw, x["Tchi_imp"]; marker=:circle, ms=2.5, label=label(x))
    end

    @page :kondo "Kondo screening across U/Γ" status = :trial begin
        @desc md"""
        Symmetric Anderson model ($\varepsilon_d = -U/2$, flat band of half-width $D = 1$,
        $\Gamma = 0.03$) solved by `WilsonNRG.thermodynamics`: Wilson's logarithmic
        discretization with $\Lambda = 2$, U(1)×U(1) symmetry, an energy cut of 6 and 18 chain
        sites, one run per $U/\Gamma$. Impurity quantities are the two-run difference (full minus
        bath), as in Krishna-murthy, Wilkins & Wilson (1980).

        **Scope.** One discretization, one truncation, one chain length. The lowest temperature
        reached is set by the chain length, so the largest $U/\Gamma$ may not be screened within it.
        Nothing here is z-averaged, and nothing is converged in $\Lambda$ or in the cut.
        """
        @section :flow "Entropy and susceptibility along the flow" begin
            @figure fchi caption = "T χ_imp against T. A local moment sits at 1/4, the free orbital at 1/8; screening takes it to 0. The dotted line is Wilson's 0.0701."
            @figure fs caption = "S_imp against T: ln 4 (free orbital) → ln 2 (local moment) → 0 (screened singlet)."
        end
        @section :tk "The Kondo scale" begin
            @desc md"""
            $T_K$ is read where $T\chi_{\rm imp}$ falls through 0.0701 (Wilson's definition) and
            compared with Haldane's $T_K = \sqrt{U\Gamma/2}\,e^{-\pi U/8\Gamma + \pi\Gamma/2U}$,
            which holds for $U \gg \Gamma$. The two definitions differ by an O(1) factor; what is
            tested is that **the ratio stays roughly constant** as $T_K$ falls by orders of
            magnitude. At $U/\Gamma = 2$ the model is mixed-valent and Haldane's form does not apply.
            """
            @table rows header = [
                "U/Γ",
                "T_K (Wilson)",
                "T_K (Haldane)",
                "ratio",
                "Tχ at lowest T",
                "S at lowest T",
            ] caption = "T_K from the flow against the analytic scale"
            @figure fcol caption = "T χ_imp against T / T_K. Where the model is in the Kondo regime the curves should fall on one universal curve."
        end
    end
    return nothing
end

function main()
    vault = DataVault.Vault(CONFIG; run="sweep", outdir=joinpath(STUDY, "out"))
    keys = ParamIO.expand(ParamIO.load(CONFIG))
    res = SweepRunner.run!(work, vault, keys; opts=RunOpts(; workers=:sequential))
    @info "swept" computed = res.done skipped = res.skipped failed = res.err
    published = Archeion.publish(
        vault,
        recipe;
        binding=BINDING,
        title=TITLE,
        out=joinpath(STUDY, "out", "report", "kondo-thermo"),
        status=:trial,
        source_repo=REPO,
        study="sweep",
        tags=["nrg", "kondo", "anderson-model", "thermodynamics", "WilsonNRG.jl"],
        question="Does the Kondo scale read off the NRG flow follow Haldane's formula as U/Γ grows?",
        remote=REMOTE,
    )
    @info "published" points = published.n record = published.record revision =
        published.rev pr = published.pr
    return nothing
end

isinteractive() || main()

# ===========================================================================
#  Impurity spectral function A(ω) — T=0 patching (Bulla–Hewson–Pruschke, PRB 57,
#  10287 (1998); review: Bulla–Costi–Pruschke, RMP 80, 395 (2008), §III.B).
#
#  The impurity creation operator d†_σ is propagated along the flow
#  ([`propagate_operator`](@ref)). At each shell N the transitions out of the
#  ground state |0⟩ — `d†|0⟩` (addition, ω>0) and `d|0⟩` (removal, ω<0) — give
#  poles at physical energy (E_r − E_0)·ω_N with weight |⟨r|d†|0⟩|². Each shell
#  contributes only in its resolution WINDOW (rescaled excitation in [w, w√Λ], so
#  the physical windows tile), then poles are broadened with a log-Gaussian.
#
#  Per-spin convention: A_σ(ω) with ∫A_σ dω = ⟨{d_σ, d†_σ}⟩ = 1 (one spin channel).
#  Accuracy: the log-Gaussian smears sharp features (the standard NRG artifact);
#  z-averaging / smaller Λ / smaller b sharpen the shape (later). The robust,
#  gate-able properties are the sum rule and particle–hole symmetry.
# ===========================================================================

# log-Gaussian kernel: ∫ over ω of this (for a single pole) returns the weight w.
function _log_gaussian(ω::Real, ωp::Real, w::Real, b::Real)
    (ωp != 0 && ω != 0 && sign(ω) == sign(ωp)) || return 0.0
    return w / (b * sqrt(π) * abs(ω)) * exp(-(log(abs(ω / ωp)) / b)^2)
end

# (physical energy, weight) poles of A_↑(ω) from ground-state d†_↑/d_↑ transitions, windowed.
function _spectral_poles(model::AbstractImpurityModel, alg::NRGAlgorithm, window::Real)
    chain = wilson_chain(alg.discretization, model, alg.nsites)
    sym = alg.symmetry
    Λ = alg.discretization.Λ
    sqrtΛ = sqrt(Λ)
    whi = window * sqrtΛ                                  # rescaled-energy window [window, window√Λ]
    st = impurity_init(model, sym, chain)
    O = deepcopy(st.F)                                   # d†_σ = impurity creation (Anderson init)
    poles = Tuple{Float64,Float64}[]
    for n in bath_sites_in_init(model):(alg.nsites - 1)
        coupling = n == 0 ? bath_coupling(model) : chain.hopping[n]
        rescale = n == 0 ? 1.0 : sqrtΛ
        diag = diagonalize_blocks(
            add_site(st, sym; coupling, rescale, onsite=chain.onsite[n + 1]), sym
        )
        plan = truncation_plan(diag.vals, alg.truncation, sym)
        gqn = argmin(qn -> minimum(diag.vals[qn][plan[qn]]), collect(keys(plan)))
        i0 = plan[gqn][argmin(diag.vals[gqn][plan[gqn]])]
        E0 = diag.vals[gqn][i0]
        ωN = shell_scale(alg.discretization, n)
        O = propagate_operator(O, diag, plan, sym)
        row = findfirst(==(i0), plan[gqn])
        addk = (gqn[1], gqn[2], 1)                        # d†_↑ : |0⟩ → (Q+1, D+1)   (ω>0)
        if haskey(O, addk)
            tgt = (gqn[1] + 1, gqn[2] + 1)
            for (j, r) in enumerate(plan[tgt])
                x = diag.vals[tgt][r] - E0
                window ≤ x < whi && push!(poles, (x * ωN, O[addk][j, row]^2))
            end
        end
        remk = (gqn[1] - 1, gqn[2] - 1, 1)                # d_↑ = (d†_↑)† : |0⟩ → (Q−1, D−1) (ω<0)
        if haskey(O, remk)
            src = (gqn[1] - 1, gqn[2] - 1)
            for (j, r) in enumerate(plan[src])
                x = diag.vals[src][r] - E0
                window ≤ x < whi && push!(poles, (-x * ωN, O[remk][row, j]^2))
            end
        end
        st = update_operators(diag, plan, sym)
    end
    return poles
end

# default log-spaced ± frequency grid spanning the flow's scales
function _default_omega(model, alg; nω=240)
    D = hasproperty(model, :D) ? model.D : 1.0
    lo = D * alg.discretization.Λ^(-(alg.nsites) / 2) / 2     # below the smallest shell scale
    hi = 2 * D
    pos = exp10.(range(log10(lo), log10(hi); length=nω))
    return vcat(-reverse(pos), pos)
end

"""
    spectral(::BHP, model::AndersonModel, alg; b = 0.6, window = 0.7, ω = nothing) -> (; ω, A)

Zero-temperature impurity spectral function `A_σ(ω)` (per spin) by Bulla–Hewson–Pruschke
patching: propagate `d†_σ` along the NRG flow, collect windowed ground-state transitions,
broaden with a log-Gaussian of width `b`. `ω` defaults to a log-spaced ± grid; pass your own.

`U = 0` recovers the resonant level `A(ω) = (Γ/π)/(ω²+Γ²)` ([`resonant_level_spectral`](@ref))
up to log-Gaussian broadening; `∫A dω = 1` and `A(ω) = A(−ω)` (at the symmetric point) hold.
"""
function spectral(
    ::BHP, model::AndersonModel, alg::NRGAlgorithm; b::Real=0.6, window::Real=0.7, ω=nothing
)
    alg.symmetry isa U1U1 ||
        throw(EngineUnimplemented("BHP spectral needs U1U1 (got $(typeof(alg.symmetry)))"))
    ωs = ω === nothing ? _default_omega(model, alg) : collect(float.(ω))
    poles = _spectral_poles(model, alg, window)
    A = [sum(_log_gaussian(w_ω, ωp, w, b) for (ωp, w) in poles; init=0.0) for w_ω in ωs]
    return (; ω=ωs, A)
end

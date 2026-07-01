# ===========================================================================
#  Time-dependent NRG (TDNRG) — real-time impurity dynamics after a quench.
#
#  Anders & Schiller, PRL 95, 196801 (2005): the impurity parameters are switched suddenly at
#  t=0 (H_i → H_f, sharing the Wilson chain), and the impurity observable evolves as
#       ⟨O(t)⟩ = ⟨G_i| e^{iH_f t} O e^{-iH_f t} |G_i⟩
#             = Σ_{s,s'} ⟨G_i|s⟩⟨s'|G_i⟩ e^{i(E_s - E_{s'})t} ⟨s|O|s'⟩,
#  the sum running over the complete basis of H_f eigenstates |s⟩ (energies E_s). H_i and H_f
#  share the chain but have DIFFERENT eigenbases, so |G_i⟩ is expressed in the H_f basis by the
#  shell-by-shell OVERLAP RECURSION  S_n = V_f^⊤ (S_{n-1} ⊗ I_site) V_i  (block-diagonal in the
#  conserved (Q,D)). This is EXACT at keep-all (short chains) — Anders–Schiller's complete-basis
#  time evolution in its exact limit; the truncated discarded-state complete-basis sum (scalable
#  to long chains, the full TDNRG) is a future increment.
#
#  Energies: the last-shell rescaled eigenvalue differences are physical up to the shell factor;
#  for the keep-all last shell, physical ΔE = ΔE_rescaled · Λ^{-(nsites-1)/2} (the same Λ^{(N)/2}
#  scaling the free-fermion gate verifies), so the physical time evolution is recovered.
# ===========================================================================

using LinearAlgebra: I

# one overlap-recursion step:  S_n[qn] = V_f[qn]^⊤ (S_{n-1} ⊗ I_site) V_i[qn], block-diagonal in qn.
# At keep-all the two runs share the product-basis segmentation (same kept dims), so the embedded
# S_{n-1} sits block-diagonally on the parent-block ranges (identity on the new site).
function _overlap_step(S, df::U1U1Diag, di::U1U1Diag)
    Snew = Dict{NTuple{2,Int},Matrix{Float64}}()
    for (qn, segs) in df.seg
        Vf = df.vecs[qn]
        Vi = di.vecs[qn]
        M = zeros(Float64, size(Vf, 1), size(Vi, 1))
        for (p, s, r) in segs
            haskey(S, p) && (M[r, r] = S[p])          # S_{n-1}[parent p] ⊗ I on site s
        end
        Snew[qn] = transpose(Vf) * M * Vi
    end
    return Snew
end

"""
    quench_dynamics(initial::AndersonModel, final::AndersonModel, alg; times) -> (; t, nd)

Real-time impurity occupation `⟨n_d(t)⟩` after a sudden quench of the impurity parameters from
`initial` to `final` at `t=0` — time-dependent NRG (Anders & Schiller, PRL 95, 196801 (2005)).
The system starts in the ground state of `initial` and evolves under `final`; `times` are the
(physical) times at which `⟨n_d⟩` is returned.

`initial` and `final` must share the bath (`Γ`, `D`) so the Wilson chain is common; they may
differ in `εd` and/or `U`. `U1U1` only. **Exact at keep-all** — evaluate on a short chain
(`alg.nsites` ≲ 7); the truncation in `alg` is not used (the truncated discarded-state
complete-basis sum for long chains is a planned extension). Checks: `⟨n_d(0)⟩ = ⟨n_d⟩` of
`initial` ([`occupation`](@ref)); the long-time signal relaxes toward the `final` equilibrium up
to finite-chain recurrences. Assumes a non-degenerate `initial` ground state.
"""
function quench_dynamics(
    initial::AndersonModel, final::AndersonModel, alg::NRGAlgorithm; times
)
    alg.symmetry isa U1U1 || throw(
        EngineUnimplemented("quench_dynamics needs U1U1 (got $(typeof(alg.symmetry)))")
    )
    (initial.Γ == final.Γ && initial.D == final.D) || throw(
        ArgumentError(
            "quench_dynamics: initial and final must share the bath (Γ, D) — the Wilson chain is common",
        ),
    )
    chain = wilson_chain(alg.discretization, final, alg.nsites)
    Λ = alg.discretization.Λ
    sqrtΛ = sqrt(Λ)
    stf = impurity_init(final, U1U1(), chain)
    sti = impurity_init(initial, U1U1(), chain)
    S = Dict(qn => Matrix{Float64}(I, length(ev), length(ev)) for (qn, ev) in stf.E)
    nd = Dict(
        (1, 1) => fill(1.0, 1, 1), (1, -1) => fill(1.0, 1, 1), (2, 0) => fill(2.0, 1, 1)
    )
    keepall = KeepN(typemax(Int))
    for n in bath_sites_in_init(final):(alg.nsites - 1)
        coupling = n == 0 ? bath_coupling(final) : chain.hopping[n]
        rescale = n == 0 ? 1.0 : sqrtΛ
        df = diagonalize_blocks(
            add_site(stf, U1U1(); coupling, rescale, onsite=chain.onsite[n + 1]), U1U1()
        )
        di = diagonalize_blocks(
            add_site(sti, U1U1(); coupling, rescale, onsite=chain.onsite[n + 1]), U1U1()
        )
        S = _overlap_step(S, df, di)
        pf = truncation_plan(df.vals, keepall, U1U1())
        pit = truncation_plan(di.vals, keepall, U1U1())
        nd = propagate_observable(nd, df, pf, U1U1())
        stf = update_operators(df, pf, U1U1())
        sti = update_operators(di, pit, U1U1())
    end
    # global H_i ground state (energies are ground-subtracted each shell ⇒ min ≈ 0)
    gE = minimum(minimum(v) for v in values(sti.E))
    blkg = first(qn for (qn, ev) in sti.E if any(e -> isapprox(e, gE; atol=1.0e-9), ev))
    gi = findfirst(e -> isapprox(e, gE; atol=1.0e-9), sti.E[blkg])
    a = S[blkg][:, gi]                                    # ⟨s_f | G_i⟩ for H_f states s in blkg
    E = stf.E[blkg]                                       # H_f rescaled energies (same block)
    O = get(nd, blkg, zeros(length(a), length(a)))        # n_d in the H_f eigenbasis
    scale = Λ^(-(alg.nsites - 1) / 2)                     # rescaled ΔE → physical
    ts = collect(float.(times))
    ndt = [
        real(
            sum(
                a[s] * a[sp] * cis((E[s] - E[sp]) * scale * t) * O[s, sp] for
                s in eachindex(a), sp in eachindex(a)
            ),
        ) for t in ts
    ]
    return (; t=ts, nd=ndt)
end

# ===========================================================================
#  Axis 2 — bath discretization.  Δ(ω) ↦ Wilson chain {εₙ, ξₙ}.
#  Each formulation is one method of `wilson_chain`. This is the cleanest,
#  fully deterministic axis (a closed-form recursion for the log grid), so it is
#  implemented and gated first.
# ===========================================================================

"""
    wilson_chain(disc::AbstractDiscretization, model::AbstractImpurityModel, nsites) -> WilsonChain

Discretize the bath of `model` into a length-`nsites` Wilson chain under
formulation `disc`. Dispatches on `disc` (the paper) and `model` (the band).
"""
function wilson_chain end

"""
    wilson_chain(disc::WilsonLog, model, nsites)

Wilson logarithmic discretization of a particle–hole-symmetric flat band.
On-site energies vanish (`εₙ = 0`) and the dimensionless hoppings are the
closed form (KWW 1980 Eq. 2.15; Bulla, Costi & Pruschke, RMP 80, 395 (2008), Eq. 32)

    ξₙ = (1 + Λ⁻¹)/2 · (1 − Λ^{−n−1}) / √[(1 − Λ^{−2n−1})(1 − Λ^{−2n−3})],   n = 0,1,…

which decreases monotonically to the asymptote [`asymptotic_hopping`](@ref)
`(1 + Λ⁻¹)/2`. The `Λ^{−n/2}` energy scale is applied by the driver, not here.
"""
function wilson_chain(disc::WilsonLog, ::AbstractImpurityModel, nsites::Integer)
    nsites ≥ 1 || throw(ArgumentError("wilson_chain: nsites must be ≥ 1 (got $nsites)"))
    Λ = disc.Λ
    onsite = zeros(Float64, nsites)
    hopping = Vector{Float64}(undef, nsites)
    @inbounds for n in 0:(nsites - 1)
        num = (1 + Λ^(-1)) / 2 * (1 - Λ^(-n - 1))
        den = sqrt((1 - Λ^(-2n - 1)) * (1 - Λ^(-2n - 3)))
        hopping[n + 1] = num / den
    end
    return WilsonChain(onsite, hopping, disc)
end

"""
    asymptotic_hopping(disc::WilsonLog) -> Float64

Large-`n` limit `ξₙ → (1 + Λ⁻¹)/2` of the logarithmic Wilson chain. Used as the
deterministic faithfulness anchor for the discretization (KWW 1980 / Bulla 2008).
"""
asymptotic_hopping(disc::WilsonLog) = (1 + disc.Λ^(-1)) / 2

"""
    shell_scale(disc::AbstractDiscretization, n) -> Float64

Characteristic energy scale `ωₙ` of NRG shell `n` (last site `f_n`), in units of the band
half-width `D`: the standard NRG ladder `ωₙ = (1+Λ⁻¹)/2 · Λ^{-(n-1)/2}` (Bulla, Costi &
Pruschke, RMP 80, 395 (2008), Eq. 3.9). A function of `Λ` and `n` only — shared by every
logarithmic discretization at fixed `Λ` (the z-averaging schemes refine the discretized
*band*, not this shell energy ladder; the O(1) prefactor is absorbed into the convention
`Tₙ = ωₙ / β̄`, `β̄ ~ 1`). Absolute spectral-function assembly for the z-shifted chains
(which would need the z-dependent scale) is not claimed here — see [`band_dos`](@ref).
"""
shell_scale(disc::AbstractDiscretization, n::Integer) =
    (1 + disc.Λ^(-1)) / 2 * disc.Λ^(-(n - 1) / 2)

"""
    hybridization(model::AndersonModel, ω) -> Float64

Hybridization function `Δ(ω) = πρ|V(ω)|²` of the bath. For the flat band of
half-width `D` it is the constant `Γ` for `|ω| < D` and `0` outside — the input
the discretization integrates over.
"""
hybridization(model::AndersonModel, ω::Real) = abs(ω) < model.D ? model.Γ : 0.0

# ===========================================================================
#  Impurity self-energy Σ(ω) from the Green's function — two formulations to
#  compare (Axis 4b). Σ is notoriously method-sensitive, so the package makes the
#  choice explicit (dispatch) with a robust default ([`default_self_energy_method`]).
# ===========================================================================

"""
    hybridization_function(model::AndersonModel, ω) -> ComplexF64

Complex hybridization `Δ(ω) = Σ_k |V_k|²/(ω-ε_k+i0⁺)` of the flat band:
`Re Δ = (Γ/π)·ln|(ω+D)/(ω-D)|`, `Im Δ = -Γ` for `|ω|<D` (else 0). The
non-interacting Green's function is `G₀(ω) = 1/(ω - ε_d - Δ(ω))`.
"""
function hybridization_function(model::AndersonModel, ω::Real)
    Γ, D = model.Γ, model.D
    return complex((Γ / π) * log(abs((ω + D) / (ω - D))), abs(ω) < D ? -Γ : 0.0)
end

"""
    default_self_energy_method() -> AbstractSelfEnergyMethod

The robust default, [`SelfEnergyTrick`](@ref) (`Σ = U·F/G`; `Σ ∝ U`, errors cancel
in `F/G`). [`Dyson`](@ref) is offered for comparison.
"""
default_self_energy_method() = SelfEnergyTrick()

"""
    self_energy([method,] model, alg; via=default_self_energy_method(), b=0.6, window=0.7, ω=nothing) -> (; ω, Σ)

Impurity self-energy `Σ_σ(ω)`. `method` is the spectral method building `G` (default
`BHP`); `via` is how `Σ` is extracted: `SelfEnergyTrick()` (robust, `Σ=U·F/G`) or
`Dyson()` (`Σ=ω-ε_d-Δ-1/G`). At the symmetric point a Fermi liquid gives
`ReΣ(0)=U/2`, `ImΣ(0)=0`; `U=0 ⇒ Σ=0` (exact for the trick).
"""
function self_energy(
    method::AbstractSpectralMethod,
    model::AndersonModel,
    alg::NRGAlgorithm;
    via::AbstractSelfEnergyMethod=default_self_energy_method(),
    b::Real=0.6,
    window::Real=0.7,
    ω=nothing,
)
    method isa BHP || throw(
        EngineUnimplemented(
            "self_energy currently needs BHP for G (got $(typeof(method)))"
        ),
    )
    alg.symmetry isa U1U1 || throw(EngineUnimplemented("self_energy needs U1U1"))
    ωs = ω === nothing ? _default_omega(model, alg) : collect(float.(ω))
    return _self_energy(via, model, alg, ωs, b, window)
end
function self_energy(model::AbstractImpurityModel, alg::NRGAlgorithm; kw...)
    self_energy(default_spectral_method(), model, alg; kw...)
end

function _self_energy(::SelfEnergyTrick, model, alg, ωs, b, window)
    poles = _gf_poles(model, alg; window, with_F=true)
    G = _correlator(poles, ωs, b, 2)
    F = _correlator(poles, ωs, b, 3)
    return (; ω=ωs, Σ=model.U .* F ./ G)
end
function _self_energy(::Dyson, model, alg, ωs, b, window)
    poles = _gf_poles(model, alg; window, with_F=false)
    G = _correlator(poles, ωs, b, 2)
    Δ = hybridization_function.(Ref(model), ωs)
    return (; ω=ωs, Σ=[ωs[i] - model.εd - Δ[i] - 1 / G[i] for i in eachindex(ωs)])
end

"""
    compare_self_energy(model, alg; vias=(SelfEnergyTrick(), Dyson()), b=0.6, window=0.7, ω=nothing)
        -> (; ω, Σ, disagreement)

Run several self-energy formulations on a common grid and report `Σ` per method plus
the max pairwise `|ΔReΣ|` near `ω=0`. Cross-method agreement is the robustness signal
(at `U=0` the trick is exactly 0 while Dyson carries the broadening error — the gap is
the point).
"""
function compare_self_energy(
    model::AbstractImpurityModel,
    alg::NRGAlgorithm;
    vias=(SelfEnergyTrick(), Dyson()),
    b::Real=0.6,
    window::Real=0.7,
    ω=nothing,
)
    ωs = ω === nothing ? _default_omega(model, alg) : collect(float.(ω))
    Σ = Dict(
        nameof(typeof(v)) => self_energy(model, alg; via=v, b, window, ω=ωs).Σ for v in vias
    )
    near0 = findall(x -> abs(x) < 0.1, ωs)
    ks = collect(keys(Σ))
    dis = if isempty(near0) || length(ks) < 2
        0.0
    else
        maximum(maximum(abs, real.(Σ[a][near0] .- Σ[c][near0])) for a in ks, c in ks)
    end
    return (; ω=ωs, Σ, disagreement=dis)
end

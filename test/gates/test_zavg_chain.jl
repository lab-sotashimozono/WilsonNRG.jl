# Dispatch/faithfulness gate — the z-averaging discretizations (Campo–Oliveira PRB 72.104432,
# Žitko–Pruschke PRB 79.085106) build a genuine Wilson chain that plugs into the SAME iterative
# engine as WilsonLog. This proves the discretization axis is fully dispatched (not just via
# `band_dos`): `nrg_solve` runs end-to-end, and at U=0 reproduces the free-fermion subset-sums of
# the independently-diagonalized chain — the exact tier-1 check of the WilsonLog engine gate, now
# for the z-shifted chains. It also pins the structure (onsite=0, O(1) rescaled hopping), confirms
# the twist z is a real degree of freedom, and checks the honest fallback for an unwired scheme.

using WilsonNRG, Test
using LinearAlgebra: Symmetric, eigvals
using WilsonNRG: wilson_chain

# independent free-fermion reference — identical convention to test_nrg_engine_u1u1.jl:
# the single-particle Wilson-chain matrix under the same √Λ recursion the engine applies.
function _single_particle_zavg(model, chain, nsites)
    Λ = chain.disc.Λ
    m = reshape([model.εd], 1, 1)
    for n in 0:(nsites - 1)
        c = n == 0 ? bath_coupling(model) : chain.hopping[n]
        r = n == 0 ? 1.0 : sqrt(Λ)
        k = size(m, 1)
        mn = zeros(k + 1, k + 1)
        mn[1:k, 1:k] = r .* m
        mn[k + 1, k + 1] = chain.onsite[n + 1]
        mn[k, k + 1] = c
        mn[k + 1, k] = c
        m = mn
    end
    return eigvals(Symmetric(m))
end
function _subset_zavg(levels)
    s = [0.0]
    for λ in levels
        s = vcat(s, s .+ λ)
    end
    return sort(s)
end

# a discretization with no wilson_chain method (to exercise the clean fallback)
struct _UnwiredDisc <: WilsonNRG.AbstractDiscretization
    Λ::Float64
end

@testset "dispatch gate · z-averaging chains (Campo–Oliveira, Žitko–Pruschke) run in nrg_solve" begin
    D = 1.0

    # ---- (1) both schemes dispatch through nrg_solve AND reproduce free fermions at U=0 ----
    for disc in (CampoOliveira(2.5), ZitkoPruschke(2.5))
        for (U, εd) in ((0.0, 0.0), (0.0, 0.3))
            model = AndersonModel(; U, εd, Γ=0.05, D)
            alg = NRGAlgorithm(;
                discretization=disc, symmetry=U1U1(), truncation=KeepN(10^9), nsites=5
            )
            got = sort(nrg_solve(model, alg).energies[end])
            sp = _single_particle_zavg(model, wilson_chain(disc, model, 5), 5)
            ref = _subset_zavg(vcat(sp, sp))                        # ↑ and ↓ spin channels
            @test length(got) == length(ref) == 4096
            # ground-relative (the engine ground-subtracts each iteration)
            @test maximum(abs, (got .- got[1]) .- (ref .- ref[1])) < 1.0e-9
        end
    end

    # ---- (2) chain structure: onsite=0 (p–h symmetric), hopping positive, finite, O(1) ----
    model = AndersonModel(; U=0.0, εd=0.0, Γ=0.1, D)
    ch = wilson_chain(ZitkoPruschke(2.5), model, 20)
    @test all(==(0.0), ch.onsite)                              # flat band ⇒ no onsite energy
    @test all(isfinite, ch.hopping)
    @test all(>(0), ch.hopping[1:(end - 1)])                   # positive tridiagonal hoppings
    @test 0.1 < ch.hopping[10] < 2.0                           # O(1) ⇒ the Λ^{n/2} rescale is right

    # ---- (3) the twist z genuinely shifts the grid (z-averaging is a real DOF, not a no-op) ----
    c1 = wilson_chain(ZitkoPruschke(2.5; z=1.0), model, 10)
    cz = wilson_chain(ZitkoPruschke(2.5; z=0.5), model, 10)
    @test maximum(abs, c1.hopping .- cz.hopping) > 1.0e-3      # different z ⇒ different chain
    # Žitko's Eq. 36 correction acts on the *shifted* first interval: at z=1 it reduces exactly to
    # the Campo–Oliveira log-mean (the "+1−z" term vanishes, interval [Λ⁻¹,1] is identical), so the
    # chains coincide; for z≠1 the fix bites and the chains differ. (Faithful to PRB 79.085106.)
    @test wilson_chain(CampoOliveira(2.5; z=1.0), model, 10).hopping ==
        wilson_chain(ZitkoPruschke(2.5; z=1.0), model, 10).hopping
    @test maximum(abs, wilson_chain(CampoOliveira(2.5; z=0.5), model, 10).hopping .-
                       wilson_chain(ZitkoPruschke(2.5; z=0.5), model, 10).hopping) > 1.0e-4

    # ---- (4) an unwired discretization fails honestly (EngineUnimplemented, not MethodError) ----
    @test_throws EngineUnimplemented wilson_chain(_UnwiredDisc(2.5), model, 3)
    alg_bad = NRGAlgorithm(; discretization=_UnwiredDisc(2.5), symmetry=U1U1(), nsites=3)
    @test_throws EngineUnimplemented nrg_solve(model, alg_bad)
end

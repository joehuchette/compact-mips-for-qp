module MIPforQP

import JuMP, MathOptInterface
import Mosek, MosekTools
const MOI = MathOptInterface
const MOIU = MOI.Utilities
const VI = MOI.VariableIndex
const SV = MOI.SingleVariable

import LinearAlgebra, SparseArrays
import Gurobi

abstract type AbstractRelaxation end

struct HongboRelaxation <: AbstractRelaxation
    num_layers::Int
end
struct NeuralNetRelaxation <: AbstractRelaxation
    num_layers::Int
end
struct Reformulation{T <: AbstractRelaxation}
    relaxation::T
    soc_lower_bound::Bool
end
Reformulation(relaxation::AbstractRelaxation) = Reformulation(relaxation, true)

function formulate_quadratics!(model::JuMP.Model, reformulation::Reformulation)
    vq_ci, vq_f, vq_s = vectorize_quadratics!(model)
    # The following is a hacky way to determine if the objective is linear or quadratic.
    obj = try
        MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        nothing
    catch InexactError
        MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
    end
    if obj === nothing
        MOI.delete(JuMP.backend(model), vq_ci)
        bridge = _relax_quadratic!(JuMP.backend(model), reformulation, vq_f, vq_s)
    else
        bridge = _relax_quadratic!(JuMP.backend(model), reformulation, vq_f, vq_s, obj)
    end
    MOIU.attach_optimizer(model)
    return bridge
end

function vectorize_quadratics!(model::JuMP.Model)
    moi_backend = JuMP.backend(model)
    @assert typeof(moi_backend) <: MOIU.CachingOptimizer
    @assert moi_backend.state in (MOIU.NO_OPTIMIZER, MOIU.EMPTY_OPTIMIZER)
    return vectorize_quadratics!(moi_backend.model_cache.model)
end

function vectorize_quadratics!(cached_model::MOIU.Model{T}) where {T}
    scalar_quad_cons = cached_model.moi_scalarquadraticfunction
    @assert isempty(scalar_quad_cons.moi_equalto)
    @assert isempty(scalar_quad_cons.moi_interval)
    @assert isempty(scalar_quad_cons.moi_semicontinuous)
    @assert isempty(scalar_quad_cons.moi_semiinteger)
    @assert isempty(scalar_quad_cons.moi_zeroone)

    num_gt = length(scalar_quad_cons.moi_greaterthan)
    num_lt = length(scalar_quad_cons.moi_lessthan)

    constants = Array{T}(undef, 0)
    affine_terms = Array{MOI.VectorAffineTerm{T}}(undef, 0)
    quad_terms = Array{MOI.VectorQuadraticTerm{T}}(undef, 0)

    func_count = 0
    for (q_ci, q_f, q_s) in scalar_quad_cons.moi_greaterthan
        func_count += 1
        push!(constants, -(q_f.constant - q_s.lower))
        for aff in q_f.affine_terms
            push!(affine_terms, MOI.VectorAffineTerm{T}(func_count, MOI.ScalarAffineTerm{T}(-aff.coefficient, aff.variable_index)))
        end
        for quad in q_f.quadratic_terms
            push!(quad_terms, MOI.VectorQuadraticTerm{T}(func_count, MOI.ScalarQuadraticTerm{T}(-quad.coefficient, quad.variable_index_1, quad.variable_index_2)))
        end
    end

    @assert func_count == num_gt

    for (q_ci, q_f, q_s) in scalar_quad_cons.moi_lessthan
        func_count += 1
        push!(constants, q_f.constant - q_s.upper)
        for aff in q_f.affine_terms
            push!(affine_terms, MOI.VectorAffineTerm{T}(func_count, MOI.ScalarAffineTerm{T}(aff.coefficient, aff.variable_index)))
        end
        for quad in q_f.quadratic_terms
            push!(quad_terms, MOI.VectorQuadraticTerm{T}(func_count, MOI.ScalarQuadraticTerm{T}(quad.coefficient, quad.variable_index_1, quad.variable_index_2)))
        end
    end

    @assert func_count == num_gt + num_lt

    vector_quad_cons = cached_model.moi_vectorquadraticfunction.moi_nonpositives
    @assert isempty(vector_quad_cons)
    constraint_index = MOI.ConstraintIndex{MOI.VectorQuadraticFunction{T},MOI.Nonpositives}(1)
    vector_func = MOI.VectorQuadraticFunction{T}(affine_terms, quad_terms, constants)
    vector_set = MOI.Nonpositives(num_gt + num_lt)
    retval = (constraint_index, vector_func, vector_set)
    push!(vector_quad_cons, retval)
    empty!(scalar_quad_cons.moi_greaterthan)
    empty!(scalar_quad_cons.moi_lessthan)
    return retval
end

struct MIPforQPBridge
    reformulation::Reformulation
    squared_vars::Dict{VI,VI}
    binaries::Dict{VI,Union{Nothing,Vector{VI}}}
end
MIPforQPBridge(reformulation::Reformulation) = MIPforQPBridge(reformulation, Dict{VI,VI}(), Dict{VI,Vector{VI}}())

function _indices_and_coefficients(
    I::AbstractVector{Int}, J::AbstractVector{Int}, V::AbstractVector{Float64},
    indices::AbstractVector{Int}, coefficients::AbstractVector{Float64},
    f::MOI.ScalarQuadraticFunction, canonical_index::Dict{VI,Int}
)
    variables_seen = 0
    for (i, term) in enumerate(f.quadratic_terms)
        vi_1 = term.variable_index_1
        vi_2 = term.variable_index_2
        if !haskey(canonical_index, vi_1)
            variables_seen += 1
            canonical_index[vi_1] = variables_seen
        end
        if !haskey(canonical_index, vi_2)
            variables_seen += 1
            canonical_index[vi_2] = variables_seen
        end

        I[i] = canonical_index[term.variable_index_1]
        J[i] = canonical_index[term.variable_index_2]
        V[i] =  term.coefficient
        # Gurobi returns a list of terms. MOI requires 0.5 x' Q x. So, to get
        # from
        #   Gurobi -> MOI => multiply diagonals by 2.0
        #   MOI -> Gurobi => multiply diagonals by 0.5
        # Example: 2x^2 + x*y + y^2
        #   |x y| * |a b| * |x| = |ax+by bx+cy| * |x| = 0.5ax^2 + bxy + 0.5cy^2
        #           |b c|   |y|                   |y|
        #   Gurobi needs: (I, J, V) = ([0, 0, 1], [0, 1, 1], [2, 1, 1])
        #   MOI needs:
        #     [SQT(4.0, x, x), SQT(1.0, x, y), SQT(2.0, y, y)]
        if I[i] == J[i]
            V[i] *= 0.5
        end
    end
    for (i, term) in enumerate(f.affine_terms)
        indices[i] = term.variable_index.value
        coefficients[i] = term.coefficient
    end
    return
end

function _indices_and_coefficients(f::MOI.ScalarQuadraticFunction, canonical_index::Dict{VI,Int})
    f_canon = MOI.Utilities.canonical(f)
    nnz_quadratic = length(f_canon.quadratic_terms)
    nnz_affine = length(f_canon.affine_terms)
    I = Vector{Int}(undef, nnz_quadratic)
    J = Vector{Int}(undef, nnz_quadratic)
    V = Vector{Float64}(undef, nnz_quadratic)
    indices = Vector{Int}(undef, nnz_affine)
    coefficients = Vector{Float64}(undef, nnz_affine)
    _indices_and_coefficients(I, J, V, indices, coefficients, f_canon, canonical_index)
    return indices, coefficients, I, J, V
end

function _get_bounds(model, x::VI)
    c_lt = MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}}(x.value)
    c_gt = MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}}(x.value)
    c_int = MOI.ConstraintIndex{MOI.SingleVariable,MOI.Interval{Float64}}(x.value)
    if MOI.is_valid(model, c_int)
        @assert !MOI.is_valid(c_lt) && !MOI.is_valid(c_gt)
        int::MOI.Interval{Float64} = MOI.get(model, MOI.ConstraintSet(), c_int)
        return int.lower, int.upper
    elseif MOI.is_valid(model, c_lt)
        lt::MOI.LessThan{Float64} = MOI.get(model, MOI.ConstraintSet(), c_lt)
        if MOI.is_valid(model, c_gt)
            gt_1::MOI.GreaterThan{Float64} = MOI.get(model, MOI.ConstraintSet(), c_gt)
            return gt_1.lower, lt.upper
        else
            return -Inf, lt.upper
        end
    elseif MOI.is_valid(model, c_gt)
        gt_2::MOI.GreaterThan{Float64} = MOI.get(model, MOI.ConstraintSet(), c_gt)
        return gt_2.lower, Inf
    else
        return -Inf, Inf
    end
end

function _map_to_interval_if_needed(model, x::VI, y::VI, l::Float64, u::Float64)
    if l == 0 && u == 1
        return x, y
    else
        Δ = u - l
        @assert Δ > 0
        x̃_v, x̃_c = MOI.add_constrained_variable(model, MOI.Interval(0.0, 1.0))
        ỹ_v, ỹ_c = MOI.add_constrained_variable(model, MOI.Interval(0.0, 1.0))
        MOIU.normalize_and_add_constraint(model, SV(x) - (l + Δ * SV(x̃_v)), MOI.EqualTo(0.0))
        MOIU.normalize_and_add_constraint(model, SV(y) - (l^2 + 2l * Δ * SV(x̃_v) + Δ^2 * SV(ỹ_v)), MOI.EqualTo(0.0))
        return x̃_v, ỹ_v
    end
end

function reformulate_unary_quadratic_term!(model::MOI.ModelLike, reformulation::Reformulation{HongboRelaxation}, _x::VI, _y::VI, l::Float64, u::Float64)
    num_layers = reformulation.relaxation.num_layers
    ξ_v = MOI.add_variables(model, num_layers)
    η_v = MOI.add_variables(model, num_layers)
    λ_1_v, λ_1_c = MOI.add_constrained_variables(model, [MOI.GreaterThan(0.0) for i in 1:num_layers])
    λ_2_v, λ_2_c = MOI.add_constrained_variables(model, [MOI.GreaterThan(0.0) for i in 1:num_layers])
    z_v, z_c = MOI.add_constrained_variables(model, [MOI.ZeroOne() for i in 1:num_layers])
    x = SV(_x)
    y = SV(_y)
    ξ = [SV(ξ_v[i]) for i in 1:num_layers]
    η = [SV(η_v[i]) for i in 1:num_layers]
    λ_1 = [SV(λ_1_v[i]) for i in 1:num_layers]
    λ_2 = [SV(λ_2_v[i]) for i in 1:num_layers]
    z = [SV(z_v[i]) for i in 1:num_layers]

    θ_min = (if l > 0
        atan(l^2 - 1) / (2l)
    elseif l == 0
        -π / 2
    else
        atan(l^2 - 1) / (2l) - π
    end)
    θ_max = (if u > 0
        atan(u^2 - 1) / (2u)
    elseif u == 0
        -π / 2
    else
        atan(u^2 - 1) / (2u) - π
    end)
    @assert -3π / 2 < θ_min < π / 2
    @assert -3π / 2 < θ_max < π / 2
    θ_d = θ_max - θ_min
    θ_mid = (θ_max + θ_min) / 2

    ν = num_layers
    # RLP inequality (1)
    MOIU.normalize_and_add_constraint(model, y - (l + u) * x + l * u, MOI.LessThan(0.0))
    # Equation (11)
    MOIU.normalize_and_add_constraint(model, ξ[ν] * cos(θ_d / 2^(ν + 1)) + η[ν] * sin(θ_d / 2^(ν + 1)) - (y + 1.0) / 2.0 * cos(θ_d / 2^(ν + 1)), MOI.GreaterThan(0.0))
    if !reformulation.soc_lower_bound
        # Equation 12
        MOIU.normalize_and_add_constraint(model, ξ[ν] * cos(θ_d / 2^ν) + η[ν] * sin(θ_d / 2^ν) - (y + 1.0) / 2.0, MOI.LessThan(0.0))
        # Equation 13
        MOIU.normalize_and_add_constraint(model, ξ[ν] - (y + 1.0) / 2.0, MOI.LessThan(0.0))
    end
    # Equation block (20)
    MOIU.normalize_and_add_constraint(model, ξ[1] - x * cos(θ_mid) - (y / 2.0 - 0.5) * sin(θ_mid), MOI.EqualTo(0.0))
    C = max(l^2, u^2) / 2 + 0.5
    MOIU.normalize_and_add_constraint(model, (1.0λ_2[1] - 1.0λ_1[1]) * C + x * sin(θ_mid) - (y / 2.0 - 0.5) * cos(θ_mid), MOI.EqualTo(0.0))
    MOI.add_constraint(model, η[1] - (1.0λ_1[1] + 1.0λ_2[1]) * C, MOI.EqualTo(0.0))
    MOIU.normalize_and_add_constraint(model, 1.0λ_1[1] - (1.0 - z[1]), MOI.LessThan(0.0))
    MOI.add_constraint(model, 1.0λ_2[1] - 1.0z[1], MOI.LessThan(0.0))
    for j in 1:(ν - 1)
        # Equation block 21
        MOI.add_constraint(model, ξ[j + 1] - ξ[j] * cos(θ_d / 2^(j + 1)) - η[j] * sin(θ_d / 2^(j + 1)), MOI.EqualTo(0.0))
        C_j = C * sin(θ_d / 2^(j + 1))
        MOI.add_constraint(model, (1.0λ_2[j + 1] - 1.0λ_1[j + 1]) * C_j + ξ[j] * sin(θ_d / 2^(j + 1)) - η[j] * cos(θ_d / 2^(j + 1)), MOI.EqualTo(0.0))
        MOI.add_constraint(model, η[j + 1] - (1.0λ_1[j + 1] + 1.0λ_2[j + 1]) * C_j, MOI.EqualTo(0.0))
        MOIU.normalize_and_add_constraint(model, 1.0λ_1[j] - (1.0 - z[j + 1]), MOI.LessThan(0.0))
        MOI.add_constraint(model, 1.0λ_2[j] - 1.0z[j + 1], MOI.LessThan(0.0))
    end
end

function reformulate_unary_quadratic_term!(model::MOI.ModelLike, reformulation::Reformulation{NeuralNetRelaxation}, _x::VI, _y::VI, l::Float64, u::Float64)
    x, y = _map_to_interval_if_needed(model, _x, _y, l, u)
    MOI.add_constraint(model, 1.0SV(x) - SV(y), MOI.GreaterThan(0.0))
    return _formulate_zero_one_quadratic!(model, x, y, reformulation.relaxation.num_layers, !reformulation.soc_lower_bound)
end

function _formulate_zero_one_quadratic!(model::MOI.ModelLike, x::VI, y::VI, num_layers::Int, impose_lower_bound::Bool)
    z_v, z_c = MOI.add_constrained_variables(model, [MOI.ZeroOne() for i in 1:num_layers])
    g_v, g_c = MOI.add_constrained_variables(model, [MOI.Interval(0.0, 1.0) for i in 1:num_layers])

    gv = Dict(i => SV(g_v[i]) for i in 1:num_layers)
    gv[0] = SV(x)

    agg_aff = 1.0 * SV(y) - SV(x) + sum(gv[s] / 2.0^(2s) for s in 1:num_layers)

    if impose_lower_bound
        agg_c = MOI.add_constraint(model, agg_aff, MOI.EqualTo(0.0))
    else
        agg_c = MOI.add_constraint(model, agg_aff, MOI.LessThan(0.0))
    end

    for i in 1:num_layers
        α_i = SV(z_v[i])
        MOI.add_constraint(model, gv[i] - 2.0gv[i - 1], MOI.LessThan(0.0))
        MOIU.normalize_and_add_constraint(model, gv[i] + 2.0gv[i - 1] - 2.0, MOI.LessThan(0.0))
        MOI.add_constraint(model, gv[i] + 2.0gv[i - 1] - 2.0 * α_i, MOI.GreaterThan(0.0))
        MOIU.normalize_and_add_constraint(model, gv[i] - 2.0gv[i - 1] + 2.0 * α_i, MOI.GreaterThan(0.0))
    end
    return z_v
end

function _get_Q_matrix(q_func::MOI.ScalarQuadraticFunction, canonical_index::Dict{VI,Int}, n::Int)
    indices, coefficients, I, J, V = _indices_and_coefficients(q_func, canonical_index)
    Q = Matrix{Float64}(SparseArrays.sparse(I, J, V, n, n))
    Q = 1 / 2 * (Q + Q')

    @assert n == size(Q, 1) == size(Q, 2)
    @assert LinearAlgebra.issymmetric(Q)
    return Q
end

function _compute_diagonal_shift(Q::Matrix{Float64})
    n = size(Q, 1)
    @assert n == size(Q, 2)
    sdp_model = JuMP.Model(JuMP.with_optimizer(Mosek.Optimizer))
    δ = JuMP.@variable(sdp_model, [i = 1:n])
    JuMP.@objective(sdp_model, Min, sum(δ))
    JuMP.@SDconstraint(sdp_model, Q + LinearAlgebra.diagm(δ .- 1e-4) >= 0)
    JuMP.optimize!(sdp_model)
    @assert JuMP.primal_status(sdp_model) == MOI.FEASIBLE_POINT
    return JuMP.value.(δ)
end

# Assumption: Constraints we wish to reformulate have been bridged to the form
# `VectorQuadraticFunction`-in-`Nonpositives`.
function _relax_quadratic!(
    model,
    reformulation::Reformulation,
    f::MOI.VectorQuadraticFunction{Float64},
    s::MOI.Nonpositives,
    obj::Union{Nothing,MOI.ScalarQuadraticFunction{Float64}}=nothing
)
    squared_vars = Dict{VI,VI}()
    binaries = Dict{VI,Union{Nothing,Vector{VI}}}()
    # Walk all constraints in vector and populate squared_vars dict
    sqt_terms = vcat([t -> t.scalar_term for t in f.quadratic_terms], obj.quadratic_terms)
    for st in sqt_terms
        for x in (st.variable_index_1, st.variable_index_2)
            if !haskey(squared_vars, x)
                l, u = _get_bounds(model, x)
                y_lb = l < 0 < u ? 0.0 : min(l^2, u^2)
                y_ub = max(l^2, u^2)
                y_v, y_c = MOI.add_constrained_variable(model, MOI.Interval{Float64}(y_lb, y_ub))
                squared_vars[x] = y_v
                if reformulation.soc_lower_bound
                    # Alternatively, could pass SOC constraint with
                    # MOI.add_constraint(model, MOI.VectorOfVariables([y_v, x]), MOI.SecondOrderCone(2))
                    MOI.add_constraint(model, 1.0SV(y_v) - 1.0SV(x) * SV(x), MOI.GreaterThan(0.0))
                end
                z_v = reformulate_unary_quadratic_term!(model, reformulation, x, y_v, l, u)
                binaries[x] = z_v
            end
        end
    end

    n = length(keys(squared_vars))
    q_funcs = MOIU.scalarize(f)
    canonical_index = Dict{VI,Int}()
    for q_func in q_funcs
        Q = _get_Q_matrix(q_func, canonical_index, n)
        δ_value = _compute_diagonal_shift(Q)
        quad_shift = MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm{Float64}[], [MOI.ScalarQuadraticTerm(2δ_value[canonical_index[xi]], xi, xi) for xi in keys(canonical_index)], 0.0)
        aff_shift = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(-δ_value[canonical_index[xi]], squared_vars[xi]) for xi in keys(canonical_index)], 0.0)
        shifted_q_func = MOIU.canonical(q_func + quad_shift + aff_shift)
        MOIU.normalize_and_add_constraint(model, shifted_q_func, MOI.LessThan(0.0))
    end
    # Now handle objective
    if obj !== nothing
        Q = _get_Q_matrix(obj, canonical_index, n)
        δ_value = _compute_diagonal_shift(Q)
        quad_shift = MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm{Float64}[], [MOI.ScalarQuadraticTerm(2δ_value[canonical_index[xi]], xi, xi) for xi in keys(canonical_index)], 0.0)
        aff_shift = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(-δ_value[canonical_index[xi]], squared_vars[xi]) for xi in keys(canonical_index)], 0.0)
        shifted_q_func = MOIU.canonical(obj + quad_shift + aff_shift)
        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), shifted_q_func)
    end

    return MIPforQPBridge(reformulation, squared_vars, binaries)
end

include("box_qp_util.jl")

end # module

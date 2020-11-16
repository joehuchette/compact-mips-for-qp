module BoxQP

import JuMP

using LinearAlgebra

using MIPforQP

import MathOptInterface
const MOI = MathOptInterface

function parse_box_qp(filename::String)
    open(filename) do fp
        n = parse(Int, readline(fp))
        c = parse.(Float64, split(readline(fp)))
        @assert n == length(c)
        Q = zeros(Float64, n, n)
        for i in 1:n
            Q[:,i] = parse.(Float64, split(readline(fp)))
        end
        return Q, c
    end
end

function formulate_box_qp(Q::Matrix{T}, c::Vector{T}, optimizer_factory; epigraph::Bool=false) where {T}
    n = length(c)
    @assert n == size(Q, 1) == size(Q, 2)
    model = JuMP.Model(JuMP.with_optimizer(optimizer_factory))
    JuMP.@variable(model, 0 ≤ x[1:n] ≤ 1)
    if epigraph
        JuMP.@variable(model, y)
        JuMP.@constraint(model, y ≥ -0.5dot(x, Q * x) - dot(c, x))
        JuMP.@objective(model, Min, y)
    else
        JuMP.@objective(model, Min, -0.5dot(x, Q * x) - dot(c, x))
    end
    return model
end

end  # module BoxQP

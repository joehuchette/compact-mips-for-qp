using MIPforQP

using MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

using JuMP, BARON, Gurobi, LinearAlgebra

const TIME_LIMIT = 5 * 60.0

const NUM_LAYERS = 3
const SOC_LOWER_BOUND = false
const EPIGRAPH = false
const FAMILIES = ("small", "basic", "extended", "extended2")

const RESULT_PATH = joinpath(@__DIR__, "result.csv")
const ROOT_DIR = joinpath(@__DIR__, "box_qp", "data")

function gurobi_direct_factory(cutoff)
    return JuMP.optimizer_with_attributes(
        Gurobi.Optimizer,
        "TimeLimit" => TIME_LIMIT,
        "NonConvex" => 2,
        "Cutoff" => cutoff
    )
end
function gurobi_direct_mipfocus_factory(cutoff, mipfocus::Int)
    return JuMP.optimizer_with_attributes(
        Gurobi.Optimizer,
        "TimeLimit" => TIME_LIMIT,
        "NonConvex" => 2,
        "MIPFocus" => mipfocus,
        "Cutoff" => cutoff
    )
end
function gurobi_mip_factory(cutoff)
    return JuMP.optimizer_with_attributes(
        Gurobi.Optimizer,
        "TimeLimit" => TIME_LIMIT,
        "MIPFocus" => 3,
        "Cutoff" => cutoff
    )
end
function baron_direct_factory(cutoff)
    return JuMP.optimizer_with_attributes(
        BARON.Optimizer,
        # NOTE: To plug in CPLEX as LP/MIP solver, uncomment the following line and fill in the correct path.
        # "CplexLibName" => "/path/to/cplex/library",
        "MaxTime" => TIME_LIMIT,
        "CutOff" => cutoff
    )
end
const primal_factory = JuMP.optimizer_with_attributes(
    Gurobi.Optimizer,
    "TimeLimit" => TIME_LIMIT,
    "NonConvex" => 2,
    "MIPFocus" => 1
)

function nn_bridge(model)
    MIPforQP.formulate_quadratics!(
        model,
        MIPforQP.Reformulation(MIPforQP.NeuralNetRelaxation(NUM_LAYERS), SOC_LOWER_BOUND)
    )
    return
end
function hongbo_bridge(model)
    MIPforQP.formulate_quadratics!(
        model,
        MIPforQP.Reformulation(MIPforQP.HongboRelaxation(NUM_LAYERS), SOC_LOWER_BOUND)
    )
    return
end

const FACTORIES_AND_BRIDGES = Dict(
    ("gurobi", "direct") => (gurobi_direct_factory, identity),
    ("gurobi_mf_3", "direct") => (co -> gurobi_direct_mipfocus_factory(co, 3), identity),
    ("gurobi", "nn")     => (gurobi_mip_factory, nn_bridge),
    ("gurobi", "hongbo") => (gurobi_mip_factory, hongbo_bridge),
    ("baron", "direct")  => (baron_direct_factory, identity),
)

const CONFIGURATIONS = collect(keys(FACTORIES_AND_BRIDGES))

const CONFIGS_TO_ANALYZE = [
    ("gurobi", "direct"),
    ("gurobi_mf_3", "direct"),
    ("gurobi", "nn"),
    ("gurobi", "hongbo"),
    ("baron", "direct"),
]

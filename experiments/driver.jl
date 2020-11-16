include("common.jl")

fp = open(RESULT_PATH, "w")
println(fp, "family,instance,solver,formulation,optimal,solve_time,true_primal,model_primal,dual")

for family in FAMILIES, instance in readdir(joinpath(ROOT_DIR, family))
    Q, c = MIPforQP.BoxQP.parse_box_qp(joinpath(ROOT_DIR, family, instance))
    primal_model = MIPforQP.BoxQP.formulate_box_qp(Q, c, primal_factory, epigraph=EPIGRAPH)
    optimize!(primal_model)
    @assert primal_status(primal_model) == MOI.FEASIBLE_POINT
    cutoff = JuMP.objective_value(primal_model)
    for (solver, method) in CONFIGURATIONS
        factory, bridge = FACTORIES_AND_BRIDGES[(solver, method)]
        model = MIPforQP.BoxQP.formulate_box_qp(Q, c, factory(cutoff), epigraph=EPIGRAPH)
        bridge(model)
        JuMP.optimize!(model)
        true_pb = Inf
        model_pb = Inf
        if primal_status(model) == MOI.FEASIBLE_POINT
            model_pb = JuMP.objective_value(model)
            x = [JuMP.variable_by_name(model, "x[$i]") for i in 1:length(c)]
            @assert isa(x, Vector{JuMP.VariableRef})
            x_val = JuMP.value.(x)
            true_pb = -0.5dot(x_val, Q * x_val) - dot(c, x_val)
        end
        db = JuMP.objective_bound(model)
        st = JuMP.solve_time(model)
        opt = JuMP.termination_status(model) == MOI.OPTIMAL
        println(fp, join([family, instance, solver, method, opt, st, true_pb, model_pb, db], ","))
        flush(fp)
    end
end

close(fp)

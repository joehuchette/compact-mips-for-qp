using CSV
using StatsBase

include("common.jl")

_shifted_geomean(vals, shift) = StatsBase.geomean(vals .+ shift) - shift

df = CSV.read(RESULT_PATH)

# Remove configurations that might be in data set, but that we
# do not wish to include in our analysis.
df = filter(df) do row
    for (solver, method) in CONFIGS_TO_ANALYZE
        if row.solver == solver && row.formulation == method
            return true
        end
    end
    return false
end

df.fam_instance = df.family .* "-" .* df.instance

# Figure out the best primal objective value available in dataset
# for each instance.
df[!, :best_primal] .= 0.0
for i in 1:size(df, 1)
    row = df[i, :]
    best_primal_value = minimum(df[(df.family .== row.family) .& (df.instance .== row.instance), :true_primal])
    df.best_primal[i] = best_primal_value
end

# Add a column to record MIP gap, with respect to best primal solution in data set.
function _mip_gap(row)
    return 100 * abs(row.dual - row.best_primal) / abs(row.best_primal)
end
df[!, :gap] .= 0.0
for i in 1:size(df, 1)
    df.gap[i] = _mip_gap(df[i,:])
end

# Split off problem classes based on how easy they were. We will do separate analysis
# on each problem class.
PROBLEM_CLASS_DFS = Dict()
PROBLEM_CLASS_DFS["solved"] = filter(df) do row
    family, instance = row.family, row.instance
    slice = df[(df.family .== row.family) .& (df.instance .== row.instance), :]
    return all(slice.optimal .== "true")
end

PROBLEM_CLASS_DFS["contested"] = filter(df) do row
    family, instance = row.family, row.instance
    slice = df[(df.family .== row.family) .& (df.instance .== row.instance), :]
    return any(slice.optimal .== "true") && !all(slice.optimal .== "true")
end

PROBLEM_CLASS_DFS["unsolved"] = filter(df) do row
    family, instance = row.family, row.instance
    slice = df[(df.family .== row.family) .& (df.instance .== row.instance), :]
    return !any(slice.optimal .== "true")
end

for (class, subset_df) in PROBLEM_CLASS_DFS
    total_instance_count = length(unique(subset_df.fam_instance))
    total_instance_count == 0 && continue
    println("Class: $class ($total_instance_count instances)")
    println("="^(30))

    fails = Dict(pair => 0 for pair in CONFIGS_TO_ANALYZE)
    best_bound = Dict(pair => 0 for pair in CONFIGS_TO_ANALYZE)
    for fam_instance in unique(subset_df.fam_instance)
        slice = subset_df[subset_df.fam_instance .== fam_instance, :]
        if size(slice, 1) == 0 error("Unexpected case, aborting.") end
        best_solve_time = minimum(slice.solve_time)
        best_gap = minimum(slice.gap)
        for (solver, formulation) in CONFIGS_TO_ANALYZE
            sub_slice = slice[(slice.solver .== solver) .& (slice.formulation .== formulation), :]
            solve_times = sub_slice.solve_time
            @assert size(solve_times, 1) == 1
            gaps = sub_slice.gap
            solve_time = minimum(solve_times)
            gap = minimum(gaps)
            best_primal = minimum(sub_slice.best_primal)
            if solve_time >= 0.999 * TIME_LIMIT
                fails[(solver, formulation)] += 1
            end
            # Either attain best bound, or reach Gurobi's optimality cutoff,
            # which is 1e-4 * |primal_bound|
            if gap == best_gap || gap < 1e-4 * abs(best_primal)
                best_bound[(solver, formulation)] += 1
            end
        end
    end

    time_shift = minimum(subset_df.solve_time)
    gap_shift = max(1e-4, minimum(subset_df.gap))

    for (solver, formulation) in CONFIGS_TO_ANALYZE
        slice = subset_df[(subset_df.solver .== solver) .& (subset_df.formulation .== formulation), :]
        println("$solver -- $formulation")
        println("-"^(length(solver) + length(formulation) + 4))
        println("  * solve time: ", _shifted_geomean(slice.solve_time, time_shift), " sec")
        println("  * MIP gap:    ", _shifted_geomean(slice.gap, gap_shift))
        num_fails = fails[(solver, formulation)]
        num_best_bound = best_bound[(solver, formulation)]
        println("  * Fails:      ", num_fails, " / ", total_instance_count)
        println("  * Best bound: ", num_best_bound, " / ", total_instance_count)
    end
end

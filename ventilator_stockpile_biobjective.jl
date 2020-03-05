if length(ARGS) == 1
    const FILENAME = ARGS[1]
else
    print("""
    > julia ventilator_stockpiling.jl [filename]

    Run the ventilator stockpiling model using the input JSON file
    at [filename], and save the output to [filename]_solution.json.
    """)
    exit(0)
end

import Distributions
import GLPK
import JSON
import LinearAlgebra
import Statistics
using JuMP

struct ParetoSolution
    y::Vector{Float64}
end

function Base.isapprox(a::ParetoSolution, b::ParetoSolution; kwargs...)
    return isapprox(a.y, b.y; kwargs...)
end

function build_model(filename)
    data = JSON.parsefile(filename)
    regions = collect(keys(data["regions"]))
    μ = [data["regions"][r]["mean"] for r in regions]
    σ = [data["regions"][r]["std"] for r in regions]
    Σ = LinearAlgebra.Symmetric([
        (i == j ? 1.0 : data["correlation"]) * σ[i] * σ[j]
        for i = 1:length(regions), j = 1:length(regions)
    ])
    demand = Distributions.MvNormal(μ, Σ)
    Ω = [
        Dict(regions[i] => max(0, d) for (i, d) in enumerate(rand(demand)))
        for _ = 1:data["number_scenarios"]
    ]
    model = direct_model(GLPK.Optimizer())
    set_silent(model)
    @variable(model, central >= 0)
    @variable(model, forward_deploy[regions] >= 0)
    @variable(model, recourse[regions, 1:length(Ω)] >= 0)
    @variable(model, unmet_demand[regions, 1:length(Ω)] >= 0)
    @constraint(model, [i=1:length(Ω)], sum(recourse[:,  i]) <= central)
    @constraint(
        model,
        [r=regions, i=1:length(Ω)],
        forward_deploy[r] +
            (1 - data["regions"][r]["wastage"]) * recourse[r, i] +
            unmet_demand[r, i] == Ω[i][r]
    )
    @variable(model, stockpile)
    @constraint(model, stockpile == central + sum(forward_deploy))
    @variable(model, EUD)
    @constraint(model, EUD == sum(unmet_demand) / length(Ω))
    @objective(model, Min, stockpile)
    if !data["solve_for_central_stockpile"]
        fix(central, data["central_stockpile"])
    end
    if !data["solve_for_regional_stockpile"]
        for r in regions
            fix(forward_deploy[r], data["regions"][r]["current_stockpile"])
        end
    end
    optimize!(model)
    return model
end

function solve_nise_weight(model, λ)
    @info "Solving: $λ"
    @objective(model, Min, (1 - λ) * model[:stockpile] + λ * model[:EUD])
    optimize!(model)
    if λ ≈ 0.0
        fix(model[:EUD], value(model[:EUD]))
        @objective(model, Min, model[:stockpile])
        optimize!(model)
    elseif λ ≈ 1.0
        fix(model[:stockpile], value(model[:stockpile]))
        @objective(model, Min, model[:EUD])
        optimize!(model)
    end
    unmet_demand = value.(model[:unmet_demand])
    probability_of_unmet_demand = 100 * sum(
        sum(unmet_demand[:, i]) > 0
        for i = 1:size(unmet_demand, 2)
    ) / size(unmet_demand, 2)
    solution = Dict(
        "objective_value" => objective_value(model),
        "stockpile" => round(value(model[:stockpile]), digits = 1),
        "regions" => value.(model[:forward_deploy]),
        "EUD" => round(value(model[:EUD]), digits = 1),
        "PUD" => round(probability_of_unmet_demand; digits = 1),
        "pareto" => ParetoSolution(
            [value(model[:EUD]), value(model[:stockpile])]
        )
    )
    if λ ≈ 0.0
        unfix(model[:EUD])
    elseif λ ≈ 1.0
        unfix(model[:stockpile])
    end
    return solution
end

function solve_nise(model; solution_limit = 20)
    solutions = Dict{Float64, Any}()
    for w in (0.0, 1.0)
        solutions[w] = solve_nise_weight(model, w)
    end
    queue = Tuple{Float64, Float64}[]
    if !(solutions[0.0]["pareto"] ≈ solutions[1.0]["pareto"])
        push!(queue, (0.0, 1.0))
    end
    while length(queue) > 0 && length(solutions) < solution_limit
        (a, b) = popfirst!(queue)
        y_d = solutions[a]["pareto"].y .- solutions[b]["pareto"].y
        w = y_d[2] / (y_d[2] - y_d[1])
        solution = solve_nise_weight(model, w)
        if solution["pareto"] ≈ solutions[a]["pareto"] || solution["pareto"] ≈ solutions[b]["pareto"]
            # We have found an existing solution. We're free to prune (a, b)
            # from the search space.
        else
            # Solution is identical to a and b, so search the domain (a, w) and
            # (w, b), and add solution as a new Pareto-optimal solution!
            push!(queue, (a, w))
            push!(queue, (w, b))
            solutions[w] = solution
        end
    end
    return [solutions[w] for w in sort(collect(keys(solutions)), rev=true)]
end

function run_stockpiling_model(filename::String)
    @info "Running: $(filename)"
    model = build_model(filename)
    solutions = solve_nise(model)
    open(replace(filename, ".json" => "_solution.json"), "w") do io
        write(io, JSON.json(solutions))
    end
    return solutions
end

run_stockpiling_model(FILENAME)

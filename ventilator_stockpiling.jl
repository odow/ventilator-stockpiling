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
using JuMP

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
    @constraint(
        model,
        con_unmet_demand,
        sum(unmet_demand) / length(Ω) <= 0.0
    )
    @objective(model, Min, central + sum(forward_deploy))
    if !data["solve_for_central_stockpile"]
        fix(central, data["central_stockpile"])
    end
    if !data["solve_for_regional_stockpile"]
        for r in regions
            fix(forward_deploy[r], data["regions"][r]["current_stockpile"])
        end
    end
    optimize!(model)
    return model, regions
end

function solve_eud_level(model, regions, L)
    if mod(L, 100) == 0
        @info "Running L = $L"
    end
    set_normalized_rhs(model[:con_unmet_demand], L)
    optimize!(model)
    stockpile = Dict(
        "Central" => round(value(model[:central]); digits = 1)
    )
    for r in regions
        stockpile[r] = round(value(model[:forward_deploy][r]); digits = 1)
    end
    unmet_demand = value.(model[:unmet_demand])
    probability_of_unmet_demand = 100 * sum(
        sum(unmet_demand[:, i]) > 0
        for i = 1:size(unmet_demand, 2)
    ) / size(unmet_demand, 2)
    return Dict(
        "stockpile" => stockpile,
        "EUD" => L,
        "PUD" => round(probability_of_unmet_demand; digits = 1)
    )
end

function run_stockpiling_model(filename::String)
    @info "Running: $(filename)"
    model, regions = build_model(filename)
    solutions = Any[]
    L = 0
    while true
        solution = solve_eud_level(model, regions, L)
        push!(solutions, solution)
        if sum(values(solution["stockpile"])) == 0
            break
        end
        L += 1
    end
    open(replace(filename, ".json" => "_solution.json"), "w") do io
        write(io, JSON.json(solutions))
    end
    return
end

run_stockpiling_model(FILENAME)

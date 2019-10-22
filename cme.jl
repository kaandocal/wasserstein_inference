"""

This file is a Julia version of the file cme.py; most of it
is a straightforward translation of the original Python code,
optimized for speed using static typing. See cme.py for more
details on the implementation and cme_julia.py for the Python
bindings to this file.

"""
using Distributions;
using RandomNumbers;
using Random;
using PyCall;

@enum ReactionType begin
	GenReactionType = 0
	UniReactionType = 1
	BiReactionType = 2
end

const ProductType = Union{Int64,Tuple{Float64,Int64}}

# All reactions are instances of this struct
# The fields specA and specB may not always be 
# defined, eg. specB for unimolecular reactions
struct Reaction
	rate :: Float64
	products :: Array{ProductType,1}
	specA :: Int64
	specB :: Int64
	type :: ReactionType
end

GenReaction(rate::Float64, products=[]) = Reaction(rate, products, -1, -1, GenReactionType);

UniReaction(rate::Float64, spec::Int64, products=[]) = Reaction(rate, products, spec, -1, UniReactionType);

BiReaction(rate::Float64, specA::Int64, specB::Int64, products=[]) = Reaction(rate, products, specA, specB, BiReactionType)

const DummyReaction = GenReaction(0.0)
   
struct ReactionSystem
    n_species :: Int64
    reactions :: Array{Reaction,1}   
    initial_state :: Array{Int64,1} 
    reactions_gen :: Array{Reaction,1}
    reactions_uni :: Array{Reaction,1}
    reactions_bi :: Array{Reaction,1}
end

function ReactionSystem(n_species::Int64, reactions::Array{Reaction,1})
    return ReactionSystem(n_species, reactions, zeros(Int64, n_species))
end

function ReactionSystem(n_species :: Int64, reactions :: Array{Reaction,1}, initial_state::Array{Int64,1})
    reactions_gen = Reaction[]
    reactions_uni = Reaction[]
    reactions_bi = Reaction[]

    for reaction in reactions
		if reaction.type == GenReactionType
            push!(reactions_gen, reaction)
        elseif reaction.type == UniReactionType
            push!(reactions_uni, reaction)
        elseif reaction.type == BiReactionType
            push!(reactions_bi, reaction)
        else
            throw(ArgumentError("Unknown reaction type for '$reaction'"))
        end
    end

    return ReactionSystem(n_species, reactions, initial_state, reactions_gen, reactions_uni, reactions_bi);
end

const EventType = Tuple{String, Reaction, Array{Int64,1}}

mutable struct ParticleSystem
    system :: ReactionSystem
    cells :: Array{Int64,1}

    rng :: MersenneTwister

    t :: Float64
    events :: Array{Tuple{Float64,EventType},1}
end

function ParticleSystem(system::ReactionSystem, seed::Union{Nothing,Int64}=nothing)
    cells = zeros(Int64, system.n_species)

    if seed == nothing
        rng = MersenneTwister()
    else
        rng = MersenneTwister(seed)
    end

    ret = ParticleSystem(system, cells, rng, 0, [])
    add_initial_molecules(ret)

    return ret
end

function add_initial_molecules(psys::ParticleSystem)
    @assert psys.t == 0

    for (specp1::Int64, n_init::Int64) in enumerate(psys.system.initial_state)
        product_log = add_products(psys, fill(specp1-1, n_init))

        event = ("gen", DummyReaction, product_log)
        push!(psys.events, (0.0, event))
    end
end

function add_products(psys::ParticleSystem, raw_products::Array{Int64,1})::Array{Int64,1}
    products = raw_products

    for product in products
        psys.cells[product+1] += 1
    end

    return products
end

function add_products(psys::ParticleSystem, raw_products::Array{ProductType,1})::Array{Int64,1}
    products = expand_products(psys, raw_products)

    for product in products
        psys.cells[product+1] += 1
    end

    return products
end

function expand_products(psys::ParticleSystem, raw_products::Array{ProductType,1})::Array{Int64,1}
    ret = Int64[]

    for prod in raw_products
        if typeof(prod) == Int64
            push!(ret, prod)
        else
            m, spec = prod
            p = 1 / m

            dist = Geometric(p)
            n = rand(psys.rng, dist)
            append!(ret, fill(spec, n))
        end
    end

    return ret
end

function compute_bi_rates(psys::ParticleSystem)
    rates = Float64[length(psys.system.reactions_bi)]

    for (i, reac) in enumerate(psys.system.reactions_bi)
        if reac.specA == reac.specB
            combs = 0.5 * psys.cells[reac.specA+1] * (psys.cells[reac.specA+1] - 1)
        else
            combs = psys.cells[reac.specA+1] * psys.cells[reac.specB+1]
        end
            
        rates[i] = reac.rate * combs
    end

    return rates
end

function perform_gen_reaction(psys::ParticleSystem, reaction::Reaction)
    product_log = add_products(psys, reaction.products)
    return ("gen", reaction, product_log)
end

function perform_uni_reaction(psys::ParticleSystem, reaction::Reaction)
    psys.cells[reaction.specA+1] -= 1
    product_log = add_products(psys, reaction.products)
    return ("uni", reaction, product_log)
end

function perform_bi_reaction(psys::ParticleSystem, reaction::Reaction)
    psys.cells[reaction.specA+1] -= 1
    psys.cells[reaction.specB+1] -= 1
    product_log = add_products(psys, reaction.products)
    return ("bi", reaction, product_log)
end
        
mutable struct PBarWrapper
    pbar_raw::PyObject
    buf::Float64
    tmax::Float64
end

function pbar_update_wrapper(pbar_update::PyObject, pbar::PBarWrapper, dt::Float64)
    if pbar.buf + dt >= pbar.tmax / 100.0
        pycall(pbar_update, Any, pbar.pbar_raw, pbar.buf+dt)
        pbar.buf = 0
    else
        pbar.buf += dt
    end
end

function run(psys::ParticleSystem, tmax::Float64; disable_pbar::Bool=true,
             pbar_create = nothing, pbar_update = nothing, pbar_close = nothing)
    t0 = psys.t

    gen_rates = [ reac.rate for reac in psys.system.reactions_gen ]

    pbar = missing
    if !disable_pbar && pbar_create !== nothing
        pbar_raw = pycall(pbar_create, Any, tmax, "Time simulated: ", "tu")
        pbar = PBarWrapper(pbar_raw, 0.0, tmax)
    end

    dt = 0
    while true    
        uni_rates = [ reac.rate * psys.cells[reac.specA+1] for reac in psys.system.reactions_uni ]
        bi_rates = compute_bi_rates(psys)
        rate = sum(gen_rates) + sum(uni_rates) + sum(bi_rates)

        if rate == 0.0 || !isfinite(rate)
            dt = 0
            break
        end

        dt = randexp(psys.rng, Float64) / rate
        psys.t += dt

        if psys.t >= t0 + tmax
            break
        end

        if !ismissing(pbar)
            pbar_update_wrapper(pbar_update, pbar, dt)
        end

        p = rand(psys.rng, Float64) * rate

        if p <= sum(gen_rates)
            for reac in psys.system.reactions_gen
                if p >= reac.rate
                    p -= reac.rate
                    continue
                end

                event = perform_gen_reaction(psys.reac)
                push!(psys.events, (psys.t, event))
                break
            end
        elseif p <= sum(gen_rates) + sum(uni_rates)
            p -= sum(gen_rates)
            
            for reac in psys.system.reactions_uni
                if p >= reac.rate * psys.cells[reac.specA+1]
                    p -= reac.rate * psys.cells[reac.specA+1]
                    continue
                end

                event = perform_uni_reaction(psys, reac)
                push!(psys.events, (psys.t, event))
                break
            end
        else
            p -= (rate - sum(bi_rates))
            
            for (reac, rates) in zip(psys.system.reactions_bi, bi_rates)
                if p >= sum(rates)
                    p -= sum(rates)
                    continue
                end
                    
                event = perform_bi_reaction(psys, reac)
                push!(psys.events, (psys.t, event))
                break
            end
        end

    end

    if !ismissing(pbar)
        pycall(pbar_update, Any, pbar.pbar_raw, pbar.buf + dt - (psys.t - t0 - tmax))
        pycall(pbar_close, Any, pbar.pbar_raw)
    end

    psys.t = t0 + tmax
end
         
function get_dist_data(psys::ParticleSystem)#::Array{Float64}
    return get_dist_data(psys,
                         zeros(Float64, zeros(Int64, psys.system.n_species)...),
                         zeros(Int64, psys.system.n_species),
                         0.0, 
                         0)
end

function get_dist_data(psys::ParticleSystem, 
                       data_old::Array{Float64}, cells_old::Array{Int64,1}, t_min::Float64, n_old_events::Int64)#::Array{Float64}
    counts = Array{Int64,2}(undef, length(psys.events) - n_old_events + 1, psys.system.n_species)
    weights = Array{Float64,1}(undef, length(psys.events) - n_old_events + 1)

    counts[1,:] = cells_old
    
    t_last :: Float64 = t_min
    t :: Float64 = t_min

    i ::Int64 = 1

	reac::Reaction = DummyReaction

    for (t, ev) in psys.events[n_old_events+1:end]
        weights[i] = t - t_last
        t_last = t

        i += 1
        counts[i,:] = counts[i-1,:]
        if ev[1] == "gen"
            product_log = ev[3]
            for spec_product in product_log
                counts[i,spec_product+1] += 1
            end
        elseif ev[1] == "uni"
            reac = ev[2]
            counts[i,reac.specA+1] -= 1

            product_log = ev[3]
            for spec_product in product_log
                counts[i,spec_product+1] += 1
            end
        elseif ev[1] == "bi"
            reac = ev[2]

            counts[i,reac.specA+1] -= 1
            counts[i,reac.specB+1] -= 1

            product_log = ev[3]
            for spec_product in product_log
                counts[i,spec_product+1] += 1
            end
        end
    end

    @assert minimum(counts) >= 0

    weights[end] = psys.t - t_last

    return convert_dist_data(counts, weights, data_old, t_min)
end

function convert_dist_data(counts::Array{Int64,2}, weights::Array{Float64})::Array{Float64}
    return convert_dist_data(counts, weights, zeros(Int64, 1, size(pseudocounts, 2)))
end


function convert_dist_data(counts::Array{Int64,2}, weights::Array{Float64},
                           data_old::Array{Float64}, weight_old::Float64=0)::Array{Float64}
    bounds_old::Tuple{Vararg{Int64}} = size(data_old)
    bounds_new::Array{Int64,1} = [ maximum(counts[:,i]) + 1 for i in 1:size(counts, 2) ]
    @assert length(bounds_old) == length(bounds_new)

    bounds::Array{Int64,1} = [ max(bo, bn) for (bo, bn) in zip(bounds_old, bounds_new) ]
    data_flat::Array{Float64,1} = zeros(Float64, prod(bounds))
	data::Array{Float64} = reshape(data_flat, bounds...)
    
    slices::Array{UnitRange{Int64}} = [ 1:bo for bo in bounds_old ]

	data_strides::Array{Int64,1} = [ s for s in strides(data) ]

    for idx::Tuple{Vararg{Int64}} in Iterators.product(slices...)
		idx_cart = CartesianIndex(idx)
        data[idx_cart] += weight_old * data_old[idx_cart]
    end

    for (i, wt) in enumerate(weights)
		idx_flat = sum(counts[i,:] .* data_strides) + 1
        data_flat[idx_flat] += wt
    end
            
    data_flat ./= sum(data_flat)

    return data
end


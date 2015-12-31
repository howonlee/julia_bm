module bm
using Base.Collections
using DataStructures
using Graphs
using Word2Vec
using Distances
using MNIST

export relax,
  train_bm,
  cool_kids_net,
  mnist_patterns,
  test_bm

function stomp_value(x)
  if x > 0
    return 1
  else
    return 0
  end
end

function logistic(x::Float64)
  1.0 / (1.0 + exp(-x))
end

function round_logistic(x::Float64)
  val = 1.0 / (1.0 + exp(-x))
  if val > 0.5
    return 1
  else
    return 0
  end
end

function relax(bm_vars::Array{Float64}, bm_net::Tuple{Dict{Int, Dict{Int, Float64}}, Dict{Tuple{Int, Int}, Float64}}, pattern::Dict{Int, Float64}, num_iters::Int, dropouts::Set{Int})
  # SA relax... I call it SA, it's GD
  dict_net, tup_net = bm_net
  for var_idx in keys(pattern)
    bm_vars[var_idx] = pattern[var_idx]
  end
  for x in 1:num_iters
    rand_node = rand(1:length(bm_vars)-1)
    while (haskey(pattern, rand_node) || in(rand_node, dropouts))
      rand_node = rand(1:length(bm_vars)-1)
    end
    energy = 0.0
    for neighbor in keys(dict_net[rand_node])
      if in(neighbor, dropouts)
        continue
      end
      energy += bm_vars[neighbor] * tup_net[rand_node, neighbor]
    end
    bm_vars[rand_node] = logistic(energy)
  end
  bm_vars
end

function make_dropouts(bm_vars, pattern, ratio)
  dropouts = Set{Int}()
  for x in 1:length(bm_vars)
    if !(haskey(pattern, x)) && rand() < ratio
      push!(dropouts, x)
    end
  end
  dropouts
end

function train_bm(bm_vars::Array{Float64}, bm_net::Tuple{Dict{Int, Dict{Int, Float64}}, Dict{Tuple{Int, Int}, Float64}}, pattern::Dict{Int, Float64}, pattern_length::Int, num_iters::Int, epsilon::Float64, dropout_ratio::Float64)
  dict_net, tup_net = bm_net
  train_vars = 0.01 * randn(size(bm_vars))
  train_vars[end] = 1
  rand_vars = 0.01 * randn(size(bm_vars))
  rand_vars[end] = 1
  dropouts::Set{Int} = make_dropouts(bm_vars, pattern, dropout_ratio)
  train_vars = relax(train_vars, bm_net, pattern, num_iters, dropouts)
  rand_vars = relax(rand_vars, bm_net, Dict{Int, Float64}(), num_iters, dropouts)

  diffs = Dict()
  for edge in keys(tup_net)
    fst, snd = edge
    if fst > snd || in(fst, dropouts) || in(snd, dropouts)
      continue # once for each edge only
    end
    diffs[fst, snd] = epsilon * ((train_vars[fst] * train_vars[snd]) - (rand_vars[fst] * rand_vars[snd]))
  end
  for pair in keys(diffs)
    fst, snd = pair
    curr_diff = diffs[pair]
    tup_net[fst, snd] += curr_diff
    tup_net[snd, fst] += curr_diff
    dict_net[fst][snd] += curr_diff
    dict_net[snd][fst] += curr_diff
  end
  train_vars, bm_net
end

function test_bm(bm_vars, bm_net, pattern_part, total_length, num_iters, dropouts)
  # requires contiguity of the pattern which begins at the beginning
  test_vars = copy(bm_vars)
  test_vars[end] = 1
  for x in 1:100
    println(x)
    test_vars = relax(test_vars, bm_net, pattern_part, num_iters, dropouts)
  end
  test_vars
end

function make_edges(num_edges_per_node, prev_node_rng, new_node_rng)
  edges = []
  for head_node in prev_node_rng
    edge_set = Set()
    for x in 1:num_edges_per_node
      tail_node = new_node_rng[rand(1:end)]
      while (head_node, tail_node) in edge_set
        tail_node = new_node_rng[rand(1:end)]
      end
      push!(edge_set, (head_node, tail_node))
    end
    for edge in edge_set
      push!(edges, edge)
    end
  end
  return edges
end

function layer_range(layer_number, start_nodes, layer_size)
  if layer_number == 1
    return Array(1:start_nodes)
  else
    return Array(((start_nodes + (layer_size * (layer_number - 2))) + 1):(start_nodes + (layer_size * (layer_number - 1))))
  end
end

function create_net(init_layer, num_layers, layer_size, num_edges_per_node)
  net = simple_inclist((init_layer + (num_layers * layer_size)), is_directed=false)
  for x in 1:(num_layers-1)
    for edge in make_edges(num_edges_per_node, layer_range(x, init_layer, layer_size), layer_range(x+1, init_layer, layer_size))
      fst, snd = edge
      add_edge!(net, fst, snd)
    end
  end
  net
end

function cool_kids_net(init_size, num_layers, layer_size, num_edges_per_node, weight_stddev)
  # call it the cool_kids_net because it's for the cool kids
  # less facetiously, it's basically like a DBN
  # sparse and extremely, extremely stupid, for easy debugging purposes
  net_size = init_size + (num_layers * layer_size)
  bm_vars = zeros(net_size + 1)
  bm_vars[end] = 1
  dict_net, tup_net = Dict{Int, Dict{Int, Float64}}(), Dict{Tuple{Int, Int}, Float64}()
  net = create_net(init_size, num_layers, layer_size, num_edges_per_node)
  println("made edges")
  ctr = 0
  for edge in collect_edges(net)
    fst, snd = edge.source, edge.target
    ctr += 1
    if ctr % 10000 == 0
      println((fst, snd))
    end
    weight = randn() * weight_stddev
    if !haskey(dict_net, fst)
      dict_net[fst] = Dict()
    end
    if !haskey(dict_net, snd)
      dict_net[snd] = Dict()
    end
    dict_net[fst][snd] = weight
    dict_net[snd][fst] = weight
    tup_net[fst, snd] = weight
    tup_net[snd, fst] = weight
  end
  for x in 1:net_size
    weight = randn() * weight_stddev
    # fill it in, friends
    if !haskey(dict_net, x)
      dict_net[x] = Dict()
    end
    if !haskey(dict_net, net_size+1)
      dict_net[net_size+1] = Dict()
    end
    dict_net[x][net_size+1] = weight
    dict_net[net_size+1][x] = weight
    tup_net[x, net_size+1] = weight
    tup_net[net_size+1, x] = weight
  end
  bm_net = dict_net, tup_net
  println("made net")
  bm_vars, bm_net
end

function mnist_patterns(num_pats::Int)
  patterns = []
  println("stating pattern reading...")
  for i=1:num_pats
    data = trainfeatures(i)
    pattern::Dict{Int, Float64} = Dict([x => stomp_value(data[x]) for x = 1:784])
    push!(patterns, pattern)
  end
  println("created the patterns")
  patterns
end

end # module bm


include("bm.jl")

init_size = parse(Int, ARGS[1])
layer_size = parse(Int, ARGS[2])
num_layers = parse(Int, ARGS[3])
num_edges_per_node = parse(Int, ARGS[4])
weight_stddev = parse(Float64, ARGS[5])
num_patterns = parse(Int, ARGS[6])
num_iters = parse(Int, ARGS[7])
epsilon = parse(Float64, ARGS[8])
dropout_ratio = parse(Float64, ARGS[9])
net_filename = ARGS[10]

function sum_weights(tup_net)
  total = 0
  for key in keys(tup_net)
    total += tup_net[key]
  end
  total
end

function sum_abs_weights(tup_net)
  total = 0
  for key in keys(tup_net)
    total += abs(tup_net[key])
  end
  total
end

function deep_weights(num_vars, num_viz, net)
  dict_net, tup_net = net
  deepness = 0.0
  abs_deepness = 0.0
  for member in keys(dict_net)
    if member > num_viz
      filter_dict = dict_net[member]
      for filter_key in keys(filter_dict)
        if filter_key > num_viz
          deepness += filter_dict[filter_key]
          abs_deepness += abs(filter_dict[filter_key])
        end
      end
    end
  end
  println("deep stuff")
  println(deepness)
  println("abs deep stuff")
  println(abs_deepness)
end

bm_vars, bm_net = bm.cool_kids_net(init_size, num_layers, layer_size, num_edges_per_node, weight_stddev)
pats = bm.mnist_patterns(num_patterns)
pat_num = 0
println(Libc.strftime(time()))
for pattern in pats
  pat_num += 1
  println(pat_num)
  println(Libc.strftime(time()))
  if pat_num % 50 == 0
    deep_weights(length(bm_vars), 784, bm_net)
  end
  for l in 1:num_layers
    curr_pat = Dict{Int, Float64}()
    layer = bm.layer_range(l, init_size, layer_size)
    for pair in enumerate(bm_vars)
      idx, val = pair
      if idx <= layer[end]
        curr_pat[idx] = val
      end
    end
    for key in keys(pattern)
      curr_pat[key] = pattern[key]
    end
    bm_vars, bm_net = bm.train_bm(bm_vars, bm_net, curr_pat, length(curr_pat), num_iters, epsilon, dropout_ratio)
  end
end
println(Libc.strftime(time()))

net_file = open(net_filename, "w+")
println(typeof(bm_net))
serialize(net_file, bm_net)
close(net_file)

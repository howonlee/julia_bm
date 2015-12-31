include("bm.jl")
using PyPlot

function try_net(num_vars, num_iters, file_name)
  net_file = open(file_name)
  pats = bm.mnist_patterns(400)
  bm_net = deserialize(net_file)
  close(net_file)
  for x in 1:10
    test_vars = 0.1 * randn(num_vars + 1)
    for key in keys(pats[x])
      test_vars[key] = pats[x][key]
    end
    test_vars[end] = 1
    close() # close pyplot
    print("sampling : ")
    println(x)
    for y in 1:50
      println(y)
      test_vars = bm.relax(test_vars, bm_net, Dict{Int, Float64}(), num_iters, Set{Int}())
    end
    test_sample = test_vars[1:784]
    imshow(reshape(test_sample, (28,28)))
    colorbar()
    title(file_name)
    println("./samples/" * string(x))
    savefig("./samples/" * string(x))
  end
  println("all saved")
end

function try_net_old(num_vars, num_iters, file_name)
  net_file = open(file_name)
  bm_net = deserialize(net_file)
  close(net_file)
  pats = bm.mnist_patterns(11000)
  test_patterns = [pats[8000]]
  partial_patterns = Set{Dict{Int, Float64}}()
  for test_pattern in test_patterns
    filtered_pattern = Dict{Int, Float64}()
    for key in keys(test_pattern)
      if key > 350
        filtered_pattern[key] = test_pattern[key]
      end
    end
    push!(partial_patterns, filtered_pattern)
  end
  bm_vars = zeros(num_vars + 1)
  bm_vars[end] = 1
  test_vars = zeros(num_vars + 1)
  bm_vars[end] = 1
  for x in 1:10
    for partial_pattern in partial_patterns
      test_vars += bm.test_bm(bm_vars, bm_net, partial_pattern, 784, num_iters, Set{Int}())
    end
  end
  test_sample = test_vars[1:784]
  print(test_sample)
  print(test_vars)
  imshow(reshape(test_sample, (28,28)))
  colorbar()
  title(file_name)
  show()
end

# assert len(ARGS) == 5
num_vars = parse(Int, ARGS[1])
num_iters = parse(Int, ARGS[2])
file_name = ARGS[3]
try_net(num_vars, num_iters, file_name)
# print_net(file_name)

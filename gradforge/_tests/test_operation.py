import timeit

setup = "from gradforge.core.tensor import Tensor; a = Tensor([5]); b = Tensor([6])"

result = timeit.repeat("c = a + b", setup=setup, number=100000, repeat=10)

best_time = min(result)         
time_per_op = best_time / 100_000
print(time_per_op * 1000000)

#0.8082507399922179

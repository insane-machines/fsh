from gradforge.core.tensor import Tensor
import tracemalloc

tracemalloc.start()
before = tracemalloc.take_snapshot()

a = Tensor([5])
b = Tensor([6])

for _ in range(100000):
    c = a + b

after = tracemalloc.take_snapshot()
stats = after.compare_to(before, "lineno")

for stat in stats[:20]:
    print(stat)
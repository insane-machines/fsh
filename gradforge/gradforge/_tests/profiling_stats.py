import pstats

stats = pstats.Stats("operation.prof")

stats.sort_stats("tottime")
stats.print_stats(30)
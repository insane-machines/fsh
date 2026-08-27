import pstats

stats = pstats.Stats("profile.prof")

stats.sort_stats("cumtime")
stats.print_stats(30)
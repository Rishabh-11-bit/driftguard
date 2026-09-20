"""driftguard.benchmark package."""
from .labelling import BenchmarkEntry, save_benchmark, load_benchmark, benchmark_stats

__all__ = ["BenchmarkEntry", "save_benchmark", "load_benchmark", "benchmark_stats"]

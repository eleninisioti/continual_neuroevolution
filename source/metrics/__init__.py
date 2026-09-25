"""What a run measures about itself while it runs.

Every module here is a pure observer: it takes a key from a private stream
(`seed + N_000_000`, the convention `track_diversity` established) and never
returns anything to the search. A run with these on and one with them off are
the same run, bit for bit. Preserve that when adding to this package.
"""

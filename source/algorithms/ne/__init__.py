"""The neuro-evolution search algorithms, independent of any task.

    es.py         ES: one class, variant set by `shaping` and `optimizer`
    ga.py         the GA (`ga`) and the consolidating GA Kinetix uses
                  (`ga_focus`)
    dns.py        Dominated Novelty Search (`dns`, `dns_gaussian`)
    variation.py  the gaussian and Iso+LineDD operators the GA and DNS breed with
    searchers.py  the shared ask/tell interface and `build_searcher`
"""

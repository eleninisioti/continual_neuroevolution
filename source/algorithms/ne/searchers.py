"""The neuroevolution methods, behind one ask/tell interface.

`source/runners/train_nes.py` runs the schedule, evaluates, and writes the
artifacts; none of that is method-specific. What differs between ES, GA and
DNS is only how a generation is proposed and how the survivors are chosen, so
that is all the searchers hold. One runner, one searcher per method, and the
comparison cannot drift into being a comparison of different training loops.

Every searcher exposes:

    init(key, mean)                -> state
    ask(key, state)                -> (population, aux)
    tell(state, aux, fitness, descriptors=None) -> state
    incumbent(state)               -> one flat parameter vector
    population_mean(state)         -> one flat parameter vector
    population(state)              -> (n, num_params)

``fitness`` is always the return, MAXIMISED.

``incumbent`` is the single agent a run would hand you, and it differs by
method: the distribution mean for ES, the best archive member for the GA, the
fittest repertoire member for DNS.

``needs_descriptors`` says whether ``tell`` requires behaviour descriptors.
Only DNS does, and the runner only pays for computing them when it is asked to.

The methods:

    es             `source/algorithms/ne/es.py`   variant set by `shaping`
                                                  and `optimizer`
    ga             `source/algorithms/ne/ga.py`   gymnax, MiniGrid, MJX
    ga_focus       `source/algorithms/ne/ga.py`   Kinetix
    dns            `source/algorithms/ne/dns.py`  Iso+LineDD
    dns_gaussian   `source/algorithms/ne/dns.py`  the GA's gaussian mutation
"""

from __future__ import annotations

from source.algorithms.ne.dns import DNSSearcher
from source.algorithms.ne.es import ESSearcher
from source.algorithms.ne.ga import FocusGASearcher, GASearcher
from source.algorithms.ne.variation import GAUSSIAN, ISOLINE

ES_METHODS = ('es',)
GA_METHODS = ('ga', 'ga_focus')
DNS_METHODS = ('dns', 'dns_gaussian')
NE_METHODS = ES_METHODS + GA_METHODS + DNS_METHODS

# The keyword arguments each searcher takes. The runner passes everything it
# has (`sigma_init`, `learning_rate`, ... plus the config's
# `searcher_kwargs`), and each searcher gets only its own.
_ES_KWARGS = ('sigma_init', 'learning_rate', 'optimizer', 'shaping',
              'sigma_lr', 'momentum')
_GA_KWARGS = ('elite_ratio', 'sigma_init', 'cross_over_rate', 'init_scale',
              'init_around_mean')
_FOCUS_KWARGS = _GA_KWARGS + ('focus_rate', 'sigma_rate', 'track_target',
                              'sigma_min', 'explore_fraction')
_DNS_KWARGS = ('iso_sigma', 'line_sigma', 'k', 'normalize_descriptors',
               'repertoire_ratio', 'traj_steps', 'obs_dim', 'init_scale',
               'init_around_mean', 'sigma_init', 'cross_over_rate')


def _pick(kwargs, names):
    return {k: v for k, v in kwargs.items() if k in names}


def build_searcher(method, num_params, population_size, descriptor_dim=2,
                   **kwargs):
    """One place that knows which settings define which method."""
    # `sigma` is the name every caller uses for a mutation scale; accept it
    # as an alias of `sigma_init`.
    if 'sigma' in kwargs:
        kwargs = dict(kwargs)
        kwargs['sigma_init'] = kwargs.pop('sigma')
    if method == 'es':
        return ESSearcher(num_params, population_size,
                          **_pick(kwargs, _ES_KWARGS))
    if method == 'ga':
        return GASearcher(num_params, population_size,
                          **_pick(kwargs, _GA_KWARGS))
    if method == 'ga_focus':
        return FocusGASearcher(num_params, population_size,
                               **_pick(kwargs, _FOCUS_KWARGS))
    if method in DNS_METHODS:
        return DNSSearcher(num_params, population_size, descriptor_dim,
                           variation=(GAUSSIAN if method == 'dns_gaussian'
                                      else ISOLINE),
                           **_pick(kwargs, _DNS_KWARGS))
    raise ValueError(f'unknown NE method {method!r}; expected one of '
                     f'{NE_METHODS}')

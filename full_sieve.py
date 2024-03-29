#!/usr/bin/env python
# -*- coding: utf-8 -*-
####
#
#   Copyright (C) 2018-2021 Team G6K
#
#   This file is part of G6K. G6K is free software:
#   you can redistribute it and/or modify it under the terms of the
#   GNU General Public License as published by the Free Software Foundation,
#   either version 2 of the License, or (at your option) any later version.
#
#   G6K is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with G6K. If not, see <http://www.gnu.org/licenses/>.
#
####


"""
Full Sieve Command Line Client
"""

from __future__ import absolute_import
import pickle as pickler
from collections import OrderedDict

from g6k.algorithms.workout import workout
from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer
from g6k.utils.util import (
    load_svpchallenge_and_randomize,
    load_matrix_file,
    db_stats,
    extra_memory,
    extra_usage,
    sieving_bandwidth,
    center_bandwidth,
    bucket_bandwidth,
)
from g6k.utils.util import sanitize_params_names, print_stats, output_profiles
import six


def full_sieve_kernel(arg0, params=None, seed=None):
    # Pool.map only supports a single parameter
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    pump_params = pop_prefixed_params("pump", params)
    verbose = params.pop("verbose")

    # NOTE: the regular variant of G6K uses reserved_n=n here.
    # This requires some extra support in the distributed sieving
    # code, as we need to ensure the reserved memory is spread across
    # all nodes. 
    reserved_n = n
    params = params.new(reserved_n=reserved_n, otf_lift=False)

    challenge_seed = params.pop("challenge_seed")
    A, _ = load_svpchallenge_and_randomize(n, s=challenge_seed, seed=seed)

    g6k = Siever(A, params, seed=seed)
    tracer = SieveTreeTracer(g6k, root_label=("full-sieve", n), start_clocks=True)

    # Actually runs a workout with very large decrements, so that the basis is kind-of reduced
    # for the final full-sieve
    workout(
        g6k,
        tracer,
        0,
        n,
        dim4free_min=0,
        dim4free_dec=15,
        pump_params=pump_params,
        verbose=verbose,
    )

    return tracer.exit()


def full_sieve():
    """
    Run a a full sieve (with some partial sieve as precomputation).
    """
    description = full_sieve.__doc__

    args, all_params = parse_args(description, challenge_seed=0)

    stats = run_all(
        full_sieve_kernel,
        list(all_params.values()),
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        step_size=args.step_size,
        trials=args.trials,
        workers=args.workers,
        seed=args.seed,
    )

    inverse_all_params = OrderedDict([(v, k) for (k, v) in six.iteritems(all_params)])
    stats = sanitize_params_names(stats, inverse_all_params)

    fmt = "{name:50s} :: n: {n:2d}, cputime {cputime:7.4f}s, walltime: {walltime:7.4f}s, |db|: 2^{avg_max:.2f}, extra_memory(MB): {extra_memory:.2f}, ratio: {extra_used_ratio:.2f}, bandwidth (MB): {sieving_bandwidth:.2f}, center bandwidth (MB): {center_bandwidth:.2f}, bucket bandwidth(MB): {bucket_bandwidth:.2f}"
    profiles = print_stats(
        fmt,
        stats,
        (
            "cputime",
            "walltime",
            "avg_max",
            "extra_memory",
            "extra_used_ratio",
            "sieving_bandwidth",
            "center_bandwidth",
            "bucket_bandwidth",
        ),
        extractf={
            "avg_max": lambda n, params, stat: db_stats(stat)[0],
            "extra_memory": lambda n, params, stat: extra_memory(stat)[0],
            "extra_used_ratio": lambda n, params, stat: extra_usage(stat)[0],
            "sieving_bandwidth": lambda n, params, stat: sieving_bandwidth(stat)[0],
            "center_bandwidth": lambda n, params, stat: center_bandwidth(stat)[0],
            "bucket_bandwidth": lambda n, params, stat: bucket_bandwidth(stat)[0],
        },
    )

    output_profiles(args.profile, profiles)

    if args.pickle:
        pickler.dump(
            stats,
            open(
                "full-sieve-%d-%d-%d-%d.sobj"
                % (args.lower_bound, args.upper_bound, args.step_size, args.trials),
                "wb",
            ),
        )


if __name__ == "__main__":    
    full_sieve()

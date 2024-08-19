import itertools
import numpy as np
import scipy.stats as stats
import multiprocessing
from joblib import Parallel, delayed



def get_significant_events(scores, shuffled_scores, q=95, tail="both"):
    """
    Return the significant events based on percentiles,
    the p-values and the standard deviation of the scores
    in terms of the shuffled scores.
    Parameters
    ----------
    scores : array of shape (n_events,)
        The array of scores for which to calculate significant events
    shuffled_scores : array of shape (n_shuffles, n_events)
        The array of scores obtained from randomized data
    q : float in range of [0,100]
        Percentile to compute, which must be between 0 and 100 inclusive.
    Returns
    -------
    sig_event_idx : array of shape (n_sig_events,)
        Indices (from 0 to n_events-1) of significant events.
    pvalues : array of shape (n_events,)
        The p-values
    stddev : array of shape (n_events,)
        The standard deviation of the scores in terms of the shuffled scores
    """
    # check shape and correct if needed
    if isinstance(scores, list) | isinstance(scores, np.ndarray):
        if shuffled_scores.shape[1] != len(scores):
            shuffled_scores = shuffled_scores.T

    n = shuffled_scores.shape[0]
    if tail == "both":
        r = np.sum(np.abs(shuffled_scores) >= np.abs(scores), axis=0)
    elif tail == "right":
        r = np.sum(shuffled_scores >= scores, axis=0)
    elif tail == "left":
        r = np.sum(shuffled_scores <= scores, axis=0)
    else:
        raise ValueError("tail must be 'left', 'right', or 'both'")
    pvalues = (r + 1) / (n + 1)

    # set nan scores to 1
    if isinstance(np.isnan(scores), np.ndarray):
        pvalues[np.isnan(scores)] = 1

    sig_event_idx = np.argwhere(
        scores > np.percentile(shuffled_scores, axis=0, q=q)
    ).squeeze()

    # calculate how many standard deviations away from shuffle
    stddev = (np.abs(scores) - np.nanmean(np.abs(shuffled_scores), axis=0)) / np.nanstd(
        np.abs(shuffled_scores), axis=0
    )

    return np.atleast_1d(sig_event_idx), np.atleast_1d(pvalues), np.atleast_1d(stddev)


def similarity_index(patterns, n_shuffles=1000, parallel=True):
    """
    Calculate the similarity index of a set of patterns.

    Based on Almeida-Filho et al., 2014 to detect similar assemblies.

    To use a quantitative criterion to compare assembly composition,
    a Similarity Index (SI) was defined as the absolute value of the
    inner product between the assembly patterns (unitary vectors) of
    two given assemblies, varying from 0 to 1. Thus, if two assemblies
    attribute large weights to the same neurons, SI will be large;
    if assemblies are orthogonal, SI will be zero.

    Input:
        patterns: list of patterns (n patterns x n neurons)
        n_shuffles: number of shuffles to calculate the similarity index
    Output:
        si: similarity index: float (0-1)
        combos: list of all possible combinations of patterns
        pvalues: list of p-values for each pattern combination
    """
    # check to see if patterns are numpy arrays
    if not isinstance(patterns, np.ndarray):
        patterns = np.array(patterns)

    # check if all values in matrix are less than 1
    if not all(i <= 1 for i in patterns.flatten()):
        raise ValueError("All values in matrix must be less than 1")

    # shuffle patterns over neurons
    def shuffle_patterns(patterns):
        return np.random.permutation(patterns.flatten()).reshape(patterns.shape)

    # calculate absolute inner product between patterns
    def get_si(patterns,return_combo=False):
        x = np.arange(0, patterns.shape[0])
        # use itertools to get all combinations of patterns
        combos = np.array(list(itertools.combinations(x, 2)))
        si = []
        for s in combos:
            si.append(np.abs(np.inner(patterns[s[0], :], patterns[s[1], :])))

        if return_combo:
            return np.array(si), combos
        else:
            return np.array(si)

    # calculate observed si
    si, combos = get_si(patterns,return_combo=True)

    # shuffle patterns and calculate si
    if parallel:
        num_cores = multiprocessing.cpu_count()
        si_shuffles = Parallel(n_jobs=num_cores)(
            delayed(get_si)(shuffle_patterns(patterns)) for _ in range(n_shuffles)
        )
    else:
        si_shuffles = [get_si(shuffle_patterns(patterns)) for _ in range(n_shuffles)]

    # calculate p-values for each pattern combination
    _, pvalues, _ = get_significant_events(si, np.array(si_shuffles))

    return si, combos, pvalues

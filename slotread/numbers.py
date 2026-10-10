"""Numbers as bins: soft ordinal targets for training, and readouts from a slot's label log-probs.

A number lives in [0, 1]. Positions are linear; sizes and radii are log-warped first, because UI
elements and objects span three orders of magnitude and a uniform grid would spend its resolution
where nothing lives.

A number is one fine slot over `arm.bins` labels, plus (coarse arms only) a coarse slot over the ten
digits: the tenth of the range the number falls in.
"""

import math

import torch

LOG_MIN = 1 / 512  # the smallest size / radius a log-warped number can say, as a fraction


def warp(v, kind):
    if kind == "lin":
        return min(max(v, 0.0), 1.0)
    return min(max(math.log(max(v, LOG_MIN) / LOG_MIN) / math.log(1 / LOG_MIN), 0.0), 1.0)


def unwarp(u, kind):
    return u if kind == "lin" else LOG_MIN * (1 / LOG_MIN) ** u


def slots_for(number, arm):
    """The slots that write one number: [x] or, in a coarse arm, [xc, x]."""
    return [number + "c", number] if arm.coarse else [number]


def centers(n):
    return (torch.arange(n) + 0.5) / n


def gauss(c, mu, sigma):
    """A normalized Gaussian over bin centres c: the soft ordinal target (one bin off costs little)."""
    w = torch.exp(-((c - mu) ** 2) / (2 * sigma ** 2))
    return w / w.sum()


def number_targets(number, u, sigma, arm):
    """Soft targets for one number at warped value u; sigma in warped units, floored at a bin."""
    out = {number: gauss(centers(arm.bins), u, max(0.6 / arm.bins, sigma))}
    if arm.coarse:
        out[number + "c"] = gauss(centers(10), u, max(0.6 / 10, sigma))
    return out


def local_peak(p):
    """The argmax bin refined by the probability-weighted mean of its +-2 neighbours, in [0, 1]."""
    p = p.detach().float().cpu()
    k, n = int(p.argmax()), len(p)
    lo, hi = max(0, k - 2), min(n, k + 3)
    w = p[lo:hi]
    return float(((torch.arange(lo, hi, dtype=torch.float32) + 0.5) * w).sum() / w.sum()) / n


def read_number(lp, number, arm, how="default"):
    """A number's warped value from its slots' label log-probabilities.

    peak      the argmax bin
    local2    the peak refined by +-2 bins (the default without a coarse slot)
    fused     the fine distribution times the coarse slot's probability for the tenth each bin falls
              in, then local2 (the default with a coarse slot)
    expected  the mean over all bins
    """
    p = torch.softmax(lp[number].detach().float().cpu(), -1)
    c = centers(len(p))
    if how == "default":
        how = "fused" if arm.coarse else "local2"
    if how == "peak":
        return float(c[int(p.argmax())])
    if how == "expected":
        return float((c * p).sum())
    if how == "fused" and number + "c" in lp:
        prior = torch.softmax(lp[number + "c"].detach().float().cpu(), -1)[(c * 10).long().clamp(max=9)]
        p = p * prior
    return local_peak(p)


def one_hot(n, i):
    t = torch.zeros(n)
    t[i] = 1.0
    return t

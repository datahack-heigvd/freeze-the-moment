"""Bounded cylindrical kit support fill."""

import numpy as np
from scipy.ndimage import distance_transform_edt


def fill_vote_bins(votes):
    """Fill unsupported bins from the nearest supported bin, periodic in angle.

    Supported bins are unchanged. An entirely unsupported part stays empty so
    its caller can retain the baked fallback instead of fabricating a colour.
    """
    votes = np.asarray(votes)
    if votes.ndim != 3 or any(size == 0 for size in votes.shape):
        raise ValueError("Expected nonempty axial x angular x colour votes")
    if not np.isfinite(votes).all() or (votes < 0).any():
        raise ValueError("Votes must be finite and nonnegative")
    supported = votes.sum(axis=-1) > 0
    if not supported.any() or supported.all():
        return votes.copy(), supported
    angles = votes.shape[1]
    periodic = np.tile(votes, (1, 3, 1))
    indices = distance_transform_edt(~np.tile(supported, (1, 3)), return_distances=False, return_indices=True)
    filled = periodic[indices[0], indices[1]][:, angles : 2 * angles].copy()
    return filled, supported

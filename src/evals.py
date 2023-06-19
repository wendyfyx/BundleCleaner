import numpy as np
from dipy.tracking import utils
from dipy.io.image import load_nifti
from dipy.segment.bundles import bundle_adjacency, bundle_shape_similarity
from nibabel.streamlines.array_sequence import ArraySequence


def bundle_density_map(bundle, ref_path):
    ref_img, affine=load_nifti(ref_path)
    return utils.density_map(bundle, affine, ref_img.shape)


def bundle_sm(b1, b2, rng=None):
    '''Return bundle shape similarity score.'''
    if rng is None:
        rng=np.random.default_rng(0)
    elif isinstance(rng, int):
        rng=np.random.default_rng(rng)
    
    if not isinstance(b1, ArraySequence):
        b1 = ArraySequence(b1)
    if not isinstance(b2, ArraySequence):
        b2 = ArraySequence(b2)

    rng = np.random.default_rng(0)
    clust_thr=(5, 3, 1.5)
    threshold=6
    return bundle_shape_similarity(b1, b2, rng, clust_thr, threshold)


def bundle_ba(b1, b2, threshold=3):
    '''Comute bundle adjacency'''
    if not isinstance(b1, ArraySequence):
        b1 = ArraySequence(b1)
    if not isinstance(b2, ArraySequence):
        b2 = ArraySequence(b2)
    return bundle_adjacency(b1, b2, threshold=threshold)


def binary_mask(arr):
    '''Return binary mask (0/1) given array.'''
    mask = np.zeros_like(arr)
    mask[arr>0]=1
    return mask
    

def dice(map1, map2):
    '''Returns the dice coefficient given two volumes of the same size.'''
    assert map1.shape==map2.shape, "Shape mismatch"
    mask1 = binary_mask(map1)
    mask2 = binary_mask(map2)
    intersection = np.logical_and(mask1, mask2)
    return np.sum(intersection)*2.0 / (np.sum(mask1) + np.sum(mask2))


def coverage(map1, map2):
    '''
        Returns the coverage for two volumes of the same size, the fraction of 
        volume in map1 overlapping with map2, and the fraction of volume in map2 
        overlapping with map1.
    '''
    assert map1.shape==map2.shape, "Shape mismatch"
    mask1 = binary_mask(map1)
    mask2 = binary_mask(map2)
    intersection = np.logical_and(mask1, mask2)
    return np.sum(intersection) / np.sum(mask1), np.sum(intersection) / np.sum(mask2)

import numpy as np
# import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter

import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
from dipy.tracking.streamline import set_number_of_points, transform_streamlines
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import mdf
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from fury.colormap import line_colors
from fury.utils import map_coordinates_3d_4d


# def bundle2lineset(bundle, colors=None):
#     '''
#         Convert bundle to lineset object in Open3d
#         Accept numpy array or ArraySequence
#     '''
#     if isinstance(bundle, np.ndarray):
#         points = bundle.reshape(-1, 3)
#         repeat_param = bundle.shape[1]-1
#     elif isinstance(bundle, ArraySequence):
#         points = bundle.get_data()
#         repeat_param = np.array([len(line)-1 for line in bundle])
#     else:
#         print("Unsupported data type.")
#         return
    
#     edges = []
#     count = 0
#     for i in range(len(bundle)):
#         for j in range(len(bundle[i])-1):
#             edges.append([count+j, count+j+1])
#         count += len(bundle[i])
        
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(points)
#     line_set.lines = o3d.utility.Vector2iVector(edges)
#     if colors is None:
#         colors = line_colors(bundle)
#     if colors=='random':
#         colors = np.array([np.random.rand(3) for si in range(len(bundle))])
#     elif not isinstance(colors, np.ndarray):
#         colors = np.array(colors).reshape(-1, 3)
#         if len(colors)==1:
#             colors = np.tile(colors, (len(bundle), 1))
#     print(colors.shape)
#     colors = np.repeat(colors, repeat_param, axis=0)
#     line_set.colors = o3d.utility.Vector3dVector(colors)
#     return line_set

# def bundle2pc(bundle, reshape=True, colors=None):
#     '''
#         Convert bundle to point cloud object in Open3d
#         Accept numpy array or ArraySequence
#     '''
#     if reshape:
#         points = bundle.reshape(-1, 3)
#     else:
#         points = bundle
#     pc = o3d.geometry.PointCloud()
#     pc.points = o3d.utility.Vector3dVector(points)
#     if colors is not None:
#         if colors=='random':
#              colors = [np.random.rand(3) for si in range(len(points))]
#         elif not isinstance(colors, np.ndarray):
#             colors = np.array(colors).reshape(-1, 3)
#             if len(colors)==1:
#                 colors = np.tile(colors, (len(points), 1))
#         pc.colors = o3d.utility.Vector3dVector(colors)
#     return pc


def centroid_mdf(b1, b2, threshold=20):
    '''Compute MDF distance between bundle centroid'''
    qb1 = QuickBundles(threshold=threshold)
    qb2 = QuickBundles(threshold=threshold)
    clt1 = qb1.cluster(b1)
    clt2 = qb2.cluster(b2)
    return mdf(clt1.centroids[0], clt2.centroids[0])


def load_nib(fpath):
    '''Load .nii files'''
    img = nib.load(fpath)
    return img.get_fdata(), img.affine


def dti2bundle(dti_fpath, lines_org):
    '''Map DTI valume to streamlines, returns DTI metric for each points in lines_org'''
    dtimap, affine = load_nib(dti_fpath)
    X_native = transform_streamlines(lines_org, np.linalg.inv(affine))
    if isinstance(X_native, ArraySequence):
        X_native = X_native.get_data()
    return map_coordinates_3d_4d(dtimap, X_native)


def resample_lines_by_percent(lines, percent=0.5, verbose=True):
    '''Subsample each streamline with set percentage'''
    if verbose:
        print(f"Resample each streamline to {100*percent}% of points.")
    ptct = [len(l) for l in lines]
    lines = [set_number_of_points(lines[i], int(percent*ptct[i])) for i in range(len(lines))]
    return ArraySequence(lines)


def resample_lines_by_ptct(lines, ptct, verbose=True):
    lines = [set_number_of_points(lines[i], ptct[i]) for i in range(len(lines))]
    return ArraySequence(lines)


def save_bundle(bundle, orig_fpath, new_fpath, verbose=True):
    if verbose:
        print(f"Saving bundle ({len(bundle)} lines) to {new_fpath}.")
    new_tractogram = StatefulTractogram(bundle, orig_fpath, Space.RASMM)
    save_tractogram(new_tractogram, new_fpath, bbox_valid_check=False)


def value2color(values, plot_cmap=True, vmin=None, vmax=None, cmap_name='jet'):
    '''Map values to color using the jet colormap'''
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = cmap.to_rgba(values)[:,:3]
    
    if plot_cmap:
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        fig.colorbar(cmap, cax=ax, orientation='horizontal')
    return colors

def pc_to_arraysequence(pc, ptct):
    '''Convert point cloud (Nx3) to array sequence, given point count per streamline'''
    ls = []
    curr_idx = 0
    for i in ptct:
        if len(ls) == 0:
            ls.append(pc[:i])
        else:
            ls.append(pc[curr_idx:curr_idx+i])
        curr_idx += i
    return ArraySequence(ls)


def random_select(data, n_sample=1, rng=None):
    '''
        Randomly select samples from data, returns index
        Example usage: data[random_select(data)]
    '''
    if rng is None:
        rng=np.random.default_rng(0)
    elif isinstance(rng, int):
        rng=np.random.default_rng(rng)
    return rng.choice(len(data), size=n_sample, replace=False)


def cluster_subsample(labels, sample_pct=0.5, min_samples=2, rng=0):
    '''Subsample from each cluster, and discard clusters smaller than min_samples'''
    if isinstance(rng, int):
        rng=np.random.default_rng(rng)
        
    new_idx = np.array([])
    for k, v in Counter(labels).items():
        clt_idx = np.nonzero(labels==k)[0]
        if v <= min_samples:
            continue
        else:
            sample_idx = clt_idx[random_select(clt_idx, round(v*sample_pct), rng)]
        new_idx = np.concatenate((new_idx, sample_idx))
    return np.sort(new_idx).astype(int)


def qbcluster_labels(clusters):
    '''Helper function for getting labesl from QuickBundles labels'''
    clt_map = []
    labels = np.zeros(sum(list(map(len, clusters)))).astype(int)
    for i in range(len(clusters)):
        size = len(clusters[i].indices)
        clt_map.extend(list(zip(clusters[i].indices, np.full((size,), i))))
    for k, v in clt_map:
        labels[k] = v
    return clt_map, labels
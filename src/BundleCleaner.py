import time
import scipy
import numpy as np
import robust_laplacian
from collections import Counter
from scipy.signal import savgol_filter

from nibabel.streamlines.array_sequence import ArraySequence
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric

from utils import pc_to_arraysequence, qbcluster_labels, cluster_subsample, random_select


class BundleCleaner:
    def __init__(self, data):
        self.data=data
        self.ptct = [len(i) for i in self.data]
        self.pc_smoothed = None
        
    def run(self, prune_threshold=5, prune_min_samples=None, verbose=False, **kwargs):
        pc = self.data.get_data()
        self.npoints_orig = len(pc)
        if verbose:
            print(f"Cleaning bundle of size {pc.shape}")
        
        start = time.time()
        # Step 1 - Pruning
        prune_idx, _, self.min_samples_prune = BundleCleaner.subsampling(self.data, threshold=prune_threshold, 
                                                min_samples=prune_min_samples, 
                                                sample_pct=1, verbose=verbose)
        self.lines_pruned = self.data[prune_idx]
        self.ptct = [len(i) for i in self.lines_pruned]
        pc = self.lines_pruned.get_data()
        
        # Step 2 - Laplacian Smoothing
        if self.pc_smoothed is None:
            self.pc_smoothed = BundleCleaner.laplacian_smoothing(pc, verbose=verbose, **kwargs)
            self.lines_smoothed = pc_to_arraysequence(self.pc_smoothed, self.ptct)
            if verbose:
                norm1 = scipy.linalg.norm(pc-self.pc_smoothed)
                print(f"Step 2 norm={round(norm1, 3)}")

        # Step 3 - Streamline smoothing    
        self.lines_smoothed2 = BundleCleaner.streamline_smoothing(self.pc_smoothed, self.ptct, verbose=verbose, **kwargs)
        if verbose:
            norm2 = scipy.linalg.norm(self.lines_smoothed.get_data()-self.lines_smoothed2.get_data())
            print(f"Step 3 norm={round(norm2, 3)}")
            norm3 = scipy.linalg.norm(pc-self.lines_smoothed2.get_data())
            print(f"Total norm={round(norm3, 3)}")

        # Step 4 - Subsampling + Pruning
        self.sample_idx, self.clt_labels, self.min_samples = BundleCleaner.subsampling(self.lines_smoothed2, verbose=verbose, **kwargs)
        self.lines_sampled = self.lines_smoothed2[self.sample_idx]
        self.random_idx = random_select(self.data, len(self.sample_idx))
        self.lines_random = self.data[self.random_idx]

        self.time_elapsed = time.time()-start
        print(f"Time elapsed {self.time_elapsed:.3f}.")
            
    @staticmethod
    def laplacian_smoothing(pc, smoothing_a=100, maxiter=2000, n_neighbors=30, verbose=True, **kwargs):
        if smoothing_a <= 0:
            return pc
        L, M = robust_laplacian.point_cloud_laplacian(pc, n_neighbors=n_neighbors)
        if verbose:
            print(f"- Laplacian smoothing: a={smoothing_a}, maxiter={maxiter}.")
        A = scipy.sparse.identity(len(pc))+smoothing_a*(L.T@L)
        Vsm = []
        for i in range(3):
            x, exit_code = scipy.sparse.linalg.cg(A, pc[:,i], maxiter=maxiter)
            if verbose:
                xnorm = scipy.linalg.norm(pc[:,i]-x)
                print(f"Smoothed axis {i+1} with exit code {exit_code}, norm={round(xnorm, 3)}")
            Vsm.append(x[..., np.newaxis])
        Vsm = np.concatenate(Vsm, axis=1)
        return Vsm
    
    @staticmethod
    def streamline_smoothing(pc, ptct, window_size=10, order=2, verbose=True, **kwargs):
        lines = pc_to_arraysequence(pc, ptct)
        if window_size <= 0:
            return lines
        if verbose:
            print(f"- Streamline smoothing: window size={window_size}, order={order}.")
        lines_smoothed = [savgol_filter(lines[i], window_size, order, axis=0) for i in range(len(lines))]
        return ArraySequence(lines_smoothed)
    
    @staticmethod
    def subsampling(lines, threshold=5, min_samples=None, sample_pct=0.5, verbose=True, **kwargs):
        feature = ResampleFeature(nb_points=20)
        metric = AveragePointwiseEuclideanMetric(feature=feature)  # a.k.a. MDF
        qb = QuickBundles(threshold=threshold, metric=metric)
        clusters = qb.cluster(lines)
        _, labels = qbcluster_labels(clusters)
        
        if min_samples is None: # default to median cluster size and no more than 100
            min_samples = np.percentile(np.array(list(Counter(labels).values())), 50)
            min_samples = min(min_samples, 100)
        sample_idx = cluster_subsample(labels, sample_pct=sample_pct, min_samples=min_samples)
        if verbose:
            print(f'- Subsampled {len(sample_idx)} from {len(labels)} streamlines with min_samples={min_samples}.')
        return sample_idx, labels, min_samples
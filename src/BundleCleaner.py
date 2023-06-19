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
        self.npoints_orig = len(self.data.get_data())

        self.lines_pruned = None # After step 1
        self.lines_smoothed = None # After step 2
        self.lines_smoothed2 = None # After step 3
        self.lines_sampled = None # After step 4
        self.lines_random = None # Random sampling
        self.pc_smoothed = None

        self.min_samples_prune = 0
        self.min_samples = 0
        self.time_elapsed = 0
        self.sample_idx = np.arange(len(self.data))
        self.random_idx = self.sample_idx
        
    def run(self, prune_threshold=5, prune_min_samples=None, min_lines=20, verbose=False, **kwargs):
        pc = self.data.get_data()
        if verbose:
            print(f"Cleaning bundle of size {pc.shape}")

        ## Only run step 3 if there aren't enough lines
        if len(self.data) < min_lines:
            self.lines_smoothed2 = BundleCleaner.streamline_smoothing(pc, self.ptct, verbose=verbose, **kwargs)
            if verbose:
                print(f"Not enough data ({len(self.data)} lines, {len(pc)} points), only running Step 3.")
            return
                
        start = time.time()
        # Step 1 - Pruning
        prune_idx, _, self.min_samples_prune = BundleCleaner.subsampling(self.data, threshold=prune_threshold, 
                                                min_samples=prune_min_samples, 
                                                sample_pct=1, verbose=verbose)
        self.lines_pruned = self.data[prune_idx]
        self.ptct = [len(i) for i in self.lines_pruned] # get new point count after pruning
        pc = self.lines_pruned.get_data()
        
        # Step 2 - Laplacian Smoothing
        if self.pc_smoothed is None:
            self.pc_smoothed = BundleCleaner.laplacian_smoothing(pc, verbose=verbose, **kwargs)
            self.lines_smoothed = pc_to_arraysequence(self.pc_smoothed, self.ptct)
            if verbose:
                norm1 = scipy.linalg.norm(pc-self.pc_smoothed)
                print(f"Step 2 norm={round(norm1, 3)}")

        # Step 3 - Streamline smoothing    
        self.lines_smoothed2 = BundleCleaner.streamline_smoothing(self.lines_smoothed, verbose=verbose, **kwargs)
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

        # Save final pruning without subsampling
        prune_idx2, _, _ = BundleCleaner.subsampling(self.lines_smoothed2, verbose=verbose, **{**kwargs, 'sample_pct': 1})
        self.lines_pruned2 = self.lines_smoothed2[prune_idx2]

        self.time_elapsed = time.time()-start
        print(f"Time elapsed {self.time_elapsed:.3f}.")
            
    @staticmethod
    def laplacian_smoothing(pc, alpha=100, maxiter=2000, n_neighbors=30, verbose=True, **kwargs):
        '''Point-cloud based smoothing with Laplacian regularization'''
        if alpha <= 0:
            return pc
        L, M = robust_laplacian.point_cloud_laplacian(pc, n_neighbors=n_neighbors)
        if verbose:
            print(f"Laplacian smoothing: a={alpha}, maxiter={maxiter}.")
        A = scipy.sparse.identity(len(pc))+alpha*(L.T@L)
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
    def streamline_smoothing(lines, window_size=10, poly_order=2, verbose=True, **kwargs):
        '''Streamline based smoothing with Savitsky-Golay filter'''
        if window_size <= 0:
            return lines
        if verbose:
            print(f"Streamline smoothing: window size={window_size}, order={poly_order}.")
        lines_smoothed = [savgol_filter(lines[i], window_size, poly_order, axis=0) for i in range(len(lines))]
        return ArraySequence(lines_smoothed)
    
    @staticmethod
    def subsampling(lines, threshold=5, min_samples=None, sample_pct=0.5, verbose=True, **kwargs):
        '''QuickBundles cluster based subsampling'''
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
            print(f'Subsampled {len(sample_idx)} from {len(labels)} streamlines with threshold={threshold} and min_samples={min_samples}.')
        return sample_idx, labels, min_samples
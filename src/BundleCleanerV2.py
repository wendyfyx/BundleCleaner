import time
import argparse

import scipy
import numpy as np
import robust_laplacian
from collections import Counter
from scipy.signal import savgol_filter

from nibabel.streamlines.array_sequence import ArraySequence
from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric

from utils import pc_to_arraysequence, qbcluster_labels, cluster_subsample, random_select, resample_lines_by_percent, save_bundle
from BundleCleaner import BundleCleaner

class BundleCleanerV2:

    def __init__(self, input_fpath, output_fpath, verbose=False):
        self.input_fpath = input_fpath          # example: data/bundle.trk
        self.output_fpath = output_fpath        # example: data/cleaned/bundle_proc.trk
        self.verbose = verbose

        # Load input bundle
        tractogram = load_tractogram(input_fpath, reference="same", bbox_valid_check=False)
        self.lines_orig = tractogram.streamlines
        print(f"Load {input_fpath} with {len(self.lines_orig)} lines, {len(self.lines_orig.get_data())} points.")

        self.ptct = [len(i) for i in self.lines_orig] # save point count per streamline for step 2
        self.npoints_orig = len(self.lines_orig.get_data()) # save total number of points in bundle

        self.lines_in = self.lines_orig
        self.lines_out = self.lines_orig

    
    def save_out(self, out_suffix):
        save_bundle(self.lines_out, self.input_fpath, f"{self.output_fpath[:-4]}{out_suffix}.trk", verbose=self.verbose)


    def step1(self, resample=0.5, threshold=5, min_samples=None, out_suffix=None):
        '''Step 1: Resampling and pruning'''

        if self.verbose:
            print("---Step 1---")
        self.lines_in = self.lines_out

        # Resample points for each streamline
        self.resample_rate = resample
        lines = resample_lines_by_percent(self.lines_in, percent=resample, verbose=self.verbose)

        # Initial pruning
        prune_idx, _, self.min_samples_prune = BundleCleaner.subsampling(lines, 
                                                                        threshold=threshold, 
                                                                        min_samples=min_samples, 
                                                                        sample_pct=1, 
                                                                        verbose=self.verbose)
        
        # Save output
        self.lines_out = lines[prune_idx]
        self.ptct = [len(i) for i in self.lines_out] # get new point count after pruning for step 2
        if out_suffix is not None:
            save_bundle(self.lines_out, self.input_fpath, f"{self.output_fpath[:-4]}{out_suffix}.trk", verbose=self.verbose)
        return self

    def step2(self, alpha=100, maxiter=2000, n_neighbors=30, out_suffix=None):
        '''Step 2: Point-cloud based smoothing with Laplacian regularization'''

        if self.verbose:
            print("---Step 2---")
        self.lines_in = self.lines_out

        # Point cloud based smoothing
        pc = self.lines_in.get_data()
        pc_smoothed = BundleCleaner.laplacian_smoothing(pc, alpha=alpha, maxiter=maxiter, n_neighbors=n_neighbors, 
                                                        verbose=self.verbose)
        if self.verbose:
            norm = scipy.linalg.norm(pc_smoothed-pc)
            print(f"Step 2 norm={round(norm, 3)}")

        # Save output
        self.lines_out = pc_to_arraysequence(pc_smoothed, self.ptct)
        if out_suffix is not None:
            save_bundle(self.lines_out, self.input_fpath, f"{self.output_fpath[:-4]}{out_suffix}.trk", verbose=self.verbose)
        return self


    def step3(self, window_size=10, poly_order=2, out_suffix=None, resample_back=None):
        '''Step 3: Streamline based smoothing'''

        if self.verbose:
            print("---Step 3---")
        self.lines_in = self.lines_out

        # Streamline based smoothing
        lines_out = BundleCleaner.streamline_smoothing(self.lines_in, window_size=window_size, poly_order=poly_order, 
                                                       verbose=self.verbose)
        if self.verbose:
            norm = scipy.linalg.norm(lines_out.get_data()-self.lines_in.get_data())
            print(f"Step 3 norm={round(norm, 3)}")

        # Save output
        self.lines_out = lines_out # don't resample back in case we want to 
        if out_suffix is not None:
            if resample_back is not None:
                pct = float(resample_back) / self.resample_rate #float(1)/self.resample_rate
                self.lines_out = resample_lines_by_percent(lines_out, percent=pct)
            save_bundle(self.lines_out, self.input_fpath, f"{self.output_fpath[:-4]}{out_suffix}.trk", verbose=self.verbose)
        return self
        

    def step4(self, threshold=5, min_samples=None, sample_pct=1, out_suffix="_cleaned", 
              out_suffix_pruned=None, out_suffix_random=None, resample_back=None):
        '''Step 4: Final pruning and subsampling'''

        if self.verbose:
            print("---Step 4---")
        self.lines_in = self.lines_out

        # Prune and subsample
        sample_idx, _, self.min_samples = BundleCleaner.subsampling(self.lines_in, 
                                                                    threshold=threshold, 
                                                                    min_samples=min_samples, 
                                                                    sample_pct=sample_pct, 
                                                                    verbose=self.verbose)
        self.lines_sampled = self.lines_in[sample_idx]

        # Save output
        if out_suffix is not None:
            if resample_back is not None:
                pct = float(resample_back) / self.resample_rate #float(1)/self.resample_rate
                self.lines_sampled = resample_lines_by_percent(self.lines_sampled, percent=pct)
            save_bundle(self.lines_sampled, self.input_fpath, f"{self.output_fpath[:-4]}{out_suffix}.trk", verbose=self.verbose)

        # Save random bundle subsampled from the original bundle with the same number of streamlines as lines_sampled (for comparison)
        if out_suffix_random is not None:
            random_idx = random_select(self.lines_orig, len(sample_idx))
            self.lines_random = self.lines_orig[random_idx]
            save_bundle(self.lines_random, self.input_fpath, f"{self.output_fpath[:-4]}{out_suffix_random}.trk", verbose=self.verbose)

        # Save final pruning without subsampling
        if out_suffix_pruned is not None and sample_pct < 1:
            prune_idx, _, _ = BundleCleaner.subsampling(self.lines_in, 
                                                        threshold=threshold, 
                                                        min_samples=min_samples, 
                                                        sample_pct=1, 
                                                        verbose=self.verbose)
            self.lines_pruned = self.lines_in[prune_idx]
            if resample_back:
                pct = float(resample_back) / self.resample_rate #float(1)/self.resample_rate
                self.lines_pruned = resample_lines_by_percent(self.lines_pruned, percent=pct)
            save_bundle(self.lines_pruned, self.input_fpath, f"{self.output_fpath[:-4]}{out_suffix_pruned}.trk", verbose=self.verbose)
        return self


    def run(self, args):
        '''Define BundleCleaner pipeline'''
        start = time.time()

        if len(self.lines_orig) < args.min_lines: # Only apply step 3 streamline based smoothing for small bundles
            self.step1(args.resample, min_samples=0, out_suffix=None) \
                .step3(args.window_size, args.poly_order, resample_back=args.resample_back, out_suffix="_cleaned")
        else: # Define main pipeline
            self.step1(args.resample, args.prune_threshold, args.prune_min_samples, out_suffix=None) \
                .step2(args.alpha, args.maxiter, args.n_neighbors, out_suffix=None) \
                .step3(args.window_size, args.poly_order, resample_back=None, out_suffix=None) \
                .step4(args.threshold, args.min_samples, args.sample_pct, resample_back=args.resample_back,
                    out_suffix="_cleaned", out_suffix_pruned="_pruned", out_suffix_random=None)
        
        self.time_elapsed = time.time()-start
        print(f"Total time elapsed {self.time_elapsed:.3f}.")


    def run2(self, args):
        '''Define BundleCleaner pipeline, save at every step'''
        start = time.time()

        if len(self.lines_orig) < args.min_lines: # Only apply step 3 streamline based smoothing for small bundles
            self.step1(args.resample, min_samples=0, out_suffix=None) \
                .step3(args.window_size, args.poly_order, resample_back=args.resample_back, out_suffix="step3")
        else: # Define main pipeline
            self.step1(args.resample, args.prune_threshold, args.prune_min_samples, out_suffix='_step1') \
                .step2(args.alpha, args.maxiter, args.n_neighbors, out_suffix='_step2') \
                .step3(args.window_size, args.poly_order, resample_back=None, out_suffix='_step3') \
                .step4(args.threshold, args.min_samples, args.sample_pct, resample_back=args.resample_back,
                    out_suffix="_step4_cleaned", out_suffix_pruned="_step4_pruned", out_suffix_random='_random')
        
        self.time_elapsed = time.time()-start
        print(f"Total time elapsed {self.time_elapsed:.3f}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fpath', '-i', type=str, required=True)
    parser.add_argument('--output_fpath', '-o', type=str, required=True)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--min_lines', '-M', type=int, default=50, required=False, 
                        help='Minimum number of lines for Laplacian smoothing & sampling')

    # Step 1 args
    parser.add_argument('--resample', type=float, default=0.5, required=False)
    parser.add_argument('--prune_threshold', type=float, default=5.0, required=False)
    parser.add_argument('--prune_min_samples', type=int, nargs='?', default=None, required=False)
    # Step 2 args
    parser.add_argument('--alpha', type=float, default=100.0, required=False)
    parser.add_argument('--maxiter', type=int, default=3000, required=False)
    parser.add_argument('--n_neighbors', type=int, default=50, required=False)
    # Step 3 args
    parser.add_argument('--window_size', type=int, default=5, required=False)
    parser.add_argument('--poly_order', type=int, default=2, required=False)
    # Step 4 args
    parser.add_argument('--threshold', type=float, default=5.0, required=False)
    parser.add_argument('--min_samples', type=int, nargs='?', default=None, required=False)
    parser.add_argument('--sample_pct', type=float, default=0.5, required=False)
    parser.add_argument('--resample_back', type=float, default=1, required=False)

    args = parser.parse_args()
    cleaner = BundleCleanerV2(args.input_fpath, args.output_fpath, verbose=args.verbose)
    cleaner.run(args)

if __name__ == '__main__':
    main()
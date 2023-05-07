import argparse
from BundleCleaner import BundleCleaner
from dipy.io.streamline import load_tractogram
from utils import subsample_by_line, save_bundle



def run_cleaner(args):

    tractogram = load_tractogram(args.input_fpath, reference="same", bbox_valid_check=False)
    lines = tractogram.streamlines
    print(f"Load {args.input_fpath} with {len(lines)} lines, {len(lines.get_data())} points.")

    if args.resample < 1 and args.resample > 0:
        lines = subsample_by_line(lines, percent=args.resample)
        print(f"Subsampled {100*args.resample}% of each line, with total of {len(lines.get_data())} points.")

    cleaner_args = {'prune_threshold' :  args.prune_threshold, 'prune_min_samples' : args.prune_min_samples,
                    'smoothing_a' : args.alpha, 'n_neighbors' : args.n_neighbors, 'maxiter' : args.max_iter,
                    'window_size' : args.window_size, 'order' : args.poly_order,
                    'threshold' : args.threshold, 'min_samples' : args.min_samples, 
                    'sample_pct' : args.sample_pct, 'verbose' : args.verbose}
    if 1 not in args.run_steps:
        cleaner_args['prune_min_samples'] = 0
    if 2 not in args.run_steps:
        cleaner_args['smoothing_a'] = 0
    if 3 not in args.run_steps:
        cleaner_args['window_size'] = 0
    if 4 not in args.run_steps:
        cleaner_args['min_samples'] = 0
        cleaner_args['sample_pct'] = 1
    cleaner =  BundleCleaner(lines)
    cleaner.run(**cleaner_args)
    
    if args.output_fpath is not None:
        save_bundle(cleaner.lines_sampled, args.input_fpath, args.output_fpath)
    if args.output_smooth_fpath is not None:
        save_bundle(cleaner.lines_smoothed2, args.input_fpath, args.output_smooth_fpath)
    if args.output_random_fpath is not None:
        save_bundle(cleaner.lines_random, args.input_fpath, args.output_random_fpath)

    return f"{cleaner.npoints_orig},{cleaner.time_elapsed},{len(cleaner.data)},{len(cleaner.sample_idx)},{cleaner.min_samples_prune},{cleaner.min_samples}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fpath', '-i', type=str, required=True)
    parser.add_argument('--output_fpath', '-o', type=str, nargs='?', default=None, required=False)
    parser.add_argument('--output_smooth_fpath', type=str, nargs='?', default=None, required=False)
    parser.add_argument('--output_random_fpath', type=str, nargs='?', default=None, required=False)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--run_steps', nargs='*', type=int, default=[1,2,3,4], required=False)

    # Step 1 Pruning args
    parser.add_argument('--resample', type=float, default=0.5, required=False)
    parser.add_argument('--prune_threshold', type=float, default=5.0, required=False)
    parser.add_argument('--prune_min_samples', type=int, nargs='?', default=None, required=False)
    # Step 2 Laplacian smoothing args
    parser.add_argument('--alpha', type=float, default=100.0, required=False)
    parser.add_argument('--n_neighbors', '-k', type=int, default=30, required=False)
    parser.add_argument('--max_iter', type=int, default=2500, required=False)
    # Step 3 Streamline smoothing args
    parser.add_argument('--window_size', type=int, default=5, required=False)
    parser.add_argument('--poly_order', type=int, default=2, required=False)
    # Step 4 Subsampling args
    parser.add_argument('--threshold', type=float, default=5.0, required=False)
    parser.add_argument('--min_samples', type=int, nargs='?', default=None, required=False)
    parser.add_argument('--sample_pct', type=float, default=0.5, required=False)

    args = parser.parse_args()
    run_cleaner(args)

if __name__ == '__main__':
    main()
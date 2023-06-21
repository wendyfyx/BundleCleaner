# *BundleCleaner*: Point-cloud based denoising and subsampling of tractography data

*Author*: Yixue Feng

## Running *BundleCleaner* 
- To run with default parameters, `python src/BundleCleanerV2.py -i test_bundles/AF_L.trk -o test_bundles/AF_L_proc.trk -v`.
To resample the streamlines back after cleaning, use the `-s` flag.
- Python implementation of select bundle shape metrics defined in [1] are available in `src/BundleInfo.py` for comparison.
- Sample bundle from [processed PPMI data](https://nih.figshare.com/articles/dataset/DIPY_Processed_Parkinson_s_Progression_Markers_Initiative_PPMI_Data_Derivatives/12033390) is provided at `test_bundles/AF_L.trk`.

<div style="display: flex">
<figure>
  <img
  src="eval_figs/AF_L_orig_lines.png"
  alt="Before Cleaning"
  width="300">
  <figcaption>AF_L before cleaning</figcaption>
</figure>
<figure>
  <img
  src="eval_figs/AF_L_step4_cleaned_lines.png"
  alt="After Cleaning"
  width="300">
  <figcaption>AF_L after cleaning</figcaption>
</figure>
</div>

## References
[1] F.-C. Yeh, “Shape analysis of the human association pathways,” NeuroImage, vol. 223, p. 117329, Dec. 2020, doi: [10.1016/j.neuroimage.2020.117329](https://linkinghub.elsevier.com/retrieve/pii/S1053811920308156).

[2] B. Q. Chandio et al., “Bundle analytics, a computational framework for investigating the shapes and profiles of brain pathways across populations,” Sci Rep, vol. 10, no. 1, p. 17149, Dec. 2020, doi: [10.1038/s41598-020-74054-4](http://www.nature.com/articles/s41598-020-74054-4).


# White Matter Hyperintensity Segmentation in FLAIR MRI

This project was developed for the Laboratory of Computational Intelligence and focuses on the automatic segmentation of White Matter Hyperintensities (WMH) in FLAIR MRI scans from the ADNI dataset.

The repository implements a complete computational intelligence pipeline for pixel-wise WMH classification. Several intensity, texture, morphological, and spatial features are extracted from the MRI slices, including local statistical descriptors, local z-score, white top-hat filtering, spatial priors, regional priors, and brain-center distance features.

Two complementary segmentation approaches are developed:

- A Mamdani Fuzzy Inference System (FIS), designed through interpretable linguistic rules based on the extracted feature maps.
- A Genetic Algorithm (GA), used to optimize the FIS rule weights and decision threshold according to segmentation performance.
- A Neural Network classifier, trained on the computed feature vectors to perform supervised pixel-wise WMH classification.

The project also includes preprocessing, robust feature normalization, feature significance analysis, model evaluation, and cross-validation experiments. Performance is assessed using metrics suitable for highly imbalanced medical segmentation tasks, such as Dice coefficient, sensitivity, precision, specificity, and confusion matrices.

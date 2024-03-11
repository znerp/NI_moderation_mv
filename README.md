# NI_moderation_mv
Code of the multivariate moderation model used for the analysis of cognitive reserve in https://www.biorxiv.org/content/10.1101/2023.10.10.561675v1. See https://reserveandresilience.com/framework/ for context of the framework of a moderation analysis in cognitive reserve.

The 4 main scripts require a specification of the paths to a couple of ingredients:
- a table containing the subjects identifiers as well as other variables (dependent/independent, covariates) to be used in the multivariate moderation model
- a binary mask that restricts the analyses to relevant voxels (e.g. a mask of grey matter or task-active regions)
- the contrast images (here assumed to be in the same folder and to contain the subject id in their filename)

1. (optional) outliers_beta.m:   
   Identification of voxels in the contrast images that are extreme outliers based on the distribution of the beta values of all participants in that voxel. For each participant the percentage of voxels with extreme outliers is saved in the table and can be used for exclusion of these participants in the subsequent stages. 
2. moderation_mv_cv_metaloop.m:   
   Selection of the optimal number of principal components for the dimensionality reduction of the contrast images.
3. moderation_mv.m:   
   Determination of the moderation coefficients for each voxel. Saves them in a map.
4. moderation_mv_bt.m:   
   Bootstrapping to identify voxels with significant moderation coefficients.

NOTE: in the scripts for steps 2-4, besides the paths, the covariates to be used in the model have to be adjusted in the appropriate section (marked by %% covariates)

More details can also be found in the scripts and in the preprint.

# How to (optionally) preprocess fMRI data to obtain atlas-defined ROIs to use our new inference framework 
Our framework is applicable to brain measurements in shared anatomical space (e.g., fMRI volume pixel (voxel) or region of interest (ROI) data, electrophysiology sensor, etc.). Therefore, this preprocessing is optional. 

## Step 1. Obtain T1 weighted image in the same template space as the brain measurements
1a. The template spaces used in our paper are MNI152NLin6Asym (mni152_nlin_6generation_asym_1mm_t1_brain.nii.gz) for the Human Connectome Project (HCP) data and ICBM2009cNlinAsym (mni152_icbm_2009c_nlin_asym_1mm_t1_brain.nii.gz) for the Courtois NeuroMod data. These files are available in the stim-effect/data directory. For other datasets one source of template files is FMRIB Software Library (FSL, https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases).

## Step 2. Downsample template T1 weighted image to the brain measurements voxel size
2a. To downsample the template T1 weighted image as we did in our paper use A0_downsampled_MNI_template_to_1.6mm.sh for the HCP data, and A1_downsampled_MNI_template_to_2mm.sh for the Courtois NeuroMod data. For other datasets either A* bash script can be used after specifying the proper T1_file (output of step 1, template T1 weighted image) and the voxel size (simply alter the parameter for the applyisoxfm flag if the voxels are isotropic).

## Step 3. Obtain atlas-defined ROI file
3a. The atlas-defined ROI file used in our paper, is the Shen atlas 268 ROI file (shen_1mm_268_parcellation.nii.gz). This file is available in the stim-effect/data directory. For other datasets any atlas nifti file will suffice. 

## Step 4. Resample and register atlas-defined ROI file to the brain measurement template space
4a. To resample and register the atlas-defined ROI file as we did in our paper use B0_shen_1mm_268_parcellation_downsampled_to_mni_1.6mm.sh for the HCP data, and B1_shen_1mm_268_parcellation_downsampled_to_mni_2mm.sh for the Courtois NeuroMod data. For other datasets either B* bash script can be used after specifying the proper input (output of step 3, atlas-defined ROI file) and the ref_file (output of step 2, downsampled template T1 weighted image).

## Step 5. Downsample voxels to ROIs
5a. To downsample voxels to ROIs as we did in the paper use C_downsample_voxels_to_ROIs.sh. This will calculate 1 value per ROI which is the mean across all of the voxels within an ROI. For other datasets this script can be used after specifying the atlas_file (output of step 4, atlas-defined ROI file in template space), and changing the code to specify the relevant input_file (to the path for brain measurement data for 1 subject) based on the data structure of the dataset.

#!/bin/bash

atlas_file= ${atlas} # ex. for HCP "../data/shen_1mm_268_parcellation_downsampled_to_mni_1.6mm.nii.gz"
participant_data=../../data/participant_data
out_dir=${participant_data}/pre_processed_data

#######################################
# The following lines are for the default data structure in the HCP data set. 
# For all other datasets you only need to run line 20 and 21, and change the input file in line 20 
# to the file path for your datasets voxel wise data.
#######################################

for d in ${participant_data}/*/
do
    sub=${d%/}
    if [ "$sub" != "$out_dir" ]
    then
        for run in tfMRI_MOVIE1_7T_AP tfMRI_MOVIE2_7T_PA tfMRI_MOVIE3_7T_PA tfMRI_MOVIE4_7T_AP 
        do
            input_file=${d}/MNINonLinear/Results/${run}/${run}_hp2000_clean.nii.gz
            3dROIstats -mask ${atlas_file} -quiet ${input_file} > ${out_dir}/${sub}_${run}_shen268_roi_ts.txt
        done
    fi
done
  

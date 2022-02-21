input=${MNI_file} #ex. for HCP "../mni152_nlin_6generation_asym_1mm_t1_brain"
flirt -in ${input} -ref ${input} -applyxfm -applyisoxfm 1.6 -nosearch -out ${MNI_file}_downsampled_to_1.6mm

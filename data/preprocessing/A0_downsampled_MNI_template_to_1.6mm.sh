input=${T1_file} #ex. for HCP T1_file="../mni152_nlin_6generation_asym_1mm_t1_brain"
flirt -in ${input} -ref ${input} -applyxfm -applyisoxfm 1.6 -nosearch -out ${T1_file}_downsampled_to_1.6mm

input=${MNI_file} #ex. for Courtois NeuroMod MNI_file="../mni152_icbm_2009c_nlin_asym_1mm_t1_brain"
flirt -in ${input} -ref ${input} -applyxfm -applyisoxfm 2 -nosearch -out ${MNI_file}_downsampled_to_2mm


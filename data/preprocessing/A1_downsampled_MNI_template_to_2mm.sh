input=${MNI_file} #ex. for Courtois NeuroMod MNI_file="../data/mni_icbm152_t1_tal_nlin_asym_09c_brain"
flirt -in ${input} -ref ${input} -applyxfm -applyisoxfm 2 -nosearch -out ${MNI_file}_downsampled_to_2mm


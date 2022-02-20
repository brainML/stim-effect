input=${MNI_file} #ex. for HCP "../data/MNI152_T1_1mm_brain"
flirt -in ${input} -ref ${input} -applyxfm -applyisoxfm 1.6 -nosearch -out ${MNI_file}_downsampled_to_1.6mm

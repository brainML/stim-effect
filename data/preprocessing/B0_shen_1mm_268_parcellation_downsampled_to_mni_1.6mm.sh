input="../data/shen_1mm_268_parcellation"
flirt -in ${input} -ref ../data/${ref_file}.nii.gz -applyxfm -noresampblur -interp nearestneighbour -out ${input}_downsampled_to_mni_1.6mm
# ex. for HCP ref_file="MNI125_T1_1mm_brain_downsampled_to_1.6mm"
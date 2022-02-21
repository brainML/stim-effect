input="../shen_1mm_268_parcellation"
flirt -in ${input} -ref ../${ref_file}.nii.gz -applyxfm -noresampblur -interp nearestneighbour -out ${input}_downsampled_to_mni_1.6mm
# ex. for HCP ref_file="mni152_nlin_6generation_asym_1mm_t1_brain_downsampled_to_1.6mm"
input="../shen_1mm_268_parcellation"
flirt -in ${input} -ref ../${ref_file}.nii.gz -applyxfm -noresampblur -interp nearestneighbour -out ${input}_downsampled_to_mni_2mm
# ex. for Courtois NeuroMod ref_file="mni152_icbm_2009c_nlin_asym_1mm_t1_brain_downsampled_to_2mm"
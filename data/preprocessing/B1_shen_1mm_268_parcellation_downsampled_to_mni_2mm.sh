input="../data/shen_1mm_268_parcellation"
flirt -in ${input} -ref ../data/${ref_file}.nii.gz -applyxfm -noresampblur -interp nearestneighbour -out ${input}_downsampled_to_mni_2mm
# ex. for Courtois NeuroMod ref_file="mni_icbm152_t1_tal_nlin_asym_09c_brain_downsampled_to_2mm"
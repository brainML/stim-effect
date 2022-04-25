# Same Cause; Different Effects in the Brain

This repository contains code for the paper [Same Cause; Different Effects in the Brain](https://arxiv.org/pdf/2202.10376.pdf)

Bibtex:
```
@inproceedings{toneva2021same,
  title={Same Cause; Different Effects in the Brain},
  author={Toneva, Mariya and Williams, Jennifer and Bollu, Anand and Dann, Christoph and Wehbe, Leila},
  booktitle={First Conference on Causal Learning and Reasoning},
  year={2021}
}
```

In this work, we propose a new inference framework that enables researchers to ask if the properties of a complex naturalistic stimulus affects two brain zones (i.e. voxels, regions, sensor-timepoints) in a similar way. We use simulated data and two real naturalistic fMRI datasets to show that our framework enables us to make such inferences. 

## Inference Framework

We illustrate the proposed inference framework below that enables making one of 4 inferences. There are four main steps:
1. Compute the encoding model performance. We cannot infer directly from the encoding model performance whether the two brain zones are similarly affected by the stimulus, so we proceed to the second step.
2. Compute the zone generalization. Zone generalization enables us to infer whether two zones respond similarly or differently to at least some stimulus properties, but it may be unable to present the complete picture in the case that the stimulus-representation does not capture all the stimulus properties. Thus, we proceed to the third step.
3. Compute the normalized source residuals, which allow us to infer the magnitude of any stimulus-related response that is not shared between the two zones, even outside of the stimulus-representation. 
4. We use both zone generalization and zone residuals to make one of four inferences.

<p align="center">
  <img width="800" src="https://github.com/brainML/stim-effect/figures/framework.png">
</p>

### Step 1: encoding model performance

An encoding model learns to predict the activity in an individual brain zone as a function of a stimulus-representation, and we can then evaluate the predictions of the encoding model on data heldout during training. To train an encoding model, we must select a stimulus-representation and a parameterization of the encoding model. In this work, we follow previous work and parametrize the encoding model as a linear function, regularized by the ridge penalty. We select the first hidden layer of a pretrained ELMo model as our stimulus representations, which were previously shown to significantly predict regions within the Language Network [Toneva et al. 2020, Toneva and Wehbe, 2019 NeurIPS]. 

We used code from [https://github.com/mtoneva/brain_language_nlp](https://github.com/mtoneva/brain_language_nlp) to train encoding models for each brain zone and generate predictions for heldout samples from the same brain zones. Given these predictions and true heldout samples, the encoding model performance is calculated as follows:
```
python metrics/encoding_model_performance.py 
      --predictions PREDICTIONS_FNAME 
      --data DATA_FNAME 
      --output OUTPUT_FNAME
```
The input files `PREDICTIONS_FNAME` and `DATA_FNAME` are 2d numpy arrays with shape number of samples by number of brain zones. The output is the encoding model performance for each brain zone, which is a 1d numpy array with length number of brain zones.

### Step 2: zone generalization

We define zone generalization as the generalization of an encoding model trained to predict one brain zone to data corresponding to a second brain zone. Given the predictions and true data corresponding to all brain zones, we compute the zone generalizations between all pairs of brain zones as follows:
```
python metrics/zone_generalization.py 
      --predictions PREDICTIONS_FNAME 
      --data DATA_FNAME 
      --output OUTPUT_FNAME 
      --zone_indices ZONE_IND_FNAME
```
The input files `PREDICTIONS_FNAME` and `DATA_FNAME` are 2d numpy arrays with shape number of samples by number of brain zones. The input file `ZONE_IND_FNAME` is a 1d numpy array that specifies the indices of all zones for which the zone generalization should be computed. 
The output is the pair-wise zone generalizations between all pairs of brain zones specified by `--zone_indices`, which is a 2d numpy array with shape number of brain zones by number of brain zones.

### Step 3: zone residuals

The key idea behind zone residuals is to estimate how much of the activity in zone i is affected by the stimulus but not shared with a second zone j. To estimate the activity in zone i that is not shared with zone j, we compute the residuals of regressing the activity of zone j onto the activity of zone i. Correlating these residuals across participants allows us to estimate how much of this activity is consistent across participants, i.e. driven by the stimulus. The intuition is that significant similarity between the brain activity of two people viewing the same stimulus must be driven by that stimulus [Hasson et al., 2004; Hebart et al., 2018]. Thus, correlating the residuals that we obtain from the brain zone-to-brain zone prediction allows us to identify whether there is any unique activity in one zone that is stimulus-dependent.

Given the data corresponding to brain zones in all participants, the zone residuals can be estimated as follows:
```
python metrics/zone_residuals.py 
      --data_file_format FILE_FORMAT 
      --output OUTPUT_FNAME 
      --zone_indices ZONE_IND_FNAME
```
The input `FILE_FORMAT` specifies the format of the participant file names, e.g. `/data_dir/participant_{}_data.npy`. The input file `ZONE_IND_FNAME` is a 1d numpy array that specifies the indices of all zones for which the zone residuals should be computed. The output is a dictionary of the pair-wise zone residuals between all pairs of brain zones specified by `--zone_indices` for each participant. The pair-wise zone residuals are a 2d numpy array with shape number of brain zones by number of brain zones.

Note that zone residuals are best estimated for datasets that contain at least 5 participants (see Appendix of the paper). 

### Step 4: make inferences
Once we have computed the zone generalizations and the zone residuals for all participants, we can proceed to make one of the four main inferences as follows:

```
python make_inference.py 
      --zone_generalization_format GENERALIZATIONS_FNAME_FORMAT 
      --zone_residuals_format RESIDUALS_FNAME_FORMAT 
      --generalization_threshold GEN_TRESH 
      --residuals_threshold RES_THRESH 
      --output OUTPUT_FNAME
```
The input `GENERALIZATIONS_FNAME_FORMAT` and `RESIDUALS_FNAME_FORMAT` specify the formats of the file names corresponding to the zone generalizations and zone residuals for all particpants, e.g. `/data_dir/participant_{}_zone_generalizations.npy`. Arguments `--generalization_threshold` and `--residuals_threshold` specify the thresholds for going down each branch of the inference framework illustrated above. These thresholds can be set empirically based on estimated significance. Alternatively, in this work we set these thresholds based on simulated data.
The output is a dictionary with keys corresponding to each inference type (A-D) and values corresponding to 2d numpy arrays that specify the pairs of brain zone indices for which we make the corresponding inference. 

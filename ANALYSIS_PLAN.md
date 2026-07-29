# Analysis Protocol

## Scope

This study evaluates image-level thrombus classification in static B-mode
ultrasound. It is an internally tested proof-of-concept study, not an externally
validated diagnostic system or a study of disaster-affected patients.

Training and analysis follow this document.

## Primary analysis

- Train the six defined architectures with augmentation.
- Use patient-level 5-fold cross-validation repeated 10 times with seeds 42–51.
- Use identical splits for every architecture.
- Within each outer fold, use three folds for training, one for validation, and
  one untouched fold for testing.
- Use validation data only for early stopping, checkpoint selection, ensemble
  membership, and decision thresholds.
- Pool the five held-out folds within each repetition before calculating metrics.

The primary endpoint is ROC AUC. Report the mean and standard deviation across
the 10 repetitions. Report a 95% patient-cluster bootstrap interval for the mean
AUC; repetitions are not treated as independent samples.

## Ensemble strategies

### Primary

**Validation-selected Top-3 Average Ensemble**

For every repetition and outer fold, rank the six models by AUC on that fold's
validation data, select the top three, average their positive-class
probabilities, and evaluate that fixed ensemble on the corresponding test fold.

### Comparators

- **Validation-selected Top-1 Individual:** use the single model with the
  highest validation AUC in each repetition and outer fold.
- **All-6 Average Ensemble:** average all six models without model selection.
- Report the six individual models as secondary results.

The primary contrast is Top-3 Average minus Top-1 Individual. Report the paired
patient-cluster bootstrap difference in AUC and its 95% interval.

As a secondary contrast, compare Top-3 Average with ViT using the same paired
patient-cluster bootstrap. Do not use this comparison to redefine the primary
endpoint.

Compare Top-3 Average with the image-characteristic logistic-regression
baseline using a paired patient-cluster bootstrap AUC difference and 95%
interval.

## Operating points

### Primary operating point

Within each repetition and outer fold, select on validation data the highest
threshold that attains sensitivity of at least 95%. Apply it unchanged to the
corresponding test fold. Report sensitivity, specificity, PPV, NPV, and the
confusion counts. Report patient-cluster bootstrap 95% intervals for
sensitivity, specificity, PPV, and NPV.

### Supplementary analysis

Also derive the Youden threshold from validation data and apply it unchanged to
the corresponding test fold. It is a conventional balanced operating point,
not the primary screening threshold. Report the same patient-cluster bootstrap
intervals.

Never derive or tune a threshold on test predictions.

## Calibration

Assess the held-out probabilities descriptively:

- Brier score
- Calibration curve

Recalibration and additional calibration summaries are outside the scope of
this protocol.

## Additional analyses

- Patient composition: positive-only, negative-only, and mixed-label patients.
- A patient-separated logistic-regression baseline using mean intensity and
  intensity standard deviation. Use the same outer test folds as the deep
  models, save all out-of-fold probabilities, and report a patient-cluster
  bootstrap 95% AUC interval.
- Source-image width and height as descriptive dataset characteristics only;
  do not use them as predictors.
- A within-patient mean-intensity difference with a patient-bootstrap 95%
  confidence interval and the number of patients with a higher positive-class
  mean.
- Positive and negative representative images.
- Case-level Grad-CAM targeted to the thrombus-positive class; state that
  Grad-CAM is exploratory and is not an attention or causal explanation for
  transformer models.
- Parameter count, FLOPs, and per-image inference time.
- Confusion matrices for Top-3 Average and ViT at the primary
  high-sensitivity operating point.
- Selection frequency of each architecture in Top-1 and Top-3.

## Output organization

Store statistical outputs under `outputs\stats` by analysis topic:

- `performance`: discrimination metrics, AUC contrasts, and ROC figures
- `operating_points`: validation thresholds, threshold-dependent metrics, and
  confusion matrices
- `calibration`: Brier-related calibration tables and figures
- `ensemble`: fold-specific membership and selection frequencies
- `cohort`: cohort composition
- `image_characteristics`: global image-feature analyses
- `complexity`: parameter count, FLOPs, and inference latency

## Training specification

Use horizontal flipping, weak affine transformation, and brightness/contrast
jitter. Do not use vertical flipping or elastic deformation. The augmentation
strengths are fixed before training.

Use the configured optimizer, learning-rate schedule, class weighting,
dropout, weight decay, and early-stopping settings. Do not choose among runs
based on their test performance.

## Reporting requirements

- State that the cohort came from routine hospital care, not a disaster cohort.
- Describe post-disaster DVT as multifactorial rather than directly caused by
  earthquakes.
- Define the study as image-level classification after frame selection.
- Describe the frame-selection and label-assignment procedures exactly.
- State limitations covering single-center/single-device data, unavailable
  acquisition metadata, frame-selection bias, report-derived labels,
  lack of external testing, lack of POCUS data, and transformer Grad-CAM.
- Discuss domain shift and provide a concise conclusion.
- Separate individual models and ensemble strategies in the main results table.
- Use Arial, a fixed colour-blind-safe model palette, and consistent labels
  across all statistical figures.
- Plot mean repetition-level ROC curves after pooling the five held-out folds
  within each repetition. Preserve vertical empirical ROC segments during
  interpolation and do not add uncertainty bands; report patient-cluster
  bootstrap AUC intervals in the table.
- Add a Code Availability section linking the public GitHub release and Zenodo
  DOI after the final code and results are frozen.

## Quality gates

Before training:

- All unit tests and leakage tests pass.
- Top-1 and Top-3 selection is verified to occur per repetition and outer fold.
- Both operating-point thresholds are verified to use validation data only.

Before release:

- Outputs are generated from one complete analysis run.
- Tables, figures, and manuscript values agree.
- The public repository contains no identifiable paths or images.
- A GitHub release is archived in Zenodo and its DOI is added to the manuscript.

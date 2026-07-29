# DVT thrombus classifier

## Overview

Code for image-level classification of thrombus presence in static B-mode
lower-extremity venous ultrasound images. Six pretrained architectures and
validation-selected probability ensembles are evaluated with repeated
patient-level cross-validation.

The complete protocol is defined in
[`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md). This software is an internally tested
proof of concept, not a validated medical device, and does not replace
compression ultrasonography.

## Installation

Python 3.10-3.12 and [uv](https://docs.astral.sh/uv/) are required. Commands in
this document are written for Windows Command Prompt (`cmd.exe`).

```bat
git clone https://github.com/ichikawalab/dvt-thrombus-classifier.git
cd dvt-thrombus-classifier
uv sync --extra train --group dev
```

The uv configuration installs the CUDA 13.0 builds of PyTorch and torchvision
from the official PyTorch index. Commands below use `uv run`, so manual
activation of `.venv` is not required. Verify the environment:

```bat
uv run --extra train python -c "import torch; print(torch.cuda.is_available())"
```

## Local data

Study data are not distributed. Place images and tables in an ignored local
directory:

```text
data\
|-- input.csv
`-- Images\
```

`data\`, `outputs\`, and model checkpoints are excluded by
`.gitignore`. Do not commit patient-derived data, including anonymized tables or
aggregate image features. Synthetic CSV examples are provided in
[`examples`](examples).

The training CSV requires:

| Column | Description |
|---|---|
| `image_path` | Path relative to `--data-root` |
| `label` | `0` = negative, `1` = positive |
| `patient_id` | Patient grouping identifier |
| `image_id` | Anonymous image identifier; optional |

See [`examples/example_train.csv`](examples/example_train.csv) and
[`examples/example_predict.csv`](examples/example_predict.csv).
The example files document the CSV schema only; they do not refer to distributed
images and are not executable datasets.

## Configuration

The study settings are stored in
[`configs/default.yaml`](configs/default.yaml). They include the repeated
cross-validation design, optimizer settings, batch size, epoch limits, early
stopping, classification head, worker count, and numerical precision. Training
records the resolved settings and software versions in `run_metadata.json`.

The six command-line names map to these pretrained `timm` backbones:

| Command-line name | `timm` model identifier |
|---|---|
| `ResNet50` | `resnet50.a1_in1k` |
| `DenseNet121` | `densenet121.ra_in1k` |
| `InceptionV3` | `inception_v3.tv_in1k` |
| `ConvNeXtV2` | `convnextv2_base.fcmae_ft_in22k_in1k_384` |
| `ViT` | `vit_base_patch16_384.augreg_in21k_ft_in1k` |
| `SwinT` | `swin_base_patch4_window12_384.ms_in22k_ft_in1k` |

The first use of an architecture downloads its pretrained weights. An internet
connection is therefore required unless those weights are already cached.

## Verify the installation and augmentation

Run the automated checks before training:

```bat
uv run --extra train --group dev pytest -q
uv run --group dev ruff check .
uv run --group dev ruff format --check .
```

Create a visual preview of the evaluation image and five independent augmented
views for each of four class-balanced images:

```bat
uv run --extra train python scripts\preview_augmentation.py ^
  --input-csv data\input.csv ^
  --data-root data ^
  --arch ViT ^
  --output outputs\augmentation_preview.png
```

Training augmentation consists of:

- horizontal flip with probability 0.5
- rotation within 10 degrees
- translation up to 5% in each direction
- scaling from 0.9 to 1.1
- brightness jitter of 0.2
- contrast jitter of 0.2

Vertical flipping and elastic deformation are not used. Resize dimensions and
normalization values are obtained from the pretrained model configuration.
Inspect `outputs\augmentation_preview.png` before starting the full analysis.

## Training

An optional one-repetition run can confirm that one architecture completes all
five outer folds and writes the expected artifacts. This is still a real
training run and may take time:

```bat
uv run --extra train dvt-train ^
  --input-csv data\input.csv ^
  --data-root data ^
  --arch ViT ^
  --run-name ViT ^
  --output-root outputs\smoke ^
  --n-repetitions 1 ^
  --batch-size 2 ^
  --num-workers 0
```

Run all six architectures with identical settings:

```bat
for %A in (ResNet50 DenseNet121 InceptionV3 ConvNeXtV2 ViT SwinT) do uv run --extra train dvt-train --input-csv data\input.csv --data-root data --arch %A --run-name %A --output-root outputs\runs
```

In a batch file, write `%%A` instead of `%A`. Each run records untouched
outer-test probabilities, validation probabilities, metadata, and fold
checkpoints. The default design is patient-level 5-fold cross-validation
repeated 10 times. Validation data are used for early stopping, checkpoint
selection, ensemble selection, and thresholds; test data are not used for
selection.

Each architecture produces:

```text
outputs\runs\<architecture>\
|-- checkpoints\
|-- predictions.csv
|-- validation_predictions.csv
`-- run_metadata.json
```

Use a new, empty `outputs\runs` directory for the definitive analysis. Do not
copy files from `outputs\smoke` into it.

## Statistical analysis

Run the report without the optional global image-characteristic analysis:

```bat
uv run dvt-stats --runs outputs\runs --output outputs\stats
```

The report includes:

- ROC AUC and Brier score for six models, validation-selected Top-1 and Top-3,
  and the All-6 average
- patient-cluster bootstrap confidence intervals, the primary paired Top-3
  minus Top-1 AUC difference, and the secondary Top-3 minus ViT difference
- validation-derived high-sensitivity operating points
- supplementary validation-derived Youden operating points
- patient-cluster bootstrap confidence intervals for sensitivity, specificity,
  PPV, and NPV
- calibration curves, confusion matrices, and ensemble membership

Statistical figures use Arial when it is installed, a fixed colour-blind-safe
model palette, consistent model labels, and 300 dpi raster output.
ROC figures pool the five held-out folds within each repetition, interpolate
each empirical curve while preserving vertical segments, and plot the mean
true-positive rate across repetitions without uncertainty bands. AUC
uncertainty is reported separately using patient-cluster bootstrap intervals.

The report always writes:

| Output | Intended use |
|---|---|
| `performance\summary_metrics.csv` | Main AUC and Brier-score results |
| `performance\auc_difference_top3_vs_top1.csv` | Primary paired ensemble contrast |
| `performance\auc_difference_top3_vs_vit.csv` | Secondary paired comparison with the best individual model |
| `performance\roc_curves.png` | ROC figure |
| `operating_points\operating_point_summary_high_sensitivity.csv` | Main threshold analysis |
| `operating_points\operating_point_summary_youden.csv` | Supplementary Youden analysis |
| `operating_points\confusion_matrices.png` | High-sensitivity confusion matrices |
| `calibration\calibration.png` | Calibration figure |
| `cohort\cohort_composition.csv` | Patient-level class composition |
| `ensemble\ensemble_membership.csv` | Fold-specific selected models |
| `ensemble\selection_frequencies.csv` | Architecture selection frequencies |

When `--image-features` is supplied, the report also writes:

| Output | Intended use |
|---|---|
| `image_characteristics\auc_difference_top3_vs_image_features.csv` | Paired contrast with the simple image baseline |
| `image_characteristics\image_feature_baseline_predictions.csv` | Repeated out-of-fold logistic-regression probabilities |
| `image_characteristics\image_feature_baseline_auc.csv` | Image-characteristic baseline AUC and patient-cluster 95% CI |
| `image_characteristics\image_feature_univariate_auc.csv` | Descriptive single-feature AUCs |
| `image_characteristics\source_image_dimensions.csv` | Descriptive source-image width and height |
| `image_characteristics\within_patient_intensity_difference.csv` | Paired intensity difference and patient-bootstrap 95% CI |

After training, extract full-image grayscale mean intensity and intensity
standard deviation together with source-image width and height, then run the
report with `--image-features`. Image dimensions are reported descriptively
and are not used as predictors. Encoded file size is not used.

```bat
uv run --extra train python scripts\prepare_image_features.py ^
  --input-csv data\input.csv ^
  --data-root data ^
  --output outputs\image_features.csv

uv run dvt-stats ^
  --runs outputs\runs ^
  --output outputs\stats ^
  --image-features outputs\image_features.csv
```

Use one of the two `dvt-stats` commands, not both. The report requires a new,
empty output directory to prevent results from different runs being mixed.
The image-characteristic report writes out-of-fold probabilities, a
patient-cluster bootstrap AUC interval, and a patient-bootstrap confidence
interval for the within-patient mean-intensity difference. Its logistic
regression uses the same outer test folds as the deep models.

Record model complexity and batch-one latency on the reporting hardware:

```bat
uv run --extra train dvt-benchmark --output outputs\stats\complexity\model_complexity.csv
```

## Prediction and Grad-CAM

For prediction, use a local input manifest containing genuinely new images that
were not used in model training. The command below averages predictions from
the checkpoints in the specified directory:

```bat
uv run --extra train dvt-predict ^
  --checkpoints outputs\runs\ViT\checkpoints ^
  --input-csv data\predict.csv ^
  --data-root data ^
  --arch ViT ^
  --output outputs\predictions.csv
```

For Grad-CAM visualization, use one checkpoint. If the image belongs to the
study dataset, use a checkpoint from a repetition and fold in which that image
was held out:

```bat
uv run --extra train dvt-gradcam ^
  --checkpoint outputs\runs\ViT\checkpoints\rep01_fold1.ckpt ^
  --input-csv data\gradcam_fold1.csv ^
  --data-root data ^
  --arch ViT ^
  --target-class 1 ^
  --output outputs\gradcam
```

Deployment thresholds must be derived from validation data before test or
deployment use. Grad-CAM is
exploratory and is neither attention nor causal evidence.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). A version DOI
can be added after creating a GitHub release and archiving it with Zenodo.

## License

MIT, Copyright 2026 Ichikawa Lab. See [`LICENSE`](LICENSE).

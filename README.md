# DVT thrombus classifier

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21698660.svg)](https://doi.org/10.5281/zenodo.21698660)

Deep learning tools for image-level classification of thrombus presence in
static B-mode lower-extremity venous ultrasound images. The repository supports
patient-level repeated cross-validation, probability-based ensembles,
statistical reporting, prediction, Grad-CAM visualization, and model-complexity
measurement.

This is research software, not a validated medical device.

## Installation

Python 3.10-3.12, [uv](https://docs.astral.sh/uv/), and a CUDA-capable GPU are
recommended. The commands below are for Windows Command Prompt (`cmd.exe`).

```bat
git clone https://github.com/ichikawalab/dvt-thrombus-classifier.git
cd dvt-thrombus-classifier
uv sync --extra train --group dev
```

Verify the installation:

```bat
uv run --extra train python -c "import torch; print(torch.cuda.is_available())"
uv run pytest -q
uv run ruff check .
```

## Input data

Clinical data are not distributed. Store local data as follows:

```text
data\
|-- input.csv
`-- Images\
```

`data\`, `outputs\`, and model checkpoints are excluded by `.gitignore`.
Synthetic CSV schemas are available in
[`examples`](examples).

The training CSV requires:

| Column | Description |
|---|---|
| `image_path` | Image path relative to `--data-root` |
| `label` | `0` = negative, `1` = positive |
| `patient_id` | Patient grouping identifier |
| `image_id` | Anonymous image identifier; optional |

## Configuration

Default settings are defined in
[`configs/default.yaml`](configs/default.yaml). The six command-line
architecture names correspond to:

| Name | `timm` model identifier |
|---|---|
| `ResNet50` | `resnet50.a1_in1k` |
| `DenseNet121` | `densenet121.ra_in1k` |
| `InceptionV3` | `inception_v3.tv_in1k` |
| `ConvNeXtV2` | `convnextv2_base.fcmae_ft_in22k_in1k_384` |
| `ViT` | `vit_base_patch16_384.augreg_in21k_ft_in1k` |
| `SwinT` | `swin_base_patch4_window12_384.ms_in22k_ft_in1k` |

Pretrained weights are downloaded automatically on first use.

## Augmentation preview

```bat
uv run --extra train python scripts\preview_augmentation.py ^
  --input-csv data\input.csv ^
  --data-root data ^
  --arch ViT ^
  --output outputs\augmentation_preview.png
```

## Training

Optional one-repetition check:

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

Train all six architectures:

```bat
for %A in (ResNet50 DenseNet121 InceptionV3 ConvNeXtV2 ViT SwinT) do uv run --extra train dvt-train --input-csv data\input.csv --data-root data --arch %A --run-name %A --output-root outputs\runs
```

Use `%%A` instead of `%A` in a batch file. Each run writes checkpoints,
out-of-fold test probabilities, validation probabilities, and metadata.

## Statistical report

```bat
uv run dvt-stats ^
  --runs outputs\runs ^
  --output outputs\stats
```

The report includes performance summaries, paired AUC comparisons, operating
points derived from validation data, calibration, confusion matrices, ensemble
membership, and figures.

Optional image-characteristic analysis:

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

Use a new, empty output directory for each report.

Optional discrimination within anatomical subgroups, from the same held-out
predictions. The local CSV must carry a subgroup column (for example the
venous segment of each image); `--group NAME=value,value` defines each
subgroup from that column's values:

```bat
uv run python scripts\within_segment_auc.py ^
  --runs outputs\runs ^
  --input-csv data\input.csv ^
  --segment-column segment ^
  --group femoral=CFV,SFV ^
  --group popliteal=POP ^
  --group femoral_popliteal=CFV,SFV,POP ^
  --output outputs\stats\within_segment
```

Measure model complexity and batch-one inference latency:

```bat
uv run --extra train dvt-benchmark ^
  --output outputs\stats\complexity\model_complexity.csv
```

## Prediction

```bat
uv run --extra train dvt-predict ^
  --checkpoints outputs\runs\ViT\checkpoints ^
  --input-csv data\predict.csv ^
  --data-root data ^
  --arch ViT ^
  --output outputs\predictions.csv
```

Use images that were not used to train the selected checkpoints.

## Grad-CAM

```bat
uv run --extra train dvt-gradcam ^
  --checkpoint outputs\runs\ViT\checkpoints\rep01_fold1.ckpt ^
  --input-csv data\gradcam_fold1.csv ^
  --data-root data ^
  --arch ViT ^
  --target-class 1 ^
  --output outputs\gradcam
```

For an image from the training dataset, use a checkpoint from a fold in which
that image was held out. Grad-CAM is an exploratory visualization and does not
provide causal evidence.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).

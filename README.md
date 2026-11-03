# mri-series-curation

Label routine MRI series automatically, for any multi-site DICOM archive: body part and sequence
type out of the box, or any labelling task you define. Each image group of a series is labelled
by an **image model** and a **metadata model**, whose probabilities are fused, then by
**header rules from MR physics**. A **confidence gate** sends uncertain series to a review list.
Every output NIfTI gets a JSON sidecar that records its label, confidence and source (rule or
model).

No trained weights and no data are included: you train on your own labels. Not included either:
the comparison methods of the paper (keyword rules, header-rule baseline, published classifiers)
and a review viewer (the review lists are plain CSV files, see below).

## How it works

```
DICOM (ZIP per study, or folder per study)
  -> index: one entry per image group        (scripts/build_index.py)
  -> image model    16 slices, ConvNeXt-Tiny, attention pooling over slices, temperature-scaled
     metadata model gradient-boosted trees on 46 header values + LSA of the description text
  -> fusion          p_image^w * p_meta^(1-w)
  -> header rules    identify (set the label) / exclude (mask classes), configurable
  -> gate            confidence >= threshold: accepted, else review_<task>.csv
  -> NIfTI + enriched sidecar                (scripts/convert.py, scripts/predict.py)
```

**Image groups.** DICOM stores ImageType and echo number per image, and one series can mix
magnitude and phase images (e.g. SWI), several echoes (PD + T2) or an embedded localizer. The index
splits such series into image groups (`curation/dicom_index.py: image_group`), each labelled
separately; the "main" group (magnitude, longest echo) carries the series label.

## Running it

**Requirements.** Python 3.10 or newer, a CUDA GPU (training, evaluation and prediction run the
image model on the GPU), and [dcm2niix](https://github.com/rordenlab/dcm2niix) on `PATH` for the
NIfTI conversion.

```bash
pip install -r requirements.txt
```

**1. Configure.** Copy the example and set at least `input` (where the DICOM data are) and
`labels` (your label CSVs, format below):

```bash
cp config.example.yaml config.yaml
```

```yaml
input:
  kind: zip              # one ZIP per study, or "folder": one sub-folder per study
  root: /path/to/dicom
labels:
  expert: labels/expert.csv
```

The example config labels body part (BRAIN / REST) and then the sequence type of brain series;
keep it as it is or define your own tasks (see below).

**2. Index the DICOM data.** Reads every header once and splits series into image groups. Writes
`work_dir/index.pkl`.

```bash
python scripts/build_index.py --config config.yaml
```

**3. Optional: evaluate on your data.** Patient-level cross-validation, optionally with a held-out
test set (one patient id per line). It suggests the fusion weight and the confidence threshold for
an error budget; copy them into the config.

```bash
python scripts/evaluate.py --config config.yaml --task body --folds 5
python scripts/evaluate.py --config config.yaml --task sequence --folds 5 --test-patients test_patients.txt
```

**4. Train** both models of every task on all labelled series (`work_dir/models/<task>/`):

```bash
python scripts/train.py --config config.yaml
```

**5. Convert to NIfTI** (optional, needed for labelled NIfTI output):

```bash
python scripts/convert.py --config config.yaml --out nifti/
```

**6. Predict** every image group. Writes `results/predictions.csv` and `results/review_<task>.csv`;
with `--series-files`, also the labelled NIfTIs with their sidecars under `results/nifti/`.

```bash
python scripts/predict.py --config config.yaml --out results/ --series-files nifti/series_files.csv
```

**7. Review and retrain.** Fill in the label column of `review_<task>.csv`, add the file as a label
source (`labels: {reviewed: results/review_sequence.csv}`), name it in the task's `train_labels`,
and run steps 4 and 6 again.

Volumes are cached as `.npy` under `work_dir/npy_cache`, so later runs are much faster. All steps
log counts only.

## Label files

One or more CSV files, named in the config:

```yaml
labels:
  expert: labels/expert.csv
  reviewed: labels/reviewed.csv
```

```
series_instance_uid,patient_id,body_part,sequence_type
1.2.840.xxxxx.1,P001,BRAIN,T1C
1.2.840.xxxxx.2,P001,SPINE,
```

* `series_instance_uid` is required; one row per series.
* `patient_id` (config `patient_column`) keeps all series of a patient on the same side of every
  split. Without it, the study is used.
* One column per task. Empty or `SKIP` means not labelled for that task.
* Several sources are merged; later sources win cell by cell.
* Each task chooses, per model, which sources it trains on (`train_labels: {image: [...], meta: [...]}`)
  and which are scored by `evaluate.py` (`eval_labels`). For example, train on expert + reviewed
  labels but evaluate on expert labels only.

**Review loop.** `predict.py` writes `review_<task>.csv` with the uncertain series and an empty
label column. Fill it in however you like (spreadsheet, your own viewer), add it as a label
source and retrain.

## Defining your own task

Everything task- or site-specific is in the config:

```yaml
tasks:
  spine_sequence:
    column: seq                         # column in your label CSVs
    classes: [T1, T2, STIR, OTHER]
    map: {T2_STIR: STIR}                # raw value -> class
    only_where: {body: SPINE}           # optional: only series the parent task labels SPINE
    loss: {type: focal, gamma: 3.0, label_smoothing: 0.1}
    group_labels:                       # how image groups inherit the series label
      localizer: drop                   # inherit | drop | <class> | {when_label, te_below, label}
      phase: OTHER
      other_echo: inherit
    rules:
      enabled: true
      disable: [stir_ti, stir_name]     # you want STIR as its own class, not OTHER
      rename: {}
    fusion_weight: 0.7
    threshold: 0.9
```

**Header rules** (`curation/rules.py`): 12 identifying and 5 excluding rules, e.g. ImageType ADC is
ADC, inversion recovery with TI > 1500 ms and TE > 60 ms is FLAIR, spin echo with TE < 60 ms is not
T2. Their target classes follow the example sequence scheme (T1 T1C T2 FLAIR DWI ADC SWI T2STAR
OTHER), in which STIR, proton density, bSSFP/CISS, phase images, synthetic MR, exponential ADC and
perfusion count as OTHER. In another class space, rename targets (`rename: {OTHER: MISC}`) or
disable rules by id. A rule whose target class is not in your task is skipped with a warning.

**Localizers** (`localizer:`): few images and a small matrix, ImageType LOCALIZER, or a keyword
in the description. They are never trained and never labelled.

## Evaluation

`scripts/evaluate.py` runs a patient-level cross-validation with pooled out-of-fold predictions.
It reports:
* image model, metadata model, fusion with and without rules, as macro-F1 and balanced accuracy
  with patient-bootstrap 95 % CIs;
* paired bootstrap differences and the best fusion weight, in the task's `metric` (`macro_f1` or
  `balanced_accuracy`; the example uses balanced accuracy for body part, macro-F1 for sequence);
* confidence thresholds for error budgets: the point estimate, and a safer one where the bootstrap
  95 % upper bound of the error stays within budget.

With `--test-patients` it also scores a held-out test set once.

## Output

`predictions.csv`: one row per image group, with label, confidence, source (`rule`, `model`,
`model+mask`), reason and an uncertain flag per task. With `--series-files`, NIfTIs are copied to
`<out>/nifti/<patient>/<date>_<study>/<LABEL>_<file>.nii.gz`. The dcm2niix sidecar is extended with
`ImagingPlane`, obliquity, `IsDerived`, `ComplexComponent`, `Derivation` and
`CurationLabels: {task: {Label, Confidence, Source, Reason, Uncertain}}`. UIDs and patient fields
are removed from the sidecar.

## Data protection

Run on pseudonymised data. The tools log counts and error types only, never UIDs, names or file
paths from the data.

## Licence

MIT, see `LICENSE`.

# 3D Cell Nuclei Pipeline Delivery

## Purpose

This delivery folder packages the full 3-step 3D nuclei workflow that was
validated on the test stack `fish7 double 3d.tif`.

Pipeline order:

1. Sparse-SIM style sparse deconvolution
2. Slice-wise local normalization
3. Cellpose-SAM 3D segmentation

The intended audience is:

- a collaborator who wants to understand and rerun the pipeline
- a downstream developer who wants to convert the workflow into an OpenClaw skill

## Canonical Test Case

- Raw test data: `test_data/fish7 double 3d.tif`
- Input layout: `ZCYX = (201, 2, 1024, 1024)`
- Selected channel for this pipeline: `channel 0`

## Recommended Settings

### Step 1: Sparse deconvolution

Algorithm implementation:

- Primary runnable implementation in this delivery: `scripts/sparse_sim_matlab.py`
- CLI wrapper: `scripts/run_sparse_sim_step.py`
- MATLAB reference package: `references/Sparse-SIM-master.zip`

Recommended parameters:

- `pixel_size_nm = 500`
- `wavelength_nm = 488`
- `effective_na = 1.0`
- `sparse_iter = 120`
- `fidelity = 150.0`
- `z_continuity = 1.0`
- `sparsity = 6.0`
- `deconv_iter = 8`
- `background_mode = "none"`
- `deblurring_method = "lucy_richardson"`
- `oversampling_method = "none"`
- `psf_integration_samples = 1024`

Recommended runtime mode on a 12 GB GPU:

- `mode = windowed_gpu`
- `window_size = 32`
- `halo = 4`
- `backend = cuda`

Reason:

- exact full-volume GPU reconstruction ran out of memory
- the validated workaround is z-windowed GPU reconstruction

### Step 2: Local normalization

Implementation:

- `scripts/local_normalization.py`

Recommended parameters:

- `radius = 30`
- `bias = 5e-4`
- `output_dtype = uint16`

Notes:

- this implementation is slice-wise in `XY`
- it uses strict sliding-window statistics with reflect padding
- it does not use `XY` block stitching

### Step 3: Cellpose-SAM

Implementation:

- `scripts/run_infer_3d.py`

Recommended parameters:

- `use_gpu = true`
- `cellprob_threshold = 0.0`
- `min_size = 50`
- `anisotropy = 1.0`
- `diameter = None`
- `rescale = 1.0`
- `do_3D = true`
- `z_axis = 0`
- `batch_size_3d = 4`
- `bsize = 256`
- `tile_overlap = 0.2`
- `augment = false`
- `stitch_threshold = 0.0`
- `flow_threshold = 0.4`

Notes:

- `tile_overlap = 0.2` was chosen as the preferred setting after comparison
- `augment = true` did not provide a clear enough benefit to justify the slower runtime

## Environment Setup

Base environment:

```powershell
conda env create -f environment.pipeline.yml
conda activate nuclei3d-pipeline
```

Then install GPU packages:

```powershell
python -m pip install cupy-cuda12x
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Important note:

- on Windows, stable `cellpose` import required `scipy=1.14.1` and `scikit-image=0.24.0` from conda

Repair command if needed:

```powershell
conda install -n nuclei3d-pipeline scipy=1.14.1 scikit-image=0.24.0 -y
```

## One-Command Full Run

Recommended full pipeline command:

```powershell
python scripts/run_step123_pipeline.py `
  --input "test_data/fish7 double 3d.tif" `
  --output_dir "outputs/step123_repro" `
  --config_json "configs/sparse_sim_fish7.json" `
  --channel_index 0 `
  --save_extracted_input `
  --mode windowed_gpu `
  --window_size 32 `
  --halo 4 `
  --backend cuda `
  --gpu_device_index 1 `
  --ln_radius 30 `
  --ln_bias 5e-4 `
  --ln_output_dtype uint16 `
  --use_gpu_step3 `
  --step3_batch_size_3d 4 `
  --step3_bsize 256 `
  --step3_tile_overlap 0.2
```

## Included Reference Outputs

Reference outputs are included for the validated test case:

- `artifacts/step1_sparse_sim.tif`
- `artifacts/step2_local_normalization.tif`
- `artifacts/step3_mask_tileoverlap02.tif`
- accompanying `json` and `log` files

These let the recipient check whether a rerun matches the expected behavior.

## Files That Matter Most

If the recipient only reads a few files, prioritize these:

- `README_HUMAN.md`
- `README_LLM.md`
- `configs/pipeline_fish7_recommended.json`
- `scripts/run_step123_pipeline.py`
- `scripts/run_infer_3d.py`
- `scripts/sparse_sim_matlab.py`
- `scripts/local_normalization.py`

## OpenClaw-Oriented Notes

Suggested OpenClaw skill decomposition:

1. Input validation
2. Channel extraction from `ZCYX` raw TIFF
3. Step 1 sparse deconvolution
4. Step 2 local normalization
5. Step 3 Cellpose-SAM inference
6. Output packaging and logging

Each step should log:

- input path
- output path
- exact parameters
- runtime
- GPU/CPU mode
- error message if failed

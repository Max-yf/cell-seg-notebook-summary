# LLM-Readable Pipeline Spec

## Task

Implement or explain a validated 3-step 3D nuclei segmentation pipeline for the
test stack `fish7 double 3d.tif`.

## Canonical Input

- File name: `fish7 double 3d.tif`
- Location inside this delivery: `test_data/fish7 double 3d.tif`
- Shape: `ZCYX = (201, 2, 1024, 1024)`
- Chosen channel for all validated runs: `channel_index = 0`

## Canonical Step Order

1. Sparse deconvolution
2. Local normalization
3. Cellpose-SAM 3D segmentation

Do not skip or reorder the steps.

## Source of Truth Files

- Step 1 implementation: `scripts/sparse_sim_matlab.py`
- Step 1 CLI wrapper: `scripts/run_sparse_sim_step.py`
- Step 2 implementation: `scripts/local_normalization.py`
- Step 3 implementation: `scripts/run_infer_3d.py`
- Full pipeline runner: `scripts/run_step123_pipeline.py`
- Step 1 parameter file: `configs/sparse_sim_fish7.json`
- Full recommended parameter file: `configs/pipeline_fish7_recommended.json`

## Step 1 Contract

### Input

- 3D single-channel stack with shape `(Z, Y, X)`
- For the canonical test case, this is extracted from raw `channel 0`

### Output

- `step1_sparse_sim.tif`
- dtype expected for downstream use: `uint16`

### Algorithm Notes

- This is a Python port aligned to MATLAB Sparse-SIM parameter semantics
- On 12 GB GPUs, full-volume exact GPU reconstruction is not feasible
- Validated execution mode is `windowed_gpu`
- Windowing is along `Z`, not `XY`

### Recommended Parameters

```json
{
  "pixel_size_nm": 500,
  "wavelength_nm": 488,
  "effective_na": 1.0,
  "sparse_iter": 120,
  "fidelity": 150.0,
  "z_continuity": 1.0,
  "sparsity": 6.0,
  "deconv_iter": 8,
  "background_mode": "none",
  "deblurring_method": "lucy_richardson",
  "oversampling_method": "none",
  "psf_integration_samples": 1024,
  "mode": "windowed_gpu",
  "window_size": 32,
  "halo": 4,
  "backend": "cuda"
}
```

## Step 2 Contract

### Input

- `step1_sparse_sim.tif`

### Output

- `step2_local_normalization.tif`

### Algorithm Notes

- Slice-wise `XY` local contrast normalization
- No `XY` tile stitching
- Sliding window is implemented with `scipy.ndimage.uniform_filter`
- Statistics window is `(1, 61, 61)` when `radius = 30`
- Reflect padding is used

### Recommended Parameters

```json
{
  "radius": 30,
  "bias": 0.0005,
  "output_dtype": "uint16"
}
```

## Step 3 Contract

### Input

- `step2_local_normalization.tif`

### Output

- `mask.tif`
- `meta.json`
- `params.json`
- `run.log`

### Algorithm Notes

- Uses finetuned Cellpose-SAM model
- 3D inference mode is enabled
- Internal Cellpose inference uses tiles
- The preferred tested setting is `tile_overlap = 0.2`
- `augment = true` was tested but not selected as the default

### Recommended Parameters

```json
{
  "cellprob_threshold": 0.0,
  "min_size": 50,
  "anisotropy": 1.0,
  "diameter": null,
  "rescale": 1.0,
  "do_3D": true,
  "z_axis": 0,
  "batch_size_3d": 4,
  "bsize": 256,
  "tile_overlap": 0.2,
  "augment": false,
  "stitch_threshold": 0.0,
  "flow_threshold": 0.4,
  "use_gpu": true
}
```

## Canonical Full Reproduction Command

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

## Reference Artifacts Included

- `artifacts/step1_sparse_sim.tif`
- `artifacts/step2_local_normalization.tif`
- `artifacts/step3_mask_tileoverlap02.tif`
- logs and params for each validated stage

## Recommended Skill Interface For OpenClaw

### Inputs

- raw TIFF path
- channel index
- output root path
- optional overrides for step1, step2, step3 parameters

### Outputs

- extracted input stack
- step1 output
- step2 output
- step3 mask
- metadata and logs

### Required Logging

- effective parameters
- runtime per stage
- CPU/GPU mode
- chosen CUDA device index when relevant
- output file paths

## Known Decisions

- Use `channel 0`
- Keep `sparsity = 6.0`
- Keep local normalization `radius = 30`
- Keep Cellpose `bsize = 256`
- Prefer `tile_overlap = 0.2`
- Do not enable `augment` by default

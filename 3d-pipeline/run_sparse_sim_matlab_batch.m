project_root = fileparts(mfilename('fullpath'));

addpath(fullfile(project_root, 'Sparse-SIM-master', 'src_win', 'Utils'));
addpath(fullfile(project_root, 'Sparse-SIM-master', 'src_win', 'SHOperation'));
addpath(fullfile(project_root, 'Sparse-SIM-master', 'src_win', 'SHIter'));
addpath(fullfile(project_root, 'Sparse-SIM-master', 'src_win', 'IterativeDeblur'));

input_file = fullfile(project_root, 'outputs', 'full_step1_input_ch0.tif');
output_dir = fullfile(project_root, 'outputs', 'matlab_full_step1_ch0');
output_name = 'full_step1_ch0';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

params = struct();
params.pixel_size_nm = 500;
params.wavelength_nm = 488;
params.effective_na = 1.0;
params.sparse_iter = 120;
params.fidelity = 150;
params.z_continuity = 1;
params.sparsity = 6;
params.deconv_iter = 8;
params.background_mode = 'none';
params.deblurring_method = 'lucy_richardson';
params.oversampling_method = 'none';
params.background_code = 6;
params.deblur_code = 1;
params.oversample_code = 3;
params.three_d = true;
params.debug = false;
params.use_gpu = true;
params.gpu_device_index = 2;

meta = struct();
meta.success = false;
meta.input_file = input_file;
meta.output_dir = output_dir;
meta.output_tif = fullfile(output_dir, [output_name, '_reconstructed.tif']);
meta.params_json = fullfile(output_dir, [output_name, '_params.json']);
meta.meta_json = fullfile(output_dir, [output_name, '_meta.json']);
meta.diary_log = fullfile(output_dir, [output_name, '_diary.log']);
meta.error_message = '';
meta.gpu = false;
meta.gpu_device_index = [];
meta.gpu_device_name = '';
meta.gpu_available_memory_before = [];
meta.gpu_available_memory_after = [];

try
    start_time = datetime('now');
    if exist(meta.diary_log, 'file')
        delete(meta.diary_log);
    end
    diary(meta.diary_log);
    diary on;
    lambda = params.wavelength_nm * 1e-9;
    Pixel = params.pixel_size_nm * 1e-9;
    if params.use_gpu
        gpu_count = gpuDeviceCount('available');
        fprintf('MATLAB available GPU count: %d\n', gpu_count);
        if gpu_count < params.gpu_device_index
            error('Requested GPU index %d, but MATLAB only sees %d GPU(s).', params.gpu_device_index, gpu_count);
        end
        selected_gpu = gpuDevice(params.gpu_device_index);
        fprintf('Using GPU %d: %s\n', selected_gpu.Index, selected_gpu.Name);
        fprintf('Available GPU memory before reconstruction: %.0f bytes\n', selected_gpu.AvailableMemory);
        gpu = cudaAvailable;
        meta.gpu = logical(gpu);
        meta.gpu_device_index = selected_gpu.Index;
        meta.gpu_device_name = selected_gpu.Name;
        meta.gpu_available_memory_before = double(selected_gpu.AvailableMemory);
    else
        gpu = false;
    end

    SIMmovie = imreadTiff(input_file, params.three_d);
    SIMmovie = single(SIMmovie);
    constant = max(SIMmovie(:));
    SIMmovie = SIMmovie ./ constant;

    if size(SIMmovie, 3) < 3
        flage3 = size(SIMmovie, 3);
        SIMmovie = padarray(SIMmovie, [0, 0, 3 - size(SIMmovie, 3)], 'replicate');
    else
        flage3 = size(SIMmovie, 3);
    end

    if params.deblur_code ~= 3
        if params.oversample_code ~= 3
            psfkernel = kernel(Pixel / 2, lambda, params.effective_na, 0, min(size(SIMmovie, 1), size(SIMmovie, 2)));
        else
            psfkernel = kernel(Pixel, lambda, params.effective_na, 0, min(size(SIMmovie, 1), size(SIMmovie, 2)));
        end
    end

    SIMmovie = SIMmovie ./ max(SIMmovie(:));
    switch params.background_code
        case 1
            backgrounds = background_estimation(SIMmovie ./ 2);
            SIMmovie = SIMmovie - backgrounds;
        case 2
            backgrounds = background_estimation(SIMmovie ./ 2.5);
            SIMmovie = SIMmovie - backgrounds;
        case 3
            medVal = mean(SIMmovie(:));
            sub_temp = SIMmovie;
            sub_temp(sub_temp > medVal) = medVal;
            backgrounds = background_estimation(sub_temp);
            SIMmovie = SIMmovie - backgrounds;
        case 4
            medVal = mean(SIMmovie(:)) ./ 2;
            sub_temp = SIMmovie;
            sub_temp(sub_temp > medVal) = medVal;
            backgrounds = background_estimation(sub_temp);
            SIMmovie = SIMmovie - backgrounds;
        case 5
            medVal = mean(SIMmovie(:)) ./ 2.5;
            sub_temp = SIMmovie;
            sub_temp(sub_temp > medVal) = medVal;
            backgrounds = background_estimation(sub_temp);
            SIMmovie = SIMmovie - backgrounds;
        otherwise
            backgrounds = [];
    end

    SIMmovie(SIMmovie < 0) = 0;
    SIMmovie = SIMmovie ./ max(SIMmovie(:));

    if ~params.debug && size(SIMmovie, 3) > 3
        SIMmovie(:, :, 3:size(SIMmovie, 3) + 2) = SIMmovie;
        SIMmovie(:, :, 2) = SIMmovie(:, :, 4);
        SIMmovie(:, :, 1) = SIMmovie(:, :, 5);
    end

    switch params.oversample_code
        case 1
            y = Spatial_Oversample(SIMmovie);
            f = single(y);
        case 2
            y = Fourier_Oversample(SIMmovie);
            f = single(y);
        otherwise
            f = single(SIMmovie);
    end

    sparse_tic = tic;
    SHVideo = SparseHessian_core(f, params.fidelity, params.z_continuity, params.sparsity, params.sparse_iter, gpu);
    sparse_elapsed = toc(sparse_tic);

    if ~params.debug && size(SHVideo, 3) > 3
        SHVideo = SHVideo(:, :, 3:end);
    end
    SHVideo = SHVideo(:, :, 1:flage3);

    if params.deblur_code ~= 3
        deblur_tic = tic;
        SHdeblur = Iterative_deblur(SHVideo, psfkernel, params.deconv_iter, params.deblur_code, gpu);
        deblur_elapsed = toc(deblur_tic);
        if gpu
            SHdeblurCPU = gather(SHdeblur);
        else
            SHdeblurCPU = SHdeblur;
        end
        SHdeblurCPU = SHdeblurCPU ./ max(SHdeblurCPU(:));
        result = SHdeblurCPU;
    else
        deblur_elapsed = 0;
        if gpu
            result = gather(SHVideo);
        else
            result = SHVideo;
        end
        result = result ./ max(result(:));
    end

    write_tif(result, output_dir, output_name, params.three_d, params.oversample_code, constant);

    meta.success = true;
    meta.gpu = logical(gpu);
    if gpu
        selected_gpu = gpuDevice;
        meta.gpu_device_index = selected_gpu.Index;
        meta.gpu_device_name = selected_gpu.Name;
        meta.gpu_available_memory_after = double(selected_gpu.AvailableMemory);
    end
    meta.input_shape_h_w_z = size(imreadTiff(input_file, params.three_d));
    meta.output_shape_h_w_z = size(result);
    meta.sparse_elapsed_seconds = sparse_elapsed;
    meta.deblur_elapsed_seconds = deblur_elapsed;
    meta.started_at = char(start_time);
    meta.finished_at = char(datetime('now'));

catch ME
    meta.success = false;
    meta.error_message = getReport(ME, 'extended', 'hyperlinks', 'off');
    meta.finished_at = char(datetime('now'));
end

if strcmp(get(0, 'Diary'), 'on')
    diary off;
end

params_fid = fopen(meta.params_json, 'w');
fprintf(params_fid, '%s', jsonencode(params, PrettyPrint=true));
fclose(params_fid);

meta_fid = fopen(meta.meta_json, 'w');
fprintf(meta_fid, '%s', jsonencode(meta, PrettyPrint=true));
fclose(meta_fid);

if ~meta.success
    error(meta.error_message);
end

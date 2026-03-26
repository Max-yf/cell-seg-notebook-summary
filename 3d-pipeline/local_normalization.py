import argparse
import os

import numpy as np
import tifffile
from scipy import ndimage
from tqdm import tqdm


def max_min_scale(img, max_value=65535, min_value=0, upper_percentile=99.8, lower_percentile=0.2):
    """
    Rescale an image to uint16 using robust percentiles to suppress outliers.
    """
    high = np.percentile(img, upper_percentile)
    low = np.percentile(img, lower_percentile)

    if high <= low:
        return np.zeros_like(img, dtype=np.uint16)

    clipped = np.clip(img, low, high)
    scaled = min_value + (clipped - low) * (max_value - min_value) / (high - low)
    return scaled.astype(np.uint16)


def local_normalize_2d(im, radius=30, bias=5e-4):
    """
    Perform slice-wise local contrast normalization on a 2D image.

    Formula:
        g*(x,y) = (g(x,y) - mean(x,y)) / (std(x,y) + bias)
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0, but got {radius}")

    im = np.asarray(im, dtype=np.float32)
    img_max = float(np.max(im))
    img_min = float(np.min(im))

    if img_max - img_min == 0:
        return np.zeros_like(im, dtype=np.float32)

    im_norm = (im - img_min) / (img_max - img_min)

    window_size = 2 * radius + 1
    im_mean = ndimage.uniform_filter(im_norm, size=window_size, mode="reflect")
    im_square_mean = ndimage.uniform_filter(np.square(im_norm), size=window_size, mode="reflect")
    im_std = np.sqrt(np.maximum(0.0, im_square_mean - np.square(im_mean)))

    return (im_norm - im_mean) / (im_std + bias)


def local_normalize_stack(stack, radius=30, bias=5e-4):
    """
    Perform slice-wise local contrast normalization on a 3D stack (Z, Y, X)
    without mixing information across Z.
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0, but got {radius}")

    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D stack, but got shape={stack.shape}")

    stack_max = np.max(stack, axis=(1, 2), keepdims=True).astype(np.float32)
    stack_min = np.min(stack, axis=(1, 2), keepdims=True).astype(np.float32)
    denom = stack_max - stack_min

    normalized_input = np.zeros_like(stack, dtype=np.float32)
    valid = denom > 0
    np.divide(stack - stack_min, denom, out=normalized_input, where=valid)

    window_size = 2 * radius + 1
    size = (1, window_size, window_size)
    im_mean = ndimage.uniform_filter(normalized_input, size=size, mode="reflect")
    im_square_mean = ndimage.uniform_filter(np.square(normalized_input), size=size, mode="reflect")
    im_std = np.sqrt(np.maximum(0.0, im_square_mean - np.square(im_mean)))

    output = (normalized_input - im_mean) / (im_std + bias)
    zero_dynamic = np.squeeze(~valid, axis=(1, 2))
    if np.any(zero_dynamic):
        output[zero_dynamic] = 0
    return output.astype(np.float32)


def process_image(img, radius=30, bias=5e-4, output_dtype="uint16"):
    """
    Process a single 2D image and return either uint16 or float32 output.
    """
    normalized = local_normalize_2d(img, radius=radius, bias=bias)

    if output_dtype == "uint16":
        return max_min_scale(normalized)
    if output_dtype == "float32":
        return normalized.astype(np.float32)

    raise ValueError(f"Unsupported output_dtype: {output_dtype}")


def process_stack(stack, radius=30, bias=5e-4, output_dtype="uint16"):
    normalized = local_normalize_stack(stack, radius=radius, bias=bias)

    if output_dtype == "float32":
        return normalized.astype(np.float32)

    if output_dtype == "uint16":
        output = np.zeros_like(normalized, dtype=np.uint16)
        for z in range(normalized.shape[0]):
            output[z] = max_min_scale(normalized[z])
        return output

    raise ValueError(f"Unsupported output_dtype: {output_dtype}")


def parse_args():
    parser = argparse.ArgumentParser(description="Slice-wise local contrast normalization for TIFF images")
    parser.add_argument("--input", type=str, required=True, help="Path to input TIFF file (single image or stack)")
    parser.add_argument("--output", type=str, required=True, help="Path to save output TIFF file")
    parser.add_argument(
        "--radius",
        type=int,
        default=30,
        help="Window radius for local normalization. The actual window size is 2 * radius + 1.",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=5e-4,
        help="Regularization term added to the local std to avoid division by zero.",
    )
    parser.add_argument(
        "--output_dtype",
        choices=["uint16", "float32"],
        default="uint16",
        help="Output TIFF dtype. uint16 is the default for downstream compatibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    if args.radius < 0:
        raise ValueError(f"--radius must be >= 0, but got {args.radius}")

    print(f"Loading {args.input}...")
    images = tifffile.imread(args.input)

    if images.ndim == 2:
        images = images[np.newaxis, ...]
        is_single = True
    elif images.ndim == 3:
        is_single = False
    else:
        raise ValueError(f"Expected a 2D image or a 3D stack, but got shape={images.shape}")

    num_images = images.shape[0]
    output_array_dtype = np.uint16 if args.output_dtype == "uint16" else np.float32
    print(f"Processing {num_images} frame(s)...")
    if images.ndim == 3:
        images_nor = process_stack(
            images,
            radius=args.radius,
            bias=args.bias,
            output_dtype=args.output_dtype,
        ).astype(output_array_dtype, copy=False)
    else:
        images_nor = np.zeros(images.shape, dtype=output_array_dtype)
        for i in tqdm(range(num_images)):
            images_nor[i] = process_image(
                images[i],
                radius=args.radius,
                bias=args.bias,
                output_dtype=args.output_dtype,
            )

    if is_single:
        images_nor = images_nor[0]

    print(f"Saving to {args.output}...")
    tifffile.imwrite(args.output, images_nor)
    print("Done.")


if __name__ == "__main__":
    main()

## Config Files

Execution is configured through YAML config files.

The solution for Project 1 is in the `config.yml` file and is the default config file.

The structure for a config file is as follows.

```yaml
options:
    testing:
        # If images should be saved during testing
        save_images: false
    # The directory to load images from
    image_dir: images
    # The directory to load kernels from
    kernel_dir: kernels
    # The directory to load test images from
    test_dir: test_input
    # The directory to save output images to
    output_dir: output
    # The log level to use
    log_level: WARN
execute:
    # A list of operations to execute
    - operation: <name>
      # Additional parameters
      ...
      # Input image
      source: <image>
      # Optional output image
      output: <image>
      # Optional test image
      test: <image>
    - pipeline:
        # A list of operations to execute
        - operation: <name>
          # Additional parameters
          ...
          # No source/output/test parameters
          # for individual operations
      # Input image
      source: <image>
      # Optional output image
      output: <image>
      # Optional test image
      test: <image>
```

## Operations

### Saving/Loading

All operations can have their input/output loaded/saved by adding

```yaml
# The input image
source: lena.png
# The output image
output: lena.png
```

### Comparison

All operations can be compared with a reference image by adding

```yaml
# The image to compare against
test: lena.png
```

### Convolution

```yaml
operation: convolve
# The kernel to use
kernel: identity.txt
# The input image
source: lena.png
```

### Reduce / Expand

```yaml
operation: reduce|expand
# No additional parameters
```

### Gaussian Pyramid / Laplacian Pyramid / Reconstruct

```yaml
operation: gaussian_pyramid|laplacian_pyramid|reconstruct
# The number of levels
levels: 5
```

### Mosaic

The most complicated of them all.

```yaml
operation: mosaic
# The left image
source: LEFT.png
# The right image
source2: RIGHT.png
```

If providing points in the config file:

```yaml
points: [[x1, y1], [x2, y2], ...]
points2: [[x1, y1], [x2, y2], ...]
```

If providing points interactively:

```yaml
interactive: true
```

Providing neither will use automatic point matching with SIFT

*A note on point matching*: Homography is delicate, so selecting good points is important.
They should be of a wide variety and if possible be a convex shape
in both images.

**Bad points will cause a bad mosaic.**

*As for the automatic point matching*: It's delicate. It can only really do panoramas that
don't have too much perspective change. If the homography it finds doesn't make sense, the
original left image will be returned and a warning will be logged.

## Pipelines

Pipelines can be defined by adding.

The `source` parameter will be the input for the first operation in the pipeline.

The `output` parameter will be the output for the last operation in the pipeline.

The `test` parameter will be the output for the last operation in the pipeline.

_These parameters should not be used on individual operations._

```yaml
pipeline:
    - operation: <name>
      # Additional parameters
      ...
      # No source/output/test parameters
      # for individual operations
source: lena.png
output: lena.png
test: lena.png
```

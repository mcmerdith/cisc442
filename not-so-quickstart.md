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

### Gaussian Pyramid

```yaml
operation: gaussian_pyramid
# The number of levels
levels: 5
```

### Laplacian Pyramid

```yaml
operation: laplacian_pyramid
# The number of levels
levels: 5
```

### Reconstruct

```yaml
operation: reconstruct
# The number of levels
levels: 5
```

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

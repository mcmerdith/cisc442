# CISC 442

## Setup

```bash
pip install opencv-python numpy numba scipy rich
```

## Running

VSCode configurations are provided

### Standard Interactive

Prompts for the image set to use, the method to use, the score function to use, and the window parameters.

```bash
python main.py
```

### Automatic

Runs the entire process for all image sets and methods.

This method produces the output described below.

```bash
python main.py --automatic
```

## Output

Deliverables are available in the deliverable.zip file.

The deliverable outputs are computed as follows:

### Parameters

I selected the following parameters:

- window_size: 9x9
- search_range: 20

I found these to be provide the best, most consistent results for the minimal computation time.

### Scoring Functions

I found NCC to be the best scoring function, although it was was the most computationally expensive.

SAD and SSD performed slightly worse, but had the advantage of being faster.


### Feature vs Region based matching

Region based matching provides a very precise result, with the disadvantage of being slower and more susceptible to noise.

#### Feature based matching

Feature based matching is much faster, but can struggle where images do not have many good features.

The region based matcher compares each section of each image, providing a full disparity map with minimal gaps to fill.

#### Feature based matching

My implementation of feature-based matching uses the harris corner detector to select keypoints to build the disparity map.

The same window based disparity computation is used, but the search keypoints are limited to the features detected.

Feature based matching provides a sparse disparity map, and relies on the gap filling method described below.

### Gap Filling

The gaps are filled by averaging the nearest non-zero neighbors if possible.

Gap filling is repeated until no gaps remain or 20 iterations have been completed (by default).

By default, the neighbordhood size is limited to 21 pixels, with a minimum of 5 neighbors.

### Validation

Before filling gaps, the disparity map is validated by comparing the left-to-right disparity map to the right-to-left disparity map.

Where the ltr and rtl disparity maps do no match, the pixel is considered invalid and is set to 0 (indicating a gap).

## Efficiency

Python is not the most efficient language for this task, but I chose it for its simplicity and ease of use.

In order to improve overall efficiency, I used the Numba JIT compiler to compile the functions that were computationally expensive.

In my usage, Numba has a minimal improvement to the initial runtime, but has a 3-4x speedup in subsequent computations.

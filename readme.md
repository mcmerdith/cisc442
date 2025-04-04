# CISC 442 Project 1

## Setup

**Requires Python 3.11+**

On Windows an environment manager like Anaconda is recommended.

```bash
# Install python version
sudo apt install python3.11 python3.11-venv
```

```bash
# Create virtual environment
python3.11 -m venv venv
# Activate virtual environment
source venv/bin/activate
```

```bash
# Install dependencies
pip install opencv-python pyyaml rich numpy
```

## Running

VSCode run configurations are provided.

It is also possible to run the program from the command line.

```
python3 main.py

usage: main.py [-h] [--test] [--config CONFIG]

Overcomplicated solution to PR1

options:
  -h, --help       show this help message and exit
  --test           Run tests and exit
  --config CONFIG  Load a config file
```

## Solution

The first 6 deliverables are obtained with by running `main.py` with no arguments. The default config file is `config.yml`.

The mosaics for question 7 and extra credit are obtained by running `main.py --config mosaic.yml`.

To run the semi-automated mosaicing, run `main.py --config interactive.yml`.

## Keybinds

- `<Space>` Proceed
- `q` Quit
- `h` Previous image (Preview only)
- `l` Next image (Preview only)

## Why.

Behold a masterpiece of scope creep. What could have been 7 functions in 2 files is now 7 modules.

The actual code is in [lib/image.py](lib/image.py). Everything else is extra architecture.

## A Note on Speed

Of all the things I did, optimizing the convolution function was something I didn't do, so it's very slow.

Fair warning, mosaicing takes a very long time.

## Special Thanks

A very special thank you to Tian for fixing my parameter ordering bug, and for reminding me that debuggers make everything easier.
# CISC 442 Project 1

## Setup

```bash
pip install -r requirements.txt
```

## Running

```
python3 main.py

usage: main.py [-h] [--test] [--config CONFIG]

Overcomplicated solution to PR1

options:
  -h, --help       show this help message and exit
  --test           Run tests and exit
  --config CONFIG  Load a config file
```

## Why.

Behold a masterpiece of scope creep. What could have been 7 functions in 2 files is now 6 modules.

The actual code is in `lib/image.py`. Everything else is extra architecture.

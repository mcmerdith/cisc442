
from lib.config import Config, read_config
from lib import util
from lib.executors import build_executor
from lib.testing import test
from argparse import ArgumentParser


def main(config: Config):
    executor = build_executor(config.execute)
    for step in executor:
        step.execute()


def entry():
    parser = ArgumentParser(description="Solution to PR1")
    parser.add_argument("--test", action="store_true",
                        help="Run tests and exit")
    parser.add_argument("--config", default="config.yml",
                        help="Load a config file")
    args = parser.parse_args()

    config = read_config(args.config)
    util.init(config)

    if args.test:
        test(config)
    else:
        main(config)


if __name__ == "__main__":
    entry()

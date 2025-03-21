from lib.testing import test
from argparse import ArgumentParser


def main():
    pass


def entry():
    parser = ArgumentParser(description="Solution to PR1")
    parser.add_argument("--test", action="store_true",
                        help="Run tests and exit")
    args = parser.parse_args()

    if args.test:
        test()
    else:
        main()


if __name__ == "__main__":
    entry()

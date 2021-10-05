import os
import numpy as np
from typing import Tuple
from scipy.stats import rankdata
from argparse import ArgumentParser


def read_input(input_path: str) -> Tuple[np.array, np.array]:
    """
    Reading file and sorting values by x
    :param input_path: path to in.txt file with input pairs
    :return:
    """

    with open(input_path) as f:
        # Read lines
        read_lines = f.readlines()
        # Map to int
        read_lines = [list(map(int, x.strip().split())) for x in read_lines]
        # Split on arrays
        x_list, y_list = list(zip(*read_lines))

    # Sort by x
    sort_idx = np.argsort(x_list)
    return np.array(x_list)[sort_idx], np.array(y_list)[sort_idx]


def main(args):
    assert os.path.isfile(args.input), 'input file should exist'

    # Step 1. Sort N pairs by X and define ranks for y
    x_list, y_list = read_input(args.input)
    N = len(x_list)

    assert N >= 9, 'the number of pairs must be more than 8'

    # by default, the middle method is used, which is indicated in the task
    ranks = rankdata(y_list)

    # Step 2. Sum first and last p ranks.
    p = int(np.round(N / 3))

    R1 = np.sum(ranks[-p:])
    R2 = np.sum(ranks[:p])

    # Step 3. Check R1 - R2 as normally distributed deviation with standard error ...
    difference = int(np.round(R1 - R2))
    std_error = int(np.round((N + 0.5) * np.sqrt(p / 6)))
    conjugation_measure = np.round(difference / (p * (N - p)), 2)

    # Saving results
    with open(args.output, "w") as f:
        f.write(f"{difference} {std_error} {conjugation_measure}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="path to input")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="path to output")

    main(parser.parse_args())

# author: Debananda Sarkar
# date: 2020-11-25

"""This script will split data for training and testing
in a specified ratio

Usage: split_dataset_train_test.py --source_data_file=<source_data_file> --train_data_file=<train_data_file> --test_data_file=<test_data_file> [--test_split_ratio=<test_split_ratio>] [--chosen_seed=<chosen_seed>]

Options:

--source_data_file=<source_data_file>           Relative path of source data file
--train_data_file=<train_data_file>             Relative path of training data file
--test_data_file=<test_data_file>               Relative path of test data file
[--test_split_ratio=<test_split_ratio>]         Ratio of test dataset size [Optional, default = 0.2]
[--chosen_seed=<chosen_seed>]                   Seed value to be used [Optional, default = 1]

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from docopt import docopt

opt = docopt(__doc__)


def main(
    source_data_file, train_data_file, test_data_file, test_split_ratio, chosen_seed
):
    try:
        if test_split_ratio is None:
            test_split_ratio = 0.2
        else:
            test_split_ratio = float(test_split_ratio)
    except ValueError as vx:
        print("Value of test_split_ratio should be float")
        print(vx)
        print(type(vx))
        exit(-1)
    except Exception as ex:
        print(ex)
        print(type(ex))
        exit(-99)

    try:
        if chosen_seed is None:
            chosen_seed = 1
        else:
            chosen_seed = int(chosen_seed)
    except ValueError as vx:
        print("Value of chosen_seed should be int")
        print(vx)
        print(type(vx))
        exit(-2)
    except Exception as ex:
        print(ex)
        print(type(ex))
        exit(-99)

    try:
        original_dataset = pd.read_csv(source_data_file)
        train_df, test_df = train_test_split(
            original_dataset, test_size=test_split_ratio, random_state=chosen_seed
        )
        train_df.to_csv(train_data_file, index=False)
        test_df.to_csv(test_data_file, index=False)
    except Exception as ex:
        print(ex)
        print(type(ex))
        exit(-99)

    print("Splitting Completed...")


if __name__ == "__main__":
    main(
        opt["--source_data_file"],
        opt["--train_data_file"],
        opt["--test_data_file"],
        opt["--test_split_ratio"],
        opt["--chosen_seed"],
    )

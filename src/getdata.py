# author: Debananda Sarkar
# date: 2020-11-20

'''This script will pull data from a source url
and store it in a specified file

Usage: getdata.py --source_url=<source_url> --target_file=<target_file>

Options:
--source_url=<source_url>       Source url to fetch data from
--target_file=<target_file>     Target file name to store data (with path)
'''

import pandas as pd
from docopt import docopt

opt = docopt(__doc__)

def main(source_url, target_file):
    try:
        hotels_dataset = pd.read_csv(source_url)
        hotels_dataset.to_csv("data/raw/hotels_dataset.csv")
        print("Download complete")
    except FileNotFoundError as fx:
        print("Error in target file path")
        print(fx)
        print(type(fx))   
    except Exception as ex:
        print("Error fetching data from source. Check the url")
        print(ex)
        print(type(ex))

if __name__=="__main__":
    main(opt["--source_url"], opt["--target_file"])

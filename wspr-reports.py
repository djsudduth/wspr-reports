import io
import requests
import statistics
import numpy as np
import pandas as pd
from bisect import bisect



def open_wspr_file():
    #open the wspr data into a dataframe and add a clean timestamp column
    df = pd.read_csv("wspr.txt", delimiter=r"\s+")
    df['TimeStamp'] = pd.to_datetime(df['Date'] + "T" + df['Time'] + ":00Z")
    df = df.sort_values('TimeStamp')
    print (df.head())


def main():

    try:
        #Read wspr data from wsprnet.org
        open_wspr_file()

    except Exception as e:
        print("Could not excute wspr-reports: ", repr(e))



#Version 0.0.1

if __name__ == '__main__':

    main()  # pylint: disable=no-value-for-parameter

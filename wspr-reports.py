import io
import requests
import statistics
import numpy as np
import pandas as pd
from bisect import bisect

WSPR_HEADER_ERR = "Your wspr file must add a tab between the words Date and Time in the DateTime column that defaults from wsprnet.org"

def open_wspr_file():
    #open the wspr data into a dataframe and add a clean timestamp column
    df = pd.read_csv("wspr.txt", sep='\t')
 
    if 'Timestamp' in df.columns:
        df['Timestamp'] = df['Timestamp'].str.strip()
        df['Timestamp'] = df['Timestamp'].str.replace(" ", "T")
        df['DateTime'] = pd.to_datetime(df['Timestamp'] + ":00Z")
        df = df.sort_values('DateTime')
        print (df.to_string())
    else:
        print ("\n" + WSPR_HEADER_ERR)


def main():
    try:
        #Read wspr data from wsprnet.org
        open_wspr_file()

    except Exception as e:
        print("Could not excute wspr-reports: ", repr(e))



#Version 0.0.1
if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter

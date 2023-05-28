import io
import requests
import statistics
import numpy as np
import pandas as pd
from bisect import bisect

WSPR_HEADER_ERR = "Your wspr file cannot have spaces in the data headers that defaults from wsprnet.org"

def open_wspr_file():
    #open the wspr data into a dataframe and add a clean timestamp column
    df = pd.read_csv("wspr.txt", sep='\t')
    return (df)

def add_wspr_dimensions(df):
    if 'Timestamp' in df.columns:
        df['Timestamp'] = df['Timestamp'].str.strip()
        df['Timestamp'] = df['Timestamp'].str.replace(" ", "T")
        df['DateTime'] = pd.to_datetime(df['Timestamp'] + ":00Z")
        df['map'] = pd.cut(df['az'], [0, 23, 68, 113, 158, 203, 248, 293, 337, 359], labels= ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'], ordered=False)
        df['drange'] = pd.cut(df['km'], [0, 800, 4000, 8000, 13000], labels=['NEAR', 'MID', 'LONG', 'VLONG'])
        df = df.sort_values('DateTime')
        print (df.to_string() + "\n\n")
    else:
        print ("\n" + WSPR_HEADER_ERR)

def get_wspr_snr_trends(df):
    slopes = []
    stdvs = []
    variances = []

    df = df.groupby('Reporter').agg({'SNR':list}).reset_index()

    for index, row in df.iterrows():
        if len(row['SNR']) > 2: #only calculate for 3 or more reports from the same reporter
            s = list(range(len(row['SNR'])))
            slope, intercept = statistics.linear_regression(s, row['SNR'])
            stdv = statistics.stdev(row['SNR'])
            variance = statistics.variance(row['SNR'])
        else:
            slope = 0
            stdv = 0
            variance = 0
        slopes.append(slope)
        stdvs.append(stdv)
        variances.append(variance)
    df['slopes'] = slopes
    df['stdvs'] = stdvs
    df['variances'] = variances
    print (df.to_string())




def main():
    try:
        #Read wspr data from wsprnet.org
        df = open_wspr_file()
        #Add columns of data
        add_wspr_dimensions(df)
        #Determine if wspr signal strength is getting stronger or weaker (nust have at least 3 reports)
        get_wspr_snr_trends(df)

    except Exception as e:
        print("Could not excute wspr-reports: ", repr(e))



#Version 0.0.1
if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter

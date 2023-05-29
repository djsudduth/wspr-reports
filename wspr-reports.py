import io
import requests
import statistics
import numpy as np
import pandas as pd
from bisect import bisect

WSPR_HEADER_ERR = "Your wspr file cannot be missing or have spaces in the data headers that defaults from wsprnet.org"

def open_wspr_file():
    #open the wspr data into a dataframe and add a clean timestamp column
    df = pd.read_csv("wspr.txt", sep='\t')
    return (df)

def add_wspr_dimensions(df):
    if 'Timestamp' in df.columns:
        df['Timestamp'] = df['Timestamp'].str.strip()
        df['Timestamp'] = df['Timestamp'].str.replace(" ", "T")
        df['DateTime'] = pd.to_datetime(df['Timestamp'] + ":00Z")
        df['Reporter'] = df['Reporter'].str.strip()
        df['map'] = pd.cut(df['az'], [0, 23, 68, 113, 158, 203, 248, 293, 337, 359], labels= ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'], ordered=False)
        df['drange'] = pd.cut(df['km'], [0, 800, 4000, 8000, 13000], labels=['NEAR', 'MID', 'LONG', 'VLONG'])
        df = df.sort_values('DateTime')
        print (df.to_string() + "\n\n")
        #print (df.loc[(df['map'] == 'NW') & (df['drange'] == 'MID')])
    else:
        print ("\n" + WSPR_HEADER_ERR)

def get_wspr_snr_trends(df):
    slopes = []
    stdvs = []
    variances = []
    maps = []
    kms = []

    df2 = df.groupby(['map', 'km', 'Reporter'])['SNR'].describe()
    print (df2.to_string() + "\n\n")

    df2 = df2.reset_index()
    #print(df2.groupby('map')['std'].mean() + "\n\n")
  

    df2 = df.groupby(['km', 'Reporter']).agg({'SNR':list}).reset_index()

    for index, row in df2.iterrows():
        if len(row['SNR']) > 1: #only calculate for 3 or more reports from the same reporter
            s = list(range(len(row['SNR'])))
            slope, intercept = statistics.linear_regression(s, row['SNR'])
            stdv =  statistics.stdev(row['SNR'])
            variance = statistics.variance(row['SNR'])
        else:
            slope = 0
            stdv = 0
            variance = 0
        #slopes.append("{:.2f}".format(slope))
        slopes.append(float(slope))
        stdvs.append("{:.2f}".format(stdv))
        variances.append("{:.2f}".format(variance))
        maps.append(df.loc[(df['Reporter'] == row['Reporter'])].iloc[0]['map'])
    df2['slopes'] = slopes
    df2['stdvs'] = stdvs
    df2['variances'] = variances
    df2['map'] = maps
    print (df2.to_string() + "\n\n")
    df2 = df2.reset_index()
    #print (df.loc[(df['Reporter'] == row['Reporter'])].iloc[0]['map'])
    
    df2 = df2.groupby('map').agg({'slopes':list}).reset_index()
    df2['Mean'] = [np.array(x).mean() for x in df2.slopes.values]
    print (df2.to_string())


 
    





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


#Saved queries
#print (df2.loc[(df2['map'] == 'SE') & (df2['km'] < 600)])

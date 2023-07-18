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

def open_goes_xray_file():
    #open the GOES satellite 6-hour xray flux data
    dfx = pd.read_json('xrays-6-hour.json')
    dfx = dfx.iloc[::2]
    dfx = dfx.rename(columns={'time_tag':'Timestamp'})
    dfx.flux = dfx.flux.apply(float).round(12)
    dfx['modflux'] = dfx['flux'] * 1e8
    return(dfx)


def join_wspr_with_goes(df, dfx):
    #join the GOES satellite 6-hour xray flux data with the wspr data and return join
    df = pd.merge(df, dfx, on='Timestamp', how='inner').reset_index()
    #print ("\n\nGOES data - first rows: ")
    #print (df.head(5))
    #input("Saving full wspr-goes-data.csv - press Enter to continue...")
    df.to_csv("wspr-goes-data.csv")
    return(df)


def add_wspr_dimensions(df):
    if 'Timestamp' in df.columns:
        df['Timestamp'] = df['Timestamp'].str.strip()
        df['Time'] = df['Timestamp'].str[-5:]
        df['Timestamp'] = df['Timestamp'].str.replace(" ", "T") +":00Z"
        df['Reporter'] = df['Reporter'].str.strip()
        df['map'] = pd.cut(df['az'], [0, 23, 68, 113, 158, 203, 248, 293, 337, 359], labels= ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'], ordered=False)
        df['drange'] = pd.cut(df['km'], [0, 800, 4000, 8000, 13000], labels=['NEAR', 'MID', 'LONG', 'VLONG'])
        df = df.sort_values('Timestamp').reset_index()
        print ("\n\nWSPR and GOES joined data - first rows: ")
        print (df.head(7) ) #+ "\n\n")
        input("Press Enter to continue...")
        #print (df.loc[(df['map'] == 'NW') & (df['drange'] == 'MID')])
        return(df)
    else:
        print ("\n" + WSPR_HEADER_ERR)


def get_wspr_snr_trends(df):
    slopes = []
    stdvs = []
    variances = []
    maps = []

    df2 = df.groupby(['map', 'km', 'Reporter'])['SNR'].describe()
    print ("\n\nWSPR mean and std dev of SNRs by map direction from your callsign location: ")
    print (df2.to_string() + "\n\n")
    df2.to_csv("wspr-map-trends.csv")
    input("Saving wspr-map-trends.csv - press Enter to continue...")

    df2 = df2.reset_index()
    df2 = df.groupby(['km', 'Reporter']).agg({'SNR':list}).reset_index()

    for index, row in df2.iterrows():
        if len(row['SNR']) > 2: #only calculate for 3 or more reports from the same reporter
            s = list(range(len(row['SNR'])))
            slope, intercept = statistics.linear_regression(s, row['SNR'])
            stdv =  statistics.stdev(row['SNR'])
            variance = statistics.variance(row['SNR'])
        else:
            slope = 0
            stdv = 0
            variance = 0
        #slopes.append("{:.2f}".format(slope))
        slopes.append(round(float(slope), 2))
        stdvs.append("{:.2f}".format(stdv))
        variances.append(round(float(variance), 1))
        maps.append(df.loc[(df['Reporter'] == row['Reporter'])].iloc[0]['map'])
    df2['slope'] = slopes
    df2['stdv'] = stdvs
    df2['variance'] = variances
    df2['map'] = maps

    print ("\n\nTrending slopes and std dev of SNRs by distance of Reporting callsign: ")
    print (df2.to_string())
    input("Saving wspr-trends-distance.csv - press Enter to continue...")
    df2.to_csv("wspr-trends-distance.csv")
    df2 = df2.reset_index()

    df3 = df.groupby(['km', 'Reporter']).agg({'flux':list}).reset_index()
    #print ("\n\nGOES flux by Reporting callsign: ")
    #print (df3.to_string() + "\n\n")
    #input("Press Enter to continue...")

    print ("\n\nSNR trending slopes by map direction from your callsign location: ")
    df4 = df2.groupby('map').agg({'slope':list}).reset_index()
    df4['snr trend'] = [np.array(x).mean() for x in df4.slope.values]
    #df2['var mean'] = [np.array(x).mean() for x in df2.variance.values]
    print (df4.to_string() + "\n\n")
    df4.to_csv("wspr-trends-by-azimuth.csv")
    input("Saving wspr-trends-by-azimuth.csv - press Enter to continue...")

    print ("\n\nSNR variances by map direction from your callsign location: ")
    df5 = df2.groupby('map').agg({'variance':list}).reset_index()
    df5['var trend'] = [np.array(x).mean() for x in df5.variance.values]
    print (df5.to_string() + "\n\n")

 

def main():
    try:
        #Read wspr data from wsprnet.org
        df = open_wspr_file()
        #Add columns of data and normalize timestamps
        df = add_wspr_dimensions(df)
        #Read the GOES xray data
        dfx = open_goes_xray_file()
        #Inner join the wspr and xray data and return join
        df = join_wspr_with_goes(df, dfx)
        #Determine if wspr signal strength is getting stronger or weaker (nust have at least 3 reports)
        get_wspr_snr_trends(df)

    except Exception as e:
        print("Could not excute wspr-reports: ", repr(e))



#Version 0.0.1
if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter


#Saved queries
#print (df2.loc[(df2['map'] == 'SE') & (df2['km'] < 600)])

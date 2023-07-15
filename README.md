# wspr-reports
Reports and charts for amateur radio wspr hf performance using propagation accuracy measurements. The goal is to determine HF antenna performance based on a time series collection of wspr signal reports. 

## wspr data
Copy-paste data directly from wsprnet.org database query results page and save the data as 'wspr.txt' in the script folder. See example wspr.txt data in the repo. 

Be sure to save the headers with the data (api forthcoming)  

## xray data
Save GOES `xrays-6-hour.json` directly to same folder from https://services.swpc.noaa.gov/json/goes/primary/  
See example GOES data in the repo. 

## usage
Requirements - numpy and pandas  (`pip install`)

Execute:  
`python wspr-reports.py`    
The wspr and xray data will be joined on timestamp. Raw data outputs:
1. wspr data with additional fields of direction from your location and distance classification
2. Joined wspr and GOES data on timestamp
3. View of reporter callsigns by map direction (using azimuth) from your location with SNR mean and standard deviation from at least 2 reports
4. View of reporter callsigns by distance, SNR list, slope trend of the SNR, std deviation, and variance
5. View of reporter callsigns by distance, GOES flux
6. Mean of the trending slopes of SNR reports by map direction from your location (+slopes - trending stronger, -slopes - weaker)
7. Mean of variances of the SNR reports by map direction from your location (large variances - widely varying reports)

Data is saved as .csv files in the script directory.

## example on how to use
 - KN0VA transmitted wspr on 30m for about 1 hour 
 - The GOES and wspr data is joined together by timestamp 
 - For report #3 above - the wspr data has the lat/lon of KNOVA to determine map direction
    - the report shows that wspr reporter KX4AZ/T which is 617km East of KNOVA has 21 reception reports with a wspr mean of -10.38 db and standard deviation of 3.73 db with a minimum report of -20.0 db / maximum of -6.0 db
 - For report #4 above 
    - the report shows the individual 21 receptions by KX4AZ/T along with the trending slope (increasing vs decreasing signal reports)
 - For report #5
    - shows the individual GOES reports for the same 21 time periods of KX4AZ/T receptions
 - For report #6
    - shows all the average slopes in one direction (KX4AZ/T is included in the list) and the trend of those slopes (are signals Eastward increasing or decreasing in report strength?)
 - For report #7
    - shows all the average variances of signal reports in one direction (KX4AZ/T is included in the list) and the trend of those variaces (are signals Eastward varying widely in report strength?)






## notes
Note that wspr SNRs vary by a standard deviation of typically around 2.0+ even when xray flux is low and steady. This is independent of distance. Example:  
KD2OM at 1270 km had snr reported values of [-11, -16, -13, -13] over a 45 min span midday on 30m -> std dev = 2.06,  variance = 4.2   
or   
LX1DQ at 6885 km had snr values of [-16, -19, -19, -18, -16, -14, -16] over 1 hour span midday on 30m ->std dev = 1.86, variance = 3.5  

Data measurements over 1-2 hours should consider D-layer ionization variability based on time of day, frequency and GOES spikes. Also, using SNRs for antenna performance characteristics should consider the std dev variability of SNR reports along with sporadic receptions from reporters that may appear or fade based on atmospheric changes. 

Changing antenna configurations and using this data to understand performance should consider multiple wspr calls throughout the day over a number of days. Shorter measurements can lead to misinterpreted results.

## example range of data
![GOES data range for example](goes-data-range.png)


## release
v0.0.1 - 05/29/2023

Thank you!  KN0VA
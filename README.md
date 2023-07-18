# wspr-reports
Reports and charts for amateur radio wspr hf performance using propagation accuracy measurements. The goal is to determine HF antenna performance based on a time series collection of wspr signal reports. 

Background: Many amateurs utilize wspr reports to determine their overall HF system performance. However, widely variable factors affect wspr reports by distant stations. These can include time of day, solar x-ray flux, ionospheric variations, etc. 
**Relying on short duration wspr reports can lead to false assumptions and correlations regarding performance**.

Examples: 
The first report shows the wspr mean and std deviations by azimuth direction (N, E, S, W...) from my location (determined by the wsprnet.org data) based on the number of reception reports by monitoring callsigns:
 
`map  km    Reporter        count    mean        std   min    25%   50%    75%   max`
`S   731   AC0G            1.0 -29.000000        NaN -29.0 -29.00 -29.0 -29.00 -29.0`
`    734   KV0S            3.0 -13.000000   1.732051 -14.0 -14.00 -14.0 -12.50 -11.0`
    918   K6RFT           7.0   0.285714   3.728909  -6.0  -1.50   2.0   3.00   3.0
    1164  W5JVS           6.0  -6.833333   3.600926 -12.0  -8.00  -7.0  -6.00  -1.0
    1475  KG5ABO          1.0 -21.000000        NaN -21.0 -21.00 -21.0 -21.00 -21.0
    1720  N5TTT           2.0 -18.000000   2.828427 -20.0 -19.00 -18.0 -17.00 -16.0
    1733  N5BIA           1.0 -17.000000        NaN -17.0 -17.00 -17.0 -17.00 -17.0
    2185  WB5B            7.0 -15.714286   2.429972 -19.0 -17.50 -16.0 -13.50 -13.0```

In this example, all of the reports received south of my location 


# usage
Requirements - numpy and pandas  (`pip install`)

## wspr data
Copy-paste data directly from https://www.wsprnet.org/drupal/wsprnet/spotquery database query results page and save the data as `wspr.txt` in the script folder. See example wspr.txt data in the repo. 

Be sure to save the headers with the data (api forthcoming)  

## xray data
Save GOES `xrays-6-hour.json` directly to same folder from https://services.swpc.noaa.gov/json/goes/primary/  
See example GOES data in the repo. 


Map direction is determined by your callsign location in the wspr.org results with this mapping deg to direction:  
[0, 23, 68, 113, 158, 203, 248, 293, 337, 359] map to labels ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']

Execute:  
`python wspr-reports.py`    
The wspr and xray data will be joined on timestamp. Raw data outputs:
1. wspr data with additional fields of direction from your location and distance classification (file only)
2. Joined wspr and GOES data on timestamp (file only)
3. View of reporter callsigns by map direction (using azimuth) from your location with SNR mean and standard deviation from at least 2 reports
4. View of reporter callsigns by distance, SNR list, slope trend of the SNR, std deviation, and variance
5. Mean of the trending slopes of SNR reports by map direction from your location (+slopes - trending stronger, -slopes - weaker)
6. Mean of variances of the SNR reports by map direction from your location (large variances - widely varying reports)

Data is saved as .csv files in the script directory.

## example on how to use
 - KN0VA transmitted wspr on 30m for about 1 hour 
 - The GOES and wspr data is joined together by timestamp in a file
 - For report #3 above - the wspr data has the lat/lon of KN0VA to determine map direction
    - the report shows that wspr reporter KX4AZ/T which is 617km East of KN0VA has 21 reception reports with a wspr mean of -10.38 db and standard deviation of 3.73 db with a minimum report of -20.0 db / maximum of -6.0 db
 - For report #4 above 
    - the report shows the individual 21 receptions by KX4AZ/T along with the trending slope of -0.23 (increasing vs decreasing signal reports)
 - For report #5
    - shows the individual GOES reports for the same 21 time periods of KX4AZ/T receptions
 - For report #6
    - shows all the average slopes in one direction (KX4AZ/T is included in the East list) and the trend of those slopes (are signals Eastward increasing or decreasing in report strength?)
 - For report #7
    - shows all the average variances of signal reports in one direction (KX4AZ/T is included in the East list) and the trend of those variaces (shows if signals Eastward varying widely in report strength)


## notes
Note that wspr SNRs vary by a standard deviation of typically around 2.0+ db even when xray flux is low and steady. This is independent of distance. Example:  
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
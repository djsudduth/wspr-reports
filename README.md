# wspr-reports
Reports and charts for amateur radio wspr hf propagation accuracy measurements

## wspr data
Copy-paste data directly from wsprnet.org database query results page and save the data as 'wspr.txt' in the script folder.  
Be sure to save the headers with the data (api forthcoming)  

## xray data
Save GOES xrays-6-hour.json directly to same folder from https://services.swpc.noaa.gov/json/goes/primary/  

## usage
`python wspr-reports.py`  
The wspr and xray data will be joined on timestamp. Raw data outputs:
1. wspr data
2. Joined wspr and goes data on timestamp
3. View of reporter callsigns by direction from your location with SNR mean and standard deviation from at least 2 reports
4. View of reporter callsigns by distance, SNR list, slope trend of the SNR, std deviation, and variance
5. Mean of the trending slopes of SNR reports by map direction from your location (+slopes - trending stronger, -slopes - weaker)
6. View of reporter callsigns by distance, goes flux

## notes
Note that wspr SNRs vary by a standard deviation of at least 2.0 even when xray flux is low and steady. This is independent of distance. Example: 
KD2OM at 1270 km had snr reported values of [-11, -16, -13, -13] over a 45 min span midday on 30m -> std dev = 2.06  variance = 4.2 

Data measurements over 1-2 hours should consider D-layer ionization variability based on time of day and GOES spikes. Also, using SNRs for antenna performance characteristics should consider the std dev variability with reports



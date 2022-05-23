# Case study from The Quick Python Book by Naomi Ceder
# import requests, namedtuple from collections library, NumPy, and matplotlib

import requests
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
# get readme.txt file

r = requests.get('https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt')
readme = r.text

# get inventory and stations files

r = requests.get('https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt')
inventory_txt = r.text
r = requests.get('https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt')
stations_txt = r.text

# save both the inventory and stations files to disk, in case we need them

with open("inventory.txt", "w") as inventory_file:
    inventory_file.write(inventory_txt)

with open("stations.txt", "w") as stations_file:
    stations_file.write(stations_txt)

# parse to named tuples
# use namedtuple to create a custom Inventory class

Inventory = namedtuple("Inventory", ['station', 'latitude', 'longitude', 
                                     'element', 'start', 'end'])

# parse inventory lines and convert some values to floats and ints
# creates list of Inventory objects from each line representing a US
# station (using a list comprehension)

inventory = [Inventory(x[0:11], float(x[12:20]), float(x[21:30]), x[31:35],
                                                      int(x[36:40]), int(x[41:45]))
             for x in inventory_txt.split("\n") if x.startswith("US")]

# create sublist of station Inventory items in which the element is TMIN or
# TMAX and with at least 95 years' worth of data (with a list comprehension)

inventory_temps = [x for x in inventory if x.element in ['TMIN', 'TMAX']
                   and x.end >= 2015 and x.start < 1920]

# custom sort stations based on difference from my location
# Worthington, OH
latitude, longitude = 40.0931, -83.0180

inventory_temps.sort(key = lambda x: abs(latitude - x.latitude) + 
                     abs(longitude - x.longitude))

# parse stations to match data with inventory records

station_id = inventory_temps[0].station # my station ID
Station = namedtuple("Station", ['station_id', 'latitude', 'longitude', 
                                 'elevation', 'state', 'name', 'start', 'end'])

# creates 1-element list of tuple with attributes of my station
stations = [(x[0:11], float(x[12:20]), float(x[21:30]), float(x[31:37]),
             x[38:40].strip(), x[41:71].strip())
            for x in stations_txt.split("\n") if x.startswith(station_id)]

station = Station(*stations[0], inventory_temps[0].start,
                                  inventory_temps[0].end)
print(station)

# fetch daily records for my station

r = requests.get('https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/all/USC00334979.dly')
weather_txt = r.text

# save into a text file, so we won't need to fetch again

with open('weather_Worthington.txt', "w") as weather_file:
    weather_file.write(weather_txt)

# read from saved daily file if needed to avoid re-downloading

with open('weather_Worthington.txt') as weather_file:
    weather_txt = weather_file.read()

def parse_line(line):
    """ parses line of weather data
        removes values of -9999 (missing value)
    """
    
    # return None if line is empty
    if not line:
        return None
    
    # split out first 4 fields and string containing temperature values
    # converts year and month to ints
    record, temperature_string = (line[:11], int(line[11:15]), 
     int(line[15:17]), line[17:21]), line[21:]
    
    # return None if the temperature string is too short
    if len(temperature_string) < 248:
        return None
    
    # use a list comprehension on the temperature_string to extract and convert values
    # 1st value occurs at index 21 in line, is 5 char long, and has 3 flag char following
    values = [float(temperature_string[i:i + 5])/10 for i in range(0, 248, 8)
              if not temperature_string[i:i + 5].startswith("-9999")]
    
    # get the number of values, the max and min, and calculate average
    count = len(values)
    tmax = round(max(values), 1)
    tmin = round(min(values), 1)
    mean = round(sum(values)/count, 1)
    
    # add the temperature summary values to the record fields extracted
    # earlier and return
    return record + (tmax, tmin, mean, count)

# process all weather data

# list comprehension, will not parse empty lines
weather_data = [parse_line(x) for x in weather_txt.split("\n") if x]

# selecting temperature data
tmax_data = [x for x in weather_data if x[3] == 'TMAX']
tmin_data = [x for x in weather_data if x[3] == 'TMIN']

# creating lists of yearly average highs and lows
# separate arrays for the 1980's just for fun
years = list(range(1917, 2023))
yearly_highs = []
yearly_lows = []
eightiesh = []
eightiesl = []

for i in years:
    sum1 = 0
    for x in tmax_data:
        if x[1] == i:
            sum1 += x[-2]
    yearly_highs.append(round(sum1/12, 1))
    if(i//10 == 198):
        eightiesh.append(round(sum1/12, 1))
    sum2 = 0
    for x in tmin_data:
        if x[1] == i:
            sum2 += x[-2]
    yearly_lows.append(round(sum2/12, 1))
    if(i//10 == 198):
        eightiesl.append(round(sum2/12, 1))

# convert to Farenheit
highs = []
lows = []
for x in yearly_highs:
    highs.append(round((x * (9/5)) + 32, 1))
for y in yearly_lows:
    lows.append(round((y * (9/5)) + 32, 1))
for i, x in enumerate(eightiesh):
    eightiesh[i] = round((x * (9/5)) + 32, 1)
for i, y in enumerate(eightiesl):
    eightiesl[i] = round((y * (9/5)) + 32, 1)

# make lists numpy arrays to use for graphing
years = np.array(years)
highs = np.array(highs)
lows = np.array(lows)
decade = np.arange(1980, 1990)

fig, ax = plt.subplots(figsize=(5, 2.7))
ax.plot(years, highs, label='high')
ax.plot(years, lows, label='low')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (F)')
ax.set_title("Average Historical Temperatures in Worthington")
ax.legend()

fig, ax = plt.subplots()
ax.plot(decade, eightiesl, label='avg yearly low')
ax.plot(decade, eightiesh, label='avg yearly high')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (F)')
ax.set_title("Average 80's Temperatures in Worthington")
ax.legend()

import requests
import pandas as pd
import numpy as np
import datetime
from pandas import json_normalize

# Initialize lists
BoosterVersion, PayloadMass, Orbit, LaunchSite, Outcome = [], [], [], [], []
Flights, GridFins, Reused, Legs, LandingPad = [], [], [], [], []
Block, ReusedCount, Serial, Longitude, Latitude = [], [], [], [], []

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Define functions
def getBoosterVersion(data):
    for x in data['rocket']:
       if x:
        response = requests.get("https://api.spacexdata.com/v4/rockets/"+str(x)).json()
        BoosterVersion.append(response['name'])

def getLaunchSite(data):
    for x in data['launchpad']:
       if x:
         response = requests.get("https://api.spacexdata.com/v4/launchpads/"+str(x)).json()
         Longitude.append(response['longitude'])
         Latitude.append(response['latitude'])
         LaunchSite.append(response['name'])

def getPayloadData(data):
    for load in data['payloads']:
       if load:
        response = requests.get("https://api.spacexdata.com/v4/payloads/"+load).json()
        PayloadMass.append(response['mass_kg'])
        Orbit.append(response['orbit'])

def getCoreData(data):
    for core in data['cores']:
            if core['core'] != None:
                response = requests.get("https://api.spacexdata.com/v4/cores/"+core['core']).json()
                Block.append(response['block'])
                ReusedCount.append(response['reuse_count'])
                Serial.append(response['serial'])
            else:
                Block.append(None)
                ReusedCount.append(None)
                Serial.append(None)
            Outcome.append(str(core['landing_success'])+' '+str(core['landing_type']))
            Flights.append(core['flight'])
            GridFins.append(core['gridfins'])
            Reused.append(core['reused'])
            Legs.append(core['legs'])
            LandingPad.append(core['landpad'])

# Load data
spacex_url = "https://api.spacexdata.com/v4/launches/past"
response = requests.get(spacex_url)

data = response.json()
data = json_normalize(data)
    
data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]

data = data[data['cores'].map(len)==1]
data = data[data['payloads'].map(len)==1]

data['cores'] = data['cores'].map(lambda x : x[0])
data['payloads'] = data['payloads'].map(lambda x : x[0])

data['date'] = pd.to_datetime(data['date_utc']).dt.date

data = data[data['date'] <= datetime.date(2020, 11, 13)]

    # Populate data
getBoosterVersion(data)
getLaunchSite(data)
getPayloadData(data)
getCoreData(data)
    
    # Create DataFrame
launch_dict = {'FlightNumber': list(data['flight_number']),
    'Date': list(data['date']),
    'BoosterVersion':BoosterVersion,
    'PayloadMass':PayloadMass,
    'Orbit':Orbit,
    'LaunchSite':LaunchSite,
    'Outcome':Outcome,
    'Flights':Flights,
    'GridFins':GridFins,
    'Reused':Reused,
    'Legs':Legs,
    'LandingPad':LandingPad,
    'Block':Block,
    'ReusedCount':ReusedCount,
    'Serial':Serial,
    'Longitude': Longitude,
    'Latitude': Latitude}
    
launchdf = pd.DataFrame(launch_dict)
print(launchdf.head())
    
    # Filter for Falcon 9 launches
data_falcon9 = launchdf[launchdf['BoosterVersion'] != 'Falcon 1']
data_falcon9.loc[:, 'FlightNumber'] = range(1, len(data_falcon9) + 1)

print(data_falcon9['BoosterVersion'].head())

    # Calculate and replace missing PayloadMass values
mean_payload_mass = data_falcon9['PayloadMass'].mean()
data_falcon9['PayloadMass'] = data_falcon9['PayloadMass'].replace(np.nan, mean_payload_mass)

    # Export the DataFrame to a CSV file
data_falcon9.to_csv('dataset_part_1.csv', index=False)


print(falcon9_launches_count = data_falcon9.shape[0])

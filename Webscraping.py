import requests
from bs4 import BeautifulSoup
import unicodedata
import pandas as pd

# Define helper functions
def date_time(table_cells):
    """
    This function returns the data and time from the HTML  table cell
    Input: the  element of a table data cell extracts extra row
    """
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
    """
    This function returns the booster version from the HTML  table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=''.join([booster_version for i,booster_version in enumerate( table_cells.strings) if i%2==0][0:-1])
    return out

def landing_status(table_cells):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=[i for i in table_cells.strings][0]
    return out


def get_mass(table_cells):
    mass=unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass=mass[0:mass.find("kg")+2]
    else:
        new_mass=0
    return new_mass


def extract_column_from_header(row):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    if (row.br):
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()
        
    colunm_name = ' '.join(row.contents)
    
    # Filter the digit and empty names
    if not(colunm_name.strip().isdigit()):
        colunm_name = colunm_name.strip()
        return colunm_name    

# Set URL and parse the page
static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
response = requests.get(static_url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all tables
html_tables = soup.find_all("table")
first_launch_table = html_tables[2]

# Create column names
column_names = []
for th in first_launch_table.find_all('th'):
    name = th.text.strip()
    if name:  # Check if name is not empty
        column_names.append(name)

print(column_names)

# Initialize empty lists for launch_dict keys


# Define the columns we need
launch_dict= dict.fromkeys(column_names)

# Remove an irrelvant column

# Let's initial the launch_dict with each value to be an empty list
launch_dict['Flight No.'] = []
launch_dict['Launch site'] = []
launch_dict['Payload'] = []
launch_dict['Payload mass'] = []
launch_dict['Orbit'] = []
launch_dict['Customer'] = []
launch_dict['Launch outcome'] = []
# Added some new columns
launch_dict['Version Booster']=[]
launch_dict['Booster landing']=[]
launch_dict['Date']=[]
launch_dict['Time']=[]

# Parse each row in the target table and fill data into launch_dict
for table in soup.find_all('table', "wikitable plainrowheaders collapsible"):
    for rows in table.find_all("tr"):
        if rows.th and rows.th.string and rows.th.string.strip().isdigit():
            row = rows.find_all('td')
            
            # Extract each required data and append to launch_dict
            flight_number = rows.th.string.strip()
            launch_dict['Flight No.'].append(flight_number)

            datatimelist = date_time(row[0])
            date = datatimelist[0].strip(',')
            time = datatimelist[1]
            launch_dict['Date'].append(date)
            launch_dict['Time'].append(time)

            bv = booster_version(row[1])
            launch_dict['Version Booster'].append(bv if bv else row[1].a.string)

            launch_dict['Launch site'].append(row[2].a.string if row[2].a else None)
            launch_dict['Payload'].append(row[3].a.string if row[3].a else None)
            launch_dict['Payload mass'].append(get_mass(row[4]))
            launch_dict['Orbit'].append(row[5].a.string if row[5].a else None)
            launch_dict['Customer'].append(row[6].a.string if row[6].a else None)
            launch_dict['Launch outcome'].append(list(row[7].strings)[0] if row[7].strings else None)
            launch_dict['Booster landing'].append(landing_status(row[8]) if row[8].strings else None)

# Convert the dictionary to a DataFrame
df = pd.DataFrame(launch_dict)
df = df.drop(columns=["Date andtime (UTC)"])

print(df.head())

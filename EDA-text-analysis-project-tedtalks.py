## Research question:
# .......

import pandas as pd
data = pd.read_csv('tedx_dataset.csv', on_bad_lines='skip', delimiter=';', skipinitialspace=True)

# First start with data cleaning to prepare the data for the analysis
data_copy = data.copy()

# Show info of the dataset, including datatype, null values and amount of rows/columns
data.info()

# Look at shape of dataset
print(data.shape)

# Check for duplicates
print(data[data.duplicated()])

# Check for outliers
# ......

# Rename column
data = data.rename(columns={'num_views': 'views'})

# Replace commas and change type of column to integer
data['views'] = data['views'].str.replace(',', '').str.strip()
data['views'] = data['views'].astype(int)


# Convert results that are in hours to right format
def convert_hours(result):
    if 'h' in result:
        if len(result) == 5:
            result = result.replace('1h ', '01:0').replace('m', ':00')
            return result
        elif len(result) == 6:
            result = result.replace('1h ', '01:').replace('m', ':00')
            return result
        else:
            return result


# Locate results that are in hours and apply function above
condition = data['duration'].str.contains('h')
duration_hours = data.loc[condition, 'duration']
data.loc[condition, 'duration'] = duration_hours.apply(convert_hours)


# Locate results that have unnecessary prefixes, and split from their values
duration_zeros = data['duration'].str.len() > 5
duration_new = data.loc[duration_zeros, 'duration']
data.loc[duration_zeros, 'duration'] = duration_new.str.slice(0, 5)

print(data['duration'][387])


# Add the right prefix to every result so the format will be hh:mm:ss
def add_prefix(value):
    if len(value) == 5:
        return '00:' + value
    elif len(value) == 4:
        return '00:0' + value
    else:
        return value


# Apply function above to the results of duration column
data['duration'] = data['duration'].apply(add_prefix)

# Strip column from any extra spaces to convert datatype
data['duration'] = data['duration'].str.strip()

# Convert column to datetime and extract only the time part
data['duration'] = pd.to_datetime(data['duration'], format='%H:%M:%S')
data['duration'] = data['duration'].dt.time


# Remove text so the string can be converted to datetime
data['posted'] = data['posted'].str.replace('Posted Jan ', '01-').str.replace('Posted Feb ', '02-').str.replace('Posted Mar ', '03-').str.replace('Posted Apr ', '04-').str.replace('Posted May ', '05-').str.replace('Posted Jun ', '06-').str.replace('Posted Jul ', '07-').str.replace('Posted Aug ', '08-').str.replace('Posted Sep ', '09-').str.replace('Posted Oct ', '10-').str.replace('Posted Nov ', '11-').str.replace('Posted Dec ', '12-').str.strip()

# Convert data type to datetime
data['posted'] = pd.to_datetime(data['posted'])
data['posted_month'] = data['posted'].dt.month
data['posted_year'] = data['posted'].dt.year

# Remove first unused column
data = data.drop(['idx'], axis=1)


# Now the data is cleaned, start with the exploratory data analysis!

# Check at which date most videos were posted
print(data['posted'].value_counts()[:5])
# Check in which years most videos were posted
print(data['posted_month'].value_counts()[:5])
# Check in which months most videos were posted
print(data['posted_year'].value_counts()[:5])


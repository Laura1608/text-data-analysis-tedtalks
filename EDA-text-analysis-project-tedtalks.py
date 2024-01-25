## Research question:
# .......

import pandas as pd

# Read the data file
data = pd.read_csv('tedx_dataset.csv', on_bad_lines='skip', delimiter=';', skipinitialspace=True)

# Make a copy of dataset
data_copy = data.copy()

# Show info of the dataset, including datatype, null values and amount of rows/columns
data.info()

# Look at shape of dataset
print(data.shape)

# Rename column with amount of views
data = data.rename(columns={'num_views': 'views'})

data.info()

# Change type of column 'views' to integer
# ......


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

# Change datatype to timedelta
data['duration'] = pd.to_timedelta(data['duration'])
print(data['duration'][:20])


# # Assuming data is your DataFrame
# data['duration'] = pd.to_datetime(data['duration'], format='%M:%S', errors='coerce')

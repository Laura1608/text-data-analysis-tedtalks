""" Research questions:
How does date influence the success of a TED Talk? (success measured in terms of views)
- What is the most common post date of all videos vs successful videos?
How does duration influence the success of a TED Talk? (success measured in terms of views)
- What is the average duration of all videos vs successful videos?
How does language influence the success of a TED Talk? (success measured in terms of views)
- What words are commonly used in all videos vs successful videos?
- What is the average title length of all videos vs successful videos?
- What sentiment is being used in all videos vs successful videos?
"""

import pandas as pd
import re
import matplotlib.pyplot as plt
import plotly.express as px

# First start with data cleaning to prepare the data for the analysis
data = pd.read_csv('tedx_dataset.csv', on_bad_lines='skip', delimiter=';', skipinitialspace=True)
data_copy = data.copy()

# Show info of the dataset, including datatype, null values and amount of rows/columns
data.info()

# Look at shape of dataset
print(data.shape)

# Check for duplicates
print(data[data.duplicated()])

# Data cleaning 'views' column
# Rename column
data = data.rename(columns={'num_views': 'views'})

# Replace commas and change type of column to integer
data['views'] = data['views'].str.replace(',', '').str.strip()
data['views'] = data['views'].astype(int)

# Data cleaning 'duration' column
# Take closer look at results that are in hours
print(data['duration'][387])


# Convert results so that duration column can be converted to the right datatype
def convert_hours(result):
    # Search for results that contains hours
    pattern_single = re.compile(r"1h ([0-9])m")
    pattern_double = re.compile(r"1h ([1-5][0-9]|60)m")

    # Extract minutes from result, and then rewrite result output
    if pattern_single.search(result):
        result = f'01:0{result[3:4]}:00'
    elif pattern_double.search(result):
        result = f'01:{result[3:5]}:00'

    return result


# Apply the function to the 'duration' column
data['duration'] = data['duration'].apply(convert_hours)
print(data['duration'][387])


# Locate results that have unnecessary prefixes, and split from value
duration_zeros = data['duration'].str.len() > 5
duration_new = data.loc[duration_zeros, 'duration']
data.loc[duration_zeros, 'duration'] = duration_new.str.slice(0, 5)


# Add the right prefix to all results so the format will be hh:mm:ss
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
# print(data['duration'][:20])

# Data cleaning 'posted' column
# Remove text so the string can be converted to datetime
data['posted'] = data['posted'].str.replace('Posted Jan ', '01-').str.replace('Posted Feb ', '02-').str.replace('Posted Mar ', '03-').str.replace('Posted Apr ', '04-').str.replace('Posted May ', '05-').str.replace('Posted Jun ', '06-').str.replace('Posted Jul ', '07-').str.replace('Posted Aug ', '08-').str.replace('Posted Sep ', '09-').str.replace('Posted Oct ', '10-').str.replace('Posted Nov ', '11-').str.replace('Posted Dec ', '12-').str.strip()

# Convert data type to datetime
data['posted'] = pd.to_datetime(data['posted'], format="%m-%Y")
data['posted_month'] = data['posted'].dt.month
data['posted_year'] = data['posted'].dt.year


# Data overview of each integer column by looking at outliers and distribution
data.boxplot('posted_year')
# plt.show()

data.boxplot('posted_month')
# plt.show()

data.boxplot('views')
# plt.show()

# Remove unused column
data = data.drop(['idx'], axis=1)

# Now the data is cleaned, start with the exploratory data analysis!
# Define successful or well-viewed videos based on statistical distribution (> 75%):
data['views'].describe().astype(int)
all_videos = data.copy()
successful_videos = data[(data['views'] > 2117389)]

print("All_videos amount: ", len(all_videos), "\n" "Successful_videos amount: ", len(successful_videos))

# Get value counts (amount) of videos posted per month per year
all_videos_posted = all_videos[['posted_year', 'posted_month']].groupby('posted_year')['posted_month'].value_counts().sort_values(ascending=False)[:8]
successful_videos_posted = successful_videos[['posted_year', 'posted_month']].groupby('posted_year')['posted_month'].value_counts().sort_values(ascending=False)[:8]
print(all_videos_posted), print(successful_videos_posted)

# Get the 5 years with most video views in total
all_videos_views = all_videos[['posted_year', 'views']].groupby('posted_year')['views'].sum().sort_values(ascending=False)[:5]
successful_videos_views = successful_videos[['posted_year', 'views']].groupby('posted_year')['views'].sum().sort_values(ascending=False)[:5]
print(all_videos_views), print(successful_videos_views)

data['views_year_total'] = data.groupby('posted_year')['views'].transform('sum').round()

data['views_year_avg'] = data.groupby('posted_year')['views'].transform('mean').round()

data['amount_videos_year'] = data.groupby('posted_year')['url'].transform('nunique')

fig_videos_year = px.bar(data, x='posted_year', y='amount_videos_year', color='views_year_avg', title='TED Talks posted per year')
fig_videos_year.show()
fig_views_year = px.bar(data, x='posted_year', y='views_year_avg', title='TED Talks avg views per year')
fig_views_year.show()

""" Research questions:
RQ1: How does date influence the success of a TED Talk? (success measured in terms of views)
- At which months were the most videos posted? (all videos vs successful videos)
- Which months had the most views (on average)?
- At which years were the most videos posted? (all videos vs successful videos)
- Which years had the most views (on average)?
RQ2: How does duration influence the success of a TED Talk? (success measured in terms of views)
- What is the average duration of all videos vs successful videos?
RQ3: How does language influence the success of a TED Talk? (success measured in terms of views)
- What words are commonly used in all videos vs successful videos?
- What is the average title length of all videos vs successful videos?
- What sentiment is being used in all videos vs successful videos?
"""

import pandas as pd
import re
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plot

import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# First start with data cleaning to prepare the data for the analysis
dataset = pd.read_csv('tedx_dataset.csv', on_bad_lines='skip', delimiter=';', skipinitialspace=True)
data = pd.DataFrame(data=dataset)
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
data['views'] = data['views'].astype('Int64')

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

# Convert duration column to datetime and extract only the time part
data['duration'] = pd.to_datetime(data['duration'], format='%H:%M:%S')
data['duration_time'] = data['duration'].dt.time

# Data cleaning 'posted' column
# Remove text so the string can be converted to datetime
data['posted'] = data['posted'].str.replace('Posted Jan ', '01-').str.replace('Posted Feb ', '02-').str.replace('Posted Mar ', '03-').str.replace('Posted Apr ', '04-').str.replace('Posted May ', '05-').str.replace('Posted Jun ', '06-').str.replace('Posted Jul ', '07-').str.replace('Posted Aug ', '08-').str.replace('Posted Sep ', '09-').str.replace('Posted Oct ', '10-').str.replace('Posted Nov ', '11-').str.replace('Posted Dec ', '12-').str.strip()

# Convert data type to datetime
data['posted'] = pd.to_datetime(data['posted'], format="%m-%Y")
data['posted_month'] = data['posted'].dt.month.astype('Int64')
data['posted_year'] = data['posted'].dt.year.astype('Int64')

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
data['views'].describe().astype('Int64')
all_videos = data.copy()
successful_videos = data[(data['views'] > 2117389)]

print("All_videos amount: ", len(all_videos), "\n" "Successful_videos amount: ", len(successful_videos))


'''RQ1: How does date influence the success of a TED Talk? (success measured in terms of views)
- At which months were the most videos posted? (all videos vs successful videos)
- Which months had the most views (on average)?
- At which years were the most videos posted? (all videos vs successful videos)
- Which years had the most views (on average)?'''

amount_videos_month = all_videos.groupby('posted_month')['url'].nunique().astype('Int64').sort_values(ascending=False)[:5]
amount_videos_month_s = successful_videos.groupby('posted_month')['url'].nunique().astype('Int64').sort_values(ascending=False)[:5]
print("Amount of videos posted per month: ", amount_videos_month, "\n", "Amount of successful videos posted per month: ", amount_videos_month_s)
# Conclusion: no big differences per month in post date. The month when a video is posted, is not of influence on its success.

amount_videos_year = all_videos.groupby('posted_year')['url'].nunique().astype('Int64')
amount_videos_year_s = successful_videos.groupby('posted_year')['url'].nunique().astype('Int64')
# Plot results in bar chart for better overview
px.bar(amount_videos_year, title='Amount of TED Talks posted per year (all videos)').show()
px.bar(amount_videos_year_s, title='Amount of TED Talks posted per year (successful videos)').show()
# Conclusion: In 2018, 2019 and 2020 were most videos posted, while only 2019 had the most successful videos posted. That makes quantity no predictor for success.

views_month_avg = all_videos.groupby('posted_month')['views'].mean().astype('Int64').sort_values(ascending=False)[:5]
views_month_avg_s = successful_videos.groupby('posted_month')['views'].mean().astype('Int64').sort_values(ascending=False)[:5]
print("Average amount of views per month (all videos): ", views_month_avg, "\n", "Average amount of views per month (successful videos): ", views_month_avg_s)
# Conclusion: on average more successful videos were viewed in March, further no big differences per month.
# Once again, the month when a video is posted, is not of influence on its success.

views_year_avg = all_videos.groupby('posted_year')['views'].mean().astype('Int64')
views_year_avg_s = successful_videos.groupby('posted_year')['views'].mean().astype('Int64')
# Plot results in bar chart for better overview
px.bar(views_year_avg, title='Average amount of views of TED Talks per year (all videos)').show()
px.bar(views_year_avg_s, title='Average amount of views of TED Talks per year (successful videos)').show()
# Conclusion: The difference between the years is bigger when all videos are included, than when only looking at the best viewed videos.
# Meaning that the best viewed videos are more consistently watched over the years.
# This makes the year not of influence on its success (except for the first year that TED Talks were published online -- 2006 jumps out).
# In the last few years, views have been decreasing on average (despite most videos being posted).

'''RQ2: How does duration influence the success of a TED Talk? (success measured in terms of views)
- What is the average duration of all videos vs successful videos?'''

# Convert duration column to integer to calculate the mean
all_videos['duration_mean'] = all_videos['duration'].values.astype(np.int64).mean()
successful_videos['duration_mean'] = successful_videos['duration'].values.astype(np.int64).mean()

# Convert column back to datetime and extract only the time part
all_videos['duration_mean'] = pd.to_datetime(all_videos['duration_mean'])
all_videos['duration_mean'] = pd.to_datetime(all_videos['duration_mean'], format='%H:%M:%S').dt.floor('s').dt.time
successful_videos['duration_mean'] = pd.to_datetime(successful_videos['duration_mean'])
successful_videos['duration_mean'] = pd.to_datetime(successful_videos['duration_mean'], format='%H:%M:%S').dt.floor('s').dt.time

# Compare duration of videos compared to average
print("Average duration of all videos: ", all_videos['duration_mean'].iloc[0])
print("Average duration of successful videos: ", successful_videos['duration_mean'].iloc[0])
# Conclusion: successful videos are on average about half a minute longer, but the difference is very small. Duration seems not to be a factor determining success.

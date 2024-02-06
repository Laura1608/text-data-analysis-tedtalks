""" Research questions:
RQ1: How does date influence the success of a TED Talk? (success measured in terms of views)
- At which months were the most videos posted? (all videos vs successful videos)
- Which months had the most views (on average)?
- At which years were the most videos posted? (all videos vs successful videos)
- Which years had the most views (on average)?
RQ2: How does duration influence the success of a TED Talk? (success measured in terms of views)
- What is the average duration of all videos vs successful videos?
RQ3: How does language influence the success of a TED Talk? (success measured in terms of views)
- What is the average title length of all videos vs successful videos?
- What sentiment is being used in all videos vs successful videos?
- What words are commonly used in all videos vs successful videos?"""

# Import necessary libraries for data cleaning, exploratory data analysis (EDA) and text analysis
import pandas as pd
import re
import numpy as np
import plotly.express as px
# nltk.download('all')
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.util import ngrams

# Ignore certain warning in code
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# Start with data cleaning to prepare the data for the analysis
dataset = pd.read_csv('tedx_dataset.csv', delimiter=';', header=0)
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


'RQ1: How does date influence the success of a TED Talk? (success measured in terms of views)'

# Create new variable with amount of videos posted, grouped by month
amount_videos_month = all_videos.groupby('posted_month')['url'].nunique().astype('Int64').sort_values(ascending=False)[:5]
amount_videos_month_s = successful_videos.groupby('posted_month')['url'].nunique().astype('Int64').sort_values(ascending=False)[:5]

print("Amount of videos posted per month: ", amount_videos_month)
print("Amount of successful videos posted per month: ", amount_videos_month_s)
# Findings: No very big differences in monthly post date.


# Create new variables with amount of videos posted, grouped by year
amount_videos_year = all_videos.groupby('posted_year')['url'].nunique().astype('Int64')
amount_videos_year_s = successful_videos.groupby('posted_year')['url'].nunique().astype('Int64')

# Plot results in bar chart for better overview
px.bar(amount_videos_year, title='Amount of TED Talks posted per year (all videos)').show()
px.bar(amount_videos_year_s, title='Amount of TED Talks posted per year (successful videos)').show()
# Findings: Most videos were posted in 2018, 2019 and 2020, while successful videos were mostly posted in 2019 only.


# Create new variables with average amount of views, grouped by month
views_month_avg = all_videos.groupby('posted_month')['views'].mean().astype('Int64').sort_values(ascending=False)[:5]
views_month_avg_s = successful_videos.groupby('posted_month')['views'].mean().astype('Int64').sort_values(ascending=False)[:5]

print("Average amount of views per month (all videos): ", views_month_avg, "\n", "Average amount of views per month (successful videos): ", views_month_avg_s)
# Findings: On average, more successful videos were viewed in March. Further, no big differences in months.


# Create new variables with average amount of views, grouped by year
views_year_avg = all_videos.groupby('posted_year')['views'].mean().astype('Int64')
views_year_avg_s = successful_videos.groupby('posted_year')['views'].mean().astype('Int64')

# Plot results in bar chart for better overview
px.bar(views_year_avg, title='Average amount of views of TED Talks per year (all videos)').show()
px.bar(views_year_avg_s, title='Average amount of views of TED Talks per year (successful videos)').show()

# Findings: In the first year that TED Talks were published online (2006), all videos got many views.
# The difference between years is bigger when all videos are included, than when only looking at the best viewed videos.
# Meaning that successful videos are more consistently watched over the years.
# In the last few years, views have been decreasing (despite most videos being posted).


# Conclusion RQ1: How does date influence the success of a TED Talk? (in terms of views)
# The month when a video was published and the quantity of videos posted, are no predictors for the success of a video.
# The year doesn't influence views either. However, what does, is novelty and the success of a video itself.


'RQ2: How does duration influence the success of a TED Talk? (success measured in terms of views)'

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
# Findings: successful videos are on average about half a minute longer, but the difference is very small.


# Conclusion RQ2: How does duration influence the success of a TED Talk?
# The duration of a video doesn't seem to be a factor in determining its success.


'RQ3: How does language influence the success of a TED Talk? (success measured in terms of views)'

# Define stop word list
final_stopwords = nltk.corpus.stopwords.words('english')
final_stopwords.append('u')

# Create empty lists to save text
all_videos_all_titles = []
successful_videos_all_titles = []


# Create function to pre-process text
def preprocess_text(row):
    # Tokenize every word in each row and remove interpunction
    tokenizer = RegexpTokenizer(r'\w+')
    tokens_title = tokenizer.tokenize(row['title'].lower())

    # Lemmatize the tokens for keeping base part only
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens_title = [lemmatizer.lemmatize(token) for token in tokens_title]

    # Remove stop words
    filtered_tokens_title = [token for token in lemmatized_tokens_title if token not in final_stopwords]

    # Join all tokens back into a string
    processed_text_title = ' '.join(filtered_tokens_title)

    # Calculate title length per row
    title_length = len(processed_text_title)

    # Append tokenized strings to empty lists for later analysis
    [all_videos_all_titles.append(token) for token in filtered_tokens_title]
    [successful_videos_all_titles.append(token) for token in filtered_tokens_title if data.loc[row.name, 'views'] > 2117389]

    return processed_text_title, title_length


# Apply function to dataframe, row by row (axis=1), and create new column
all_videos[['processed_title', 'title_length']] = all_videos.apply(preprocess_text, axis=1, result_type='expand')
successful_videos[['processed_title', 'title_length']] = successful_videos.apply(preprocess_text, axis=1, result_type='expand')

# Calculate the average length of titles
print("Average title length of all videos: ", all_videos['title_length'].astype(np.int64).mean().round(1))
print("Average title length of successful videos: ", successful_videos['title_length'].astype(np.int64).mean().round(1))
# Findings: Titles of successful videos are shorter on average, but the difference is very small.


# Create function to perform sentiment analysis
def get_sentiment(row):
    analyzer = SentimentIntensityAnalyzer()
    score_title = analyzer.polarity_scores(row['processed_title'])
    sentiment_title = 1 if score_title['pos'] > 0 else 0
    return sentiment_title


# Apply function to dataframe, row by row (axis=1), and create new column
all_videos['sentiment_title'] = all_videos.apply(get_sentiment, axis=1, result_type='expand')
successful_videos['sentiment_title'] = successful_videos.apply(get_sentiment, axis=1, result_type='expand')

# Plot pie chart to compare sentiment analysis for both datasets
px.pie(values=all_videos.index, names=all_videos['sentiment_title'].map({1: 'Positive', 0: 'Negative'}), title='Sentiment in titles all videos').show()
px.pie(values=successful_videos.index, names=successful_videos['sentiment_title'].map({1: 'Positive', 0: 'Negative'}), title='Sentiment in titles successful videos').show()
# Findings: The sentiment of titles in both datasets is more or less equal. In both cases 1/3 positive and 2/3 negative outcome.


# Frequency distribution to find most common words in titles of both datasets
print("Most common words in titles of all videos: ", "\n", FreqDist(all_videos_all_titles).most_common(5))
print("Most common words in titles of successful videos: ", "\n", FreqDist(successful_videos_all_titles).most_common(5))

# Pair words through ngrams and find most common 2-word combinations
all_videos_bigram_title = list(ngrams(all_videos_all_titles, 2))
all_videos_common_bigram_title = FreqDist(all_videos_bigram_title).most_common(5)
print("Most common 2-word combinations in titles of all videos: ", "\n", all_videos_common_bigram_title)

successful_videos_bigram_title = list(ngrams(successful_videos_all_titles, 2))
successful_videos_common_bigram_title = FreqDist(successful_videos_bigram_title).most_common(5)
print("Most common 2-word combinations in titles of successful videos: ", "\n", successful_videos_common_bigram_title)

# Pair words through ngrams and find most common 3-word combinations
all_videos_trigram_title = list(ngrams(all_videos_all_titles, 3))
all_videos_common_trigram_title = FreqDist(all_videos_trigram_title).most_common(5)
print("Most common 3-word combinations in titles of all videos: ", "\n", all_videos_common_trigram_title)

successful_videos_trigram_title = list(ngrams(successful_videos_all_titles, 3))
successful_videos_common_trigram_title = FreqDist(successful_videos_trigram_title).most_common(5)
print("Most common 3-word combinations in titles of successful videos: ", "\n", successful_videos_common_trigram_title)

# Findings: There is some overlap in the words used in all videos and successful videos.
# In both datasets, words that often occur, are: "life", "world", and "make".
# In successful videos the emphasis lies more on "way" and "work" (active), while all videos contain more words like "future" and "new" (descriptive).

# In both datasets, word combinations that often occur, are: "climate change", "3 way[s]", and "look like".
# In successful videos the emphasis lies on topics such as "mental health", "lesson longest study", "good night sleep", and "food [you] eat affect".
# Thus, topics that refer to or improve daily life and mental/physical health.


# Conclusion RQ3: How does language influence the success of a TED Talk?
# Both length and sentiment of titles do not seem to be factors that influence success (amount of views). Most titles are more negative that positive.
# Successful videos are more often about topics as life quality and health, and ways to improve it.


'Overall conclusion data & text analysis:'
# The month or year when a video was published, the quantity of videos posted, and the duration of a video are no predictors for success.
# However, novelty and the success of a video, are. These factors are not specifically researched here, but seem to increase the amount of views a video gets.
# Both length and sentiment of titles do not seem to be predictors of success. Most titles are more negative that positive in general.
# Successful videos are more often about topics as life quality and health, and ways to improve it. It shows an interest of viewers in these kind of topics.
#

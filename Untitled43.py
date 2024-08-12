#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
csv_path = r"C:\Users\Administrator\Downloads\twitter_validation.csv"
pf = pd.read_csv(csv_path)
pf.head()


# In[13]:


# Check for missing values
print(pf.isnull().sum())

# Display the first few rows of the dataset
print(pf.head())


# In[18]:


pf = [
    {'column1': 'value1', 'column2': 'value2', 'column3': None},
    {'column1': 'value3', 'column2': None, 'column3': 'value4'},
    {'column1': 'value5', 'column2': 'value6', 'column3': 'value7'},
    # ... more rows
]


# In[19]:


# Initialize a dictionary to count missing values
missing_values = {}

# Loop through each row in the data
for row in pf:
    for key, value in row.items():
        if value is None:  # Check if the value is missing
            if key in missing_values:
                missing_values[key] += 1
            else:
                missing_values[key] = 1

# Display missing values count
for key, value in missing_values.items():
    print(f"{key}: {value} missing values")


# In[20]:


# Number of rows to display
n_rows = 5

# Loop through and print the first few rows
for i, row in enumerate(pf):
    if i < n_rows:
        print(row)
    else:
        break


# In[63]:


import matplotlib.pyplot as plt
from collections import Counter
import random

# Example data structure
pf = [
    {'sentiment': 'negative', 'Positive': 'bad experience horrible terrible'},
    {'sentiment': 'negative', 'Positive': 'horrible bad service'},
    {'sentiment': 'positive', 'Positive': 'great experience excellent service'}
]

# Extract and join negative sentiment words
negative_words = ' '.join([row['Positive'] for row in pf if row['sentiment'] == 'negative'])

# Count word frequencies
word_freq = Counter(negative_words.split())

# Define colors
background_color = 'silver'
text_colors = ['red', 'yellow', 'blue', 'orange', 'green']

# Plot a basic word cloud with custom colors
plt.figure(figsize=(1,1))
plt.gca().set_facecolor(background_color)  # Set background color

for i, (word, freq) in enumerate(word_freq.items()):
    plt.text(i % 10, +i // 10, word, fontsize=freq * 7, color=random.choice(text_colors))

plt.axis('off')
plt.title("Negative Sentiment Word Cloud", color='brown')  # Title color
plt.show()


# In[65]:


import matplotlib.pyplot as plt
from collections import Counter
import random

# Example data structure (make sure to use a DataFrame like object, or convert your data into a pandas DataFrame)
df = [
    {'sentiment': 'negative', 'Positive': 'bad experience horrible terrible'},
    {'sentiment': 'negative', 'Positive': 'horrible bad service'},
    {'sentiment': 'positive', 'Positive': 'great experience excellent service'}
]

# Convert to DataFrame (if necessary)
import pandas as pd
pf = pd.DataFrame(df)

# Extract and join negative sentiment words
negative_words = ' '.join([row['Positive'] for row in df if row['sentiment'] == 'negative'])

# Count word frequencies
word_freq = Counter(negative_words.split())

# Define colors
background_color = 'black'
text_colors = ['red', 'yellow', 'white', 'orange', 'green']

# Plot a basic word cloud with custom colors
plt.figure(figsize=(10, 5))
plt.gca().set_facecolor(background_color)  # Set background color

for i, (word, freq) in enumerate(word_freq.items()):
    plt.text(i % 10, -i // 10, word, fontsize=freq * 5, color=random.choice(text_colors))

plt.axis('off')
plt.title("Negative Sentiment Word Cloud", color='white')  # Title color
plt.show()

# Plot sentiment distribution pie chart
pf['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['orange', 'yellow', 'purple'])
plt.title('Sentiment Distribution')
plt.ylabel('')
plt.show()


# In[67]:


import pandas as pd
import numpy as np
start_date = pd.to_datetime('2023-01-01')
num_entries = len(df)
date_range = pd.date_range(start=start_date, periods=num_entries, freq='D')
pf['proxy_date'] = date_range
pf.head()


# In[75]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'proxy_date' is the column with date information
pf['date'] = pd.to_datetime(pf['proxy_date'])

# Group by the period (e.g., month) and sentiment
sentiment_over_time = pf.groupby([pf['date'].dt.to_period('M'), 'sentiment']).size().unstack().fillna(9)

# Plotting
sentiment_over_time.plot(kind='line', marker='o')  
plt.title('Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Sentiments')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# In[78]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Make a copy of the DataFrame
pf_encoded = pf.copy()

# Initialize a dictionary to store label encoders
label_encoders = {}

# Encode categorical columns
for column in pf_encoded.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    pf_encoded[column] = label_encoders[column].fit_transform(pf_encoded[column].astype(str))

# Compute the correlation matrix
correlation_matrix = pf_encoded.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


# In[86]:


import seaborn as sns
sns.boxplot(x='sentiment', y='proxy_date', data=pf)
plt.title('proxy_date Distribution Across Brands')
plt.xticks(rotation=90)
plt.show()


# In[ ]:





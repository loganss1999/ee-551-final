import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report


df_meta = pd.read_csv('movies_metadata.csv')
df_meta = df_meta[df_meta['vote_count'] >= 5]
df_keywords = pd.read_csv('keywords.csv')

df_keywords['keywords'] = [keyword.strip("[]").replace('{', '').replace('\'', '').replace(': ', '').replace(', ', '').replace('name', '').replace('id', '').replace(' ', '').replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').replace('}', ' ') for keyword in df_keywords['keywords']]
df_meta['id'] = df_meta['id'].astype('int64')
df = pd.merge(df_meta, df_keywords, on="id", how="inner")
df = df[df['revenue'] > 100000]
df['vote_sentiment'] = df['vote_average'].apply(lambda vote : +1 if vote > 7.5 else (-1 if vote < 5.0 else 0))
df = df[df['vote_sentiment'] != 0]
df['rev_sentiment'] = df['revenue'].apply(lambda rev : +1 if rev > 5000000 else -1)

conditions = [
    ((df['vote_sentiment'] == 1) & (df['rev_sentiment'] == 1)),
    ((df['vote_sentiment'] == 1) & (df['rev_sentiment'] == -1)),
    ((df['vote_sentiment'] == -1) & (df['rev_sentiment'] == 1)),
    ((df['vote_sentiment'] == -1) & (df['rev_sentiment'] == -1))
    ]
values = ['hit', 'underrated', 'overrated', 'flop']    
df['sentiment'] = np.select(conditions, values)
print(df['sentiment'])


good = df[df['vote_sentiment'] == 1]
bad = df[df['vote_sentiment'] == -1]
hit = good[good['rev_sentiment'] == 1]
underrated = good[good['rev_sentiment'] == -1]
overrated = bad[bad['rev_sentiment'] == 1]
flop = bad[bad['rev_sentiment'] == 1]

# scatterplot
# sc = df.plot.scatter(x='vote_average', y='revenue', c='Black')
# scatter = sc.get_figure()
# scatter.savefig("scatter.png")

# histogram
# fig1 = px.histogram(df, x="vote_average", nbins=20)
# fig1.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  # marker_line_width=1.5)
# fig1.update_layout(title_text='Vote Average')
# fig1.write_image("vote_histogram.png")

# fig2 = px.histogram(df, x="revenue", nbins=10000)
# fig2.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  # marker_line_width=1.5)
# fig2.update_layout(title_text='Revenue')
# fig2.write_image("rev_histogram.png")

def wordcloud(name, df) :
    stopwords = set(STOPWORDS) 
    stopwords.update(["br", "href"])
    textt = " ".join(str(words) for words in df.keywords)
    wordcloud = WordCloud(stopwords=stopwords).generate(textt)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(name + '_wordcloud.png')

wordcloud('good', good)
wordcloud('bad', bad)
wordcloud('hit', hit)
wordcloud('flop', flop)
wordcloud('overrated', overrated)
wordcloud('underrated', underrated)

df_sen = df[['keywords','sentiment']]

df['random_number'] = np.random.randn(len(df.index))
train = df[df['random_number'] <= 0.8]
test = df[df['random_number'] > 0.8]

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['keywords'])
test_matrix = vectorizer.transform(test['keywords'])

lr = LogisticRegression()

X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']

lr.fit(X_train,y_train)
predictions = lr.predict(X_test)

new = np.asarray(y_test)
print(confusion_matrix(predictions,y_test))

print(classification_report(predictions,y_test))
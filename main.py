from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import tweepy
import numpy as np
import pandas as pd
from credencial import token
from datetime import datetime
import plotly.express as px


client = tweepy.Client(bearer_token=token)
hashtag = '#BrasilNaCopa'
print(hashtag)
qtdnegativo = 0
qtdneutro = 0
qtdpositivo = 0
df = pd.DataFrame(columns = ['Tweet','Negativo', 'Neutro', 'Positivo', 'Considerado'])

query = hashtag + ' -is:retweet'
# has:geo
tweets = tweepy.Paginator(client.search_recent_tweets, query=query, tweet_fields=['context_annotations', 'created_at'], max_results=100).flatten(limit=100)
for tweet in tweets:
    try:
        frase = tweet.text
        print(frase)
        tweet_words = []
        for word in frase.split(' '):
            if word.startswith('@') and len(word) > 1:
                word = '@user'
            elif word.startswith('http'):
                word = "http"
            tweet_words.append(word)
        tweet_proc = " ".join(tweet_words)

        xlm_roberta = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        model = AutoModelForSequenceClassification.from_pretrained(xlm_roberta)
        tokenizer = AutoTokenizer.from_pretrained(xlm_roberta)
        encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
        output = model(**encoded_tweet)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        considerado = ''
        max_value = max(scores)
        max_value_index = np.argmax(scores)
        if max_value_index == 0:
            considerado = 'NEGATIVO'
            qtdnegativo += 1
        elif max_value_index == 1:
            considerado = 'NEUTRO'
            qtdneutro += 1
        elif max_value_index == 2:
            considerado = 'POSITIVO'
            qtdpositivo += 1
        print(' >> ', considerado)
        new_row = pd.Series({'Tweet': frase, 'Negativo': scores[0], 'Neutro': scores[1], 'Positivo': scores[2], 'Considerado': considerado})
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    except Exception as e:
        print(e)


nomefile = 'resultado'+datetime.now().strftime("_%m%d%Y_%H%M%S")+'.csv'
df.to_csv(nomefile, sep=';', index=False)
temp = pd.DataFrame(df['Considerado'].value_counts())
temp.index.name = 'val'
temp.columns = ['count']
temp = temp.reset_index()

fig = px.pie(temp, names='val', values='count')
fig.show()
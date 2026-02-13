import pandas as pd
import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

df = pd.read_csv("Semana 1/sentences.csv")

df['Sentimiento'] = df['sentence'].apply(lambda x: sid.polarity_scores(x)['compound'])

df.to_csv("Semana 1/Mensajes_con_sentimiento.csv")

"""
Vader = VADER means Valence Aware Dictionary and sEntiment Reasoner.
Este es un gran diccionario con muchas palabras que permite hacer analisis de sentimiento pero solo en ingles
Cada palabra tiene un puntaje emocional:

great → +3.1
excellent → +3.8
terrible → −3.6
Cuando analizas una frase, suma esos valores.

"""
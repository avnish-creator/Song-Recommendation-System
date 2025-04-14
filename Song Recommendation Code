import pandas as pd
import re
import numpy as np
from transformers import BertTokenizer, TFBertModel
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, HTML
# Load dataset
df_yt = pd.read_csv('/content/US_videos_data.csv')
df_yt = df_yt[['title','channelTitle','likes','dislikes','thumbnail_link','description']]

# Drop duplicates and nulls
df_yt = df_yt.drop_duplicates(subset=['title'])
df_yt.dropna(inplace=True)

# Clean titles
df_yt['clean_title'] = df_yt['title'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x.lower()))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')
def get_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=512)
    outputs = model(inputs)
    return outputs.pooler_output.numpy()[0]  # Single vector
df_yt['embeddings'] = df_yt['clean_title'].apply(lambda x: get_bert_embeddings(x, tokenizer, model))
def compute_cosine_similarity(embedding, embeddings):
    return cosine_similarity(embedding.reshape(1, -1), np.vstack(embeddings)).flatten()

def recommend_videos(title, df, tokenizer, model, top_n=5):
    cleaned_title = re.sub('[^A-Za-z0-9]+', ' ', title.lower())
    embedding = get_bert_embeddings(cleaned_title, tokenizer, model)
    similarities = compute_cosine_similarity(embedding, df['embeddings'].tolist())
    df['similarity'] = similarities
    df_sorted = df.sort_values(by='similarity', ascending=False)
    return df_sorted[df_sorted['title'] != title].head(top_n)[['title', 'channelTitle','likes','dislikes','thumbnail_link', 'similarity']]
def display_recommendations(recommendations):
    html = '<div style="display: flex; flex-wrap: wrap; justify-content: space-around;">'
    for _, row in recommendations.iterrows():
        html += f'''
        <div style="width: 20%; margin: 2px; text-align: center; border: 1px solid #ddd; padding: 2px; border-radius: 10px;">
            <img src="{row['thumbnail_link']}" alt="{row['title']}" style="width: 100%; border-radius: 5px;">
            <h4>{row['title']}</h4>
            <p>Channel: {row['channelTitle']}</p>
            <p>Likes: {row['likes']} | Dislikes: {row['dislikes']}</p>
            <p>Similarity: {row['similarity']:.2f}</p>
        </div>
        '''
    html += '</div>'
    display(HTML(html))
title_to_recommend = input("Enter a YouTube video title: ")
recommendations = recommend_videos(title_to_recommend, df_yt, tokenizer, model, top_n=5)
display_recommendations(recommendations)

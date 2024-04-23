import json
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_combined_embedding(chunk):
    combined_text = chunk["chunk_title"] + " " + chunk["chunk_content"]
    combined_embedding = model.encode([combined_text])[0]
    return combined_embedding

def visualize_embeddings(qualified_chunks, sent_to_llm, used_for_generating_answers):
    all_chunks = sent_to_llm + used_for_generating_answers + qualified_chunks

    # Generate embeddings using SentenceTransformer
    embeddings = []
    for chunk in all_chunks:
        combined_embedding = generate_combined_embedding(chunk)
        embeddings.append({"chunk_title": chunk["chunk_title"], "chunk_content": chunk["chunk_content"], "combined_embedding": combined_embedding, "label": chunk["label"]})

    df = pd.DataFrame(embeddings)

    # Apply PCA for dimensionality reduction to 3 dimensions
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(df['combined_embedding'].tolist())

    df['x'] = embeddings_3d[:, 0]
    df['y'] = embeddings_3d[:, 1]
    df['z'] = embeddings_3d[:, 2]

    # Define colors for each label
    colors = {'Qualified Chunks': 'red', 'Sent to LLM': 'blue', 'Used in Answers': 'green'}

    # Create the 3D scatter plot using Plotly
    fig = go.Figure()
    for label, color in colors.items():
        data = df[df['label'] == label]
        fig.add_trace(go.Scatter3d(
            x=data['x'],
            y=data['y'],
            z=data['z'],
            mode='markers',
            marker=dict(
                size=8,
                color=color,
                opacity=0.7,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            name=label,
            text=data['chunk_title']
        ))

    # Update layout
    fig.update_layout(
        title='Word Embeddings Visualization',
        legend=dict(
            title='Legend',
            itemsizing='constant'
        )
    )

    # Add hover info
    fig.update_traces(hoverinfo='text+name')

    # Save the plot as an HTML file
    fig.write_html("3d_plot.html")

def main(input_file):
    # Load JSON data
    data = load_json(input_file)

    # Extract chunks
    qualified_chunks = data["generative_answers"]["qualified_chunks"]["chunks"]
    sent_to_llm = data["sent_to_llm"]
    used_for_generating_answers = data["used_for_generating_answers"]

    # Load SentenceTransformer model
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-cos-v1")

    # Visualize embeddings
    visualize_embeddings(qualified_chunks, sent_to_llm, used_for_generating_answers)

if __name__ == "__main__":
    input_file = 'chunks.json'  # Change this to your input file path
    main(input_file)

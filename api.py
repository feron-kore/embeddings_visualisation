from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

def generate_combined_embedding(chunk, model):
    combined_text = chunk["chunk_text"]
    combined_embedding = model.encode([combined_text])[0]
    return combined_embedding

def visualize_embeddings(chunks):
    # Load SentenceTransformer model
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-cos-v1")
    
    # Generate embeddings
    embeddings = []
    for chunk in chunks:
        combined_embedding = generate_combined_embedding(chunk, model)
        embeddings.append({
            "chunk_text": chunk["chunk_text"], 
            "combined_embedding": combined_embedding,
            "category": chunk["category"]
        })

    df = pd.DataFrame(embeddings)

    # Apply PCA for dimensionality reduction to 3 dimensions
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(df['combined_embedding'].tolist())

    df['x'] = embeddings_3d[:, 0]
    df['y'] = embeddings_3d[:, 1]
    df['z'] = embeddings_3d[:, 2]

    # Define colors for each category
    colors = {'Qualified Chunks': 'red', 'Sent to LLM': 'blue', 'Used in Answers': 'green'}

    # Create the 3D scatter plot using Plotly
    fig = go.Figure()
    for category, color in colors.items():
        data = df[df['category'] == category]
        fig.add_trace(go.Scatter3d(
            x=data['x'],
            y=data['y'],
            z=data['z'],
            mode='markers',
            marker=dict(
                size=8,
                opacity=0.7,
                color=color
            ),
            name=category
        ))

    # Update layout
    fig.update_layout(
        title='Chunk Embeddings Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    # Save the plot as an HTML string
    html_output = fig.to_html(full_html=False)
    return html_output

@app.route('/visualize', methods=['POST'])
def visualize():
    data = None
    if 'file' in request.files and request.files['file']:
        file = request.files['file']
        if file.filename:
            data = json.load(file)
    elif request.is_json:
        data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No valid JSON data provided'})

    # Extract and categorize chunks
    extracted_chunks = extract_and_categorize(data)

    # Visualize embeddings
    html_output = visualize_embeddings(extracted_chunks)
    return html_output

def extract_and_categorize(data):
    qualified_chunks = data["generative_answers"]["qualified_chunks"]["chunks"]
    extracted_chunks = []
    for chunk in qualified_chunks:
        category = "Qualified Chunks"
        if chunk["sent_to_LLM"]:
            category = "Sent to LLM" if not chunk["used_in_answer"] else "Used in Answers"
        extracted_chunk = {
            "chunk_text": chunk["chunk_text"],
            "source_name": chunk["source_name"],
            "category": category  # Adding category here
        }
        extracted_chunks.append(extracted_chunk)
    return extracted_chunks

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Change port to 5000

import json

# Load the JSON data from the file with explicit encoding
with open('chunks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract "qualified_chunks"
qualified_chunks = data["generative_answers"]["qualified_chunks"]["chunks"]

# Extract required fields from each chunk
extracted_chunks = []
for chunk in qualified_chunks:
    extracted_chunk = {
        "chunk_text": chunk["chunk_text"],
        "source_name": chunk["source_name"],
        "sent_to_llm": chunk["sent_to_LLM"],
        "used_in_answer": chunk["used_in_answer"]
    }
    extracted_chunks.append(extracted_chunk)

# Write the extracted chunks to another JSON file
output_file = 'extracted_chunks.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_chunks, f, indent=4)

print(f"Extracted chunks saved to {output_file}")

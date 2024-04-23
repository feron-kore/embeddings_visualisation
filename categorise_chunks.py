import json

# Load the extracted chunks from the JSON file
with open('extracted_chunks.json', 'r', encoding='utf-8') as f:
    extracted_chunks = json.load(f)

# Categorize the extracted content
used_for_generating_answers = []
sent_to_llm = []
qualified_chunks = []

for chunk in extracted_chunks:
    if chunk["sent_to_llm"] and chunk["used_in_answer"]:
        used_for_generating_answers.append({
            "chunk_title": chunk["source_name"],
            "chunk_content": chunk["chunk_text"]
        })
    elif chunk["sent_to_llm"] and not chunk["used_in_answer"]:
        sent_to_llm.append({
            "chunk_title": chunk["source_name"],
            "chunk_content": chunk["chunk_text"]
        })
    else:
        qualified_chunks.append({
            "chunk_title": chunk["source_name"],
            "chunk_content": chunk["chunk_text"]
        })

# Write the categorized content to a new JSON file
categorized_content = {
    "used_for_generating_answers": used_for_generating_answers,
    "sent_to_llm": sent_to_llm,
    "qualified_chunks": qualified_chunks
}

# Save the categorized content to a JSON file
with open('categorized_chunks.json', 'w') as outfile:
    json.dump(categorized_content, outfile, indent=4)

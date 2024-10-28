from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")

def chunk_text(text, max_length):
    """Divide text into chunks of max_length tokens."""
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)[0]
    chunks = []
    
    for i in range(0, len(input_ids), max_length):
        chunks.append(input_ids[i:i + max_length])
    
    return chunks

def summarize(text):
    # Set a max length for chunks (e.g., 512 tokens)
    max_chunk_length = 512
    
    # Calculate min and max lengths based on input text length
    input_length = len(text.split())
    min_length = max(30, input_length // 3)  # Ensures min_length is at least 30
    max_length = min(200, (input_length * 2) // 3)  # Cap max_length at 200

    # Chunk the input text
    chunks = chunk_text(text, max_chunk_length)

    summaries = []
    for chunk in chunks:
        # Generate the summary for each chunk
        generated_ids = model.generate(
            input_ids=chunk.unsqueeze(0),  # Add batch dimension
            num_beams=2,
            min_length=min_length,
            max_length=max_length,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        
        # Decode the generated text
        chunk_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        summaries.append(chunk_summary)

    # Combine the summaries from all chunks
    final_summary = " ".join(summaries)
    return final_summary
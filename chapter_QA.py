from transformers import AutoTokenizer, AutoModelWithLMHead
import numpy as np

# Load model and tokenizer
model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)

def chunk_context(context, max_length=512):
    """
    Breaks the context into smaller chunks if it exceeds the maximum length.

    Args:
        context (str): The full context string.
        max_length (int): The maximum length of each chunk.

    Returns:
        list: A list of context chunks.
    """
    words = context.split()
    chunks = []
    current_chunk = []

    for word in words:
        # If adding the next word exceeds the max_length, store the current chunk and reset
        if len(tokenizer(' '.join(current_chunk + [word]), return_tensors='pt')['input_ids'][0]) > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]  # Start a new chunk with the current word
        else:
            current_chunk.append(word)

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def generate_answer(question, context):
    """
    Generates an answer based on the input question and context.

    Args:
        question (str): The question to be answered.
        context (str): The context in which to find the answer.

    Returns:
        str: The best generated answer from the context.
    """
    # Prepare the input prompt for the model
    input_text = f"question: {question} context: {context}"
    
    # Check if the context exceeds model's limit
    if len(tokenizer(input_text, return_tensors='pt')['input_ids'][0]) > 512:
        # Chunk the context
        context_chunks = chunk_context(context)
        answers = []

        for chunk in context_chunks:
            input_text_chunk = f"question: {question} context: {chunk}"
            encoded_input = tokenizer([input_text_chunk], return_tensors='pt', max_length=512, truncation=True)
            output = model.generate(input_ids=encoded_input.input_ids, attention_mask=encoded_input.attention_mask)
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
            answers.append(answer)

        # Choose the best answer (you can customize the selection strategy)
        best_answer = max(answers, key=len)  # Example strategy: choose the longest answer
    else:
        encoded_input = tokenizer([input_text], return_tensors='pt', max_length=512, truncation=True)
        output = model.generate(input_ids=encoded_input.input_ids, attention_mask=encoded_input.attention_mask)
        best_answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return best_answer

# Example usage
question = input("enter your question")
context = input("enter your chapter")
answer = generate_answer(question, context)
print("Answer:", answer)
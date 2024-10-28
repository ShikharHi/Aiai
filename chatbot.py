from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Arjun-G-Ravi/chat-GPT2 model and tokenizer
model_name = "Arjun-G-Ravi/chat-GPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Chatbot function
def chatbot_response(user_input, chat_history_ids=None):
    # Encode the user input and add it to the conversation history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Generate the model response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the model's response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Main chat loop
print("AI Assistant: Hello! How can I help you today? (Type 'quit' to stop)")
chat_history_ids = None

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("AI Assistant: Goodbye!")
        break

    response, chat_history_ids = chatbot_response(user_input, chat_history_ids)
    print("AI Assistant:", response)

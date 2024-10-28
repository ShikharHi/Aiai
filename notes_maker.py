from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the pre-trained T5 model and tokenizer
model_name = "t5-base"  # You can use other variants like 't5-small' or 't5-large'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate point-wise notes
def generate_pointwise_notes(chapter_text):
    # Prepare the input by prefixing with a specific prompt for bullet points
    input_text = "generate bullet points: " + chapter_text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate output
    output_ids = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)

    # Decode the generated output
    notes = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Optional: Split the output into bullet points if necessary
    bullet_points = notes.split('. ')
    return bullet_points

# Example chapter text
chapter_text = input("input")

# Generate point-wise notes
notes = generate_pointwise_notes(chapter_text)
print("Generated Point-wise Notes:")
for point in notes:
    print(f"- {point}")
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox
import speech_recognition as sr
from summarizer import summarize
from chapter_QA import generate_answer
from notes_maker import generate_pointwise_notes
from resources_for_study import fetch_wikipedia_links, youtube_search
from chatbot import tokenizer, model
import torch


def handle_chat(user_input, chat_history_ids=None):
    
    if "summarize" in user_input.lower():
        text = simpledialog.askstring("Input", "Please provide the text you want to summarize:")
        if text:
            summary = summarize(text)
            messagebox.showinfo("Summary", summary)
    
    elif "question" in user_input.lower() or "answer" in user_input.lower():
        context = simpledialog.askstring("Input", "Please provide the context:")
        question = simpledialog.askstring("Input", "Please provide your question:")
        if context and question:
            answer = generate_answer(question, context)
            messagebox.showinfo("Answer", answer)
    
    elif "notes" in user_input.lower() or "bullet points" in user_input.lower():
        chapter_text = simpledialog.askstring("Input", "Please provide the chapter text:")
        if chapter_text:
            notes = generate_pointwise_notes(chapter_text)
            messagebox.showinfo("Generated Notes", "\n".join(f"- {point}" for point in notes))
    
    elif "resources" in user_input.lower() or "links" in user_input.lower():
        topic = simpledialog.askstring("Input", "Please provide the study topic:")
        if topic:
            wikipedia_links = fetch_wikipedia_links(topic)
            youtube_links = youtube_search(topic)
            resources = "\n".join(wikipedia_links + youtube_links)
            messagebox.showinfo("Suggested Resources", resources)

    else:
        
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response, chat_history_ids
    
    return None, chat_history_ids


def run_chatbot():
    chat_history_ids = None
    while True:
        user_input = input("You (chatbot): ")
        if user_input.lower() == "quit":
            break
        
        response, chat_history_ids = handle_chat(user_input, chat_history_ids)
        if response:
            print("Chatbot:", response)


def activate_assistant():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for activation phrase...")
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print("You said:", command)
        if "assistant" in command.lower():  
            user_input = command.replace("assistant", "").strip()
            if user_input:
                handle_chat(user_input)
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")

def create_gui():
    root = tk.Tk()
    root.title("AI Assistant")

    activate_button = tk.Button(root, text="Activate Assistant (Voice)", command=activate_assistant)
    activate_button.pack(pady=20)
    
    threading.Thread(target=run_chatbot, daemon=True).start()

    root.mainloop()

if __name__ == "__main__":
    create_gui()

import tkinter as tk
from tkinter import messagebox
from transformers import pipeline

model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipe = pipeline("sentiment-analysis", model=model_id)

def analyze_sentiment():
    text = entry.get()
    if text.strip() == "":
        messagebox.showinfo("Error", "Please enter some text.")
        return

    sentiment_results = sentiment_pipe(text)
    processed_sentiment = process_sentiment_results(sentiment_results)
    display_result(processed_sentiment)

def process_sentiment_results(sentiment_results):
    processed_sentiment = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
    
    for result in sentiment_results:
        label = result['label'].lower()
        score = round(result['score'] * 100, 2)  # Convert to percentage and round to 2 decimal places
        
        if score <= 50:
            processed_sentiment['negative'] = score
            processed_sentiment['neutral'] = 100.0 - score
        else:
            processed_sentiment['positive'] = score
            processed_sentiment['negative'] = 100.0 - score
            
    return processed_sentiment

def display_result(sentiment_data):
    positive_label.config(text=f"Positive: {sentiment_data['positive']}%")
    negative_label.config(text=f"Negative: {sentiment_data['negative']}%")
    neutral_label.config(text=f"Neutral: {sentiment_data['neutral']}%")

def clear_text():
    entry.delete(0, tk.END)

# Create the main GUI window
root = tk.Tk()
root.title("Sentiment Analysis Tool")

# Create and place GUI widgets
label = tk.Label(root, text="Enter your sentence:")
label.pack()

entry = tk.Entry(root, width=50)
entry.pack()

analyze_button = tk.Button(root, text="Analyze Sentiment", command=analyze_sentiment)
analyze_button.pack()

result_frame = tk.Frame(root)
result_frame.pack()

positive_label = tk.Label(result_frame, text="Positive: 0.0%")
positive_label.pack(side=tk.LEFT, padx=5)

negative_label = tk.Label(result_frame, text="Negative: 0.0%")
negative_label.pack(side=tk.LEFT, padx=5)

neutral_label = tk.Label(result_frame, text="Neutral: 0.0%")
neutral_label.pack(side=tk.LEFT, padx=5)

clear_button = tk.Button(root, text="Clear", command=clear_text)
clear_button.pack()

# Start the main event loop
root.mainloop()
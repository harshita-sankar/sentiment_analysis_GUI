from transformers import pipeline


model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_pipe = pipeline("sentiment-analysis", model=model_id)
data = ['I dont know how i feel about you', 'i hate you', 'i speak Japanese']
#sentiment_dict = sentiment_pipe(data)

# Function to process sentiment results and create the new dictionary for each text
def process_sentiment_results(sentiment_results):
    processed_sentiments = []
    
    for result in sentiment_results:
        label = result['label'].lower()
        score = round(result['score'] * 100, 2)
        
        new_dict = {'negative': 0.0, 'positive': 0.0, 'neutral': 0.0}
        
        if score <= 65:
            if label == 'negative':
                new_dict['negative'] = score 
                new_dict['neutral'] = (100 - score) 
            elif label == 'positive':
                new_dict['positive'] = score 
                new_dict['neutral'] = (100 - score) 
        else:
            if label == 'negative':
                new_dict['negative'] = score
                new_dict['positive'] = (100 - score) 
            elif label == 'positive':
                new_dict['positive'] = score
                new_dict['negative'] = (100 - score) 

            elif label == "neutral":
                new_dict['neutral'] = 100.00
                
        processed_sentiments.append(new_dict)
    
    return processed_sentiments

# Analyze the sentiments of the texts in the data
sentiment_results = sentiment_pipe(data)

# Create a list of dictionaries with processed sentiment results for each text
processed_sentiments_list = process_sentiment_results(sentiment_results)

# Print the processed sentiment results for each text
for text, processed_sentiments in zip(data, processed_sentiments_list):
    print(f'Text: {text}')
    print(processed_sentiments)
    print()
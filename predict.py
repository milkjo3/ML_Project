from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

# 1. Load the trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./results/checkpoint-344')  # Load from saved directory
tokenizer = DistilBertTokenizerFast.from_pretrained('./results/checkpoint-344')  # Load from saved directory

# 2. Define the function to make predictions
def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)

    # Make prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
    
    # Get prediction logits and apply softmax to get probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)

    # Get the predicted class (0 or 1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()  # .item() converts tensor to Python integer

    return predicted_class, probabilities

while True:
    # Replace with your input text
    text = input("Enter a piece of text [1 to exit]: ")
    if text == '1': break
    predicted_class, probabilities = predict(text)
    
    # 4. Display the results
    print(f"\nPredicted class: {predicted_class}\nProbablity matrix: {probabilities}")  # 0 or 1


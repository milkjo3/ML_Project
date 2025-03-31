import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

# 1. Load the dataset (example assumes a CSV with 'text' and 'label' columns)
df = pd.read_csv('datasets/HateSpeechDatasetBalanced.csv') # Replace with your filename/path
#print(df.head())
df = df[['Content', 'Label']].dropna()
# Ensure Label column has no NaN values
df = df.dropna(subset=['Label'])
# Randomly select 50k samples
df = df.sample(n=10000, random_state=42)  

# Get unique labels and ensure they are standard Python types
unique_labels = sorted(df['Label'].unique().tolist())  # Ensure list format

# Convert NumPy int64 to standard Python int
label2id = {str(label): int(i) for i, label in enumerate(unique_labels)}
id2label = {int(i): str(label) for i, label in enumerate(unique_labels)}

# Convert df['Label'] using label2id mapping
df['Label'] = df['Label'].astype(str).map(label2id)

# 3. Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Content'],
    df['Label'],
    test_size=0.45,
    random_state=42,
    stratify=df['Label']
)

# 4. Convert to Hugging Face Datasets
train_data = Dataset.from_dict({'content': train_texts, 'label':
train_labels})
val_data = Dataset.from_dict({'content': val_texts, 'label': val_labels})

# 5. Tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
def tokenize_function(example):
    return tokenizer(example['content'], truncation=True, padding='max_length', max_length=128)
train_dataset = train_data.map(tokenize_function, batched=True)
val_dataset = val_data.map(tokenize_function, batched=True)
# 6. Set format for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask',
'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask',
'label'])
# 7. Load Model
model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)
# 8. Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1, # Adjust
    per_device_train_batch_size=16, # Adjust based on GPU memory
    per_device_eval_batch_size=32,
    eval_strategy='steps',
    save_strategy='steps',
    eval_steps=1000,  # Evaluate every 1000 steps
    save_steps=1000,  # Save every 1000 steps
    logging_dir='./logs',
    logging_steps=500,
    load_best_model_at_end=True
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 10. Train
trainer.train()

output_dir = training_args.output_dir  # This is where the model is saved by Trainer
model.save_pretrained(output_dir)  # Save the model
tokenizer.save_pretrained(output_dir)  # Save the tokenizer

# 11. Evaluate
eval_results = trainer.evaluate()
print("Evaluation:", eval_results)

# 2025 ML Project: The Bully Metre

This project trains a **DistilBERT** model for binary classification to detect hate speech from text. The model is trained on a curated dataset from https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset?resource=download&select=HateSpeechDatasetBalanced.csv. Our model is a binary classifier to determine whether a given text can be classified as cyberbulling.
---

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Training Instructions](#training-instructions)
- [Evaluation Instructions](#evaluation-instructions)
- [Inference Instructions](#inference-instructions)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project uses a **DistilBERT** model to detect hate speech in text data. The dataset is pre-processed and split into training and validation sets. The model is trained for 3 epochs and evaluated using accuracy and F1 score metrics.

---

## Technologies Used

- **Programming Languages**: Python
- **Libraries**: TensorFlow, PyTorch, Hugging Face Transformers, Scikit-learn
- **Tools**: Git
- **Dataset**: Hate Speech Dataset (can be downloaded from [[URL](https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset?resource=download&select=HateSpeechDatasetBalanced.csv)])

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/milkjo3/ML_Project
   cd hate-speech-detection
2. Create a virtual enviornment with python version 3.10 - 3.11 and activate it:
    ```
    python -m venv myvenv
    source myvenv/bin/activate # Linux/MacOS users
    myvenv\Scripts\activate # Windows Users
    ```
3. Install Dependencies
    ```
    pip install -r requirements.txt
    ```
4. Gather Dataset from [Hate Speech Detection curated Dataset](https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset?resource=download&select=HateSpeechDatasetBalanced.csv) and place it into a folder in the main directory titled "datasets"

---
## Training

1. Start by looking at the train.py script. Be sure to look at the training arguments and other adjustments:
    ```
    # Randomly select 50k samples
    df = df.sample(n=10000, random_state=42)

    # Training and Test Splits
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Content'],
        df['Label'],
        test_size=0.45,
        random_state=42,
        stratify=df['Label']
    )

    # Training Arguments. Determines number of epochs, evaluations, output directory, etc.

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


    ```

    This is very important as this dataset is very large in size. Adjust these arguments/parameters as needed to train the model. In my sample run, I only trained my model on 50k samples out of 720k samples on one epoch. 

2. Run the train.py script
    ```
    python train.py
    ```
3. Results and best model will be saved in a folder named 
    ```
    ./results
    ```
    Note however when we save the token tokenizer, it will reside outside of the directory where the models are saved. Just move it into the latest directory with the newest model.

---
## Inference Instructions (Making Predictions)

1. Start by opening the predict.py script. And view the following lines:
    ```
    # 1. Load the trained model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained('./results/checkpoint-344')  # Load from saved directory

    tokenizer = DistilBertTokenizerFast.from_pretrained('./results/checkpoint-344')  # Load from saved directory
    ```

    Make sure this directory contains the tokenizer and trained model from above. In my case, the tokenizer and model resided in ./results/checkpoint-344.

2. Run the python script by:
    ```
    python predict.py
    ```
3. It will ask you for an input:
    ```
    Enter a piece of text [1 to exit]:
    ```
    Enter some text. It will return a number and a probablity matrix. A zero indicates the text is not cyberbullying and a one indicates cyberbullying. Heres an example of the two scenarios:
    
    **Cyberbulling:**
    ```
    Enter a piece of text [1 to exit]: I hate you loser. Do not ever talk to me. Go kick rocks or something punk.

    Predicted class: 1
    Probablity matrix: tensor([[0.1257, 0.8743]])
    ```

    **Not Cyberbullying:**
    ```
    Enter a piece of text [1 to exit]: Hows it going Dave? It's so good to see you! I am really happy to hear that you got a raise, you deserve it!

    Predicted class: 0
    Probablity matrix: tensor([[0.8818, 0.1182]])
    ```

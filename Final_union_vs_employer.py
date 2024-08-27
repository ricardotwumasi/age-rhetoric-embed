
pip install datasets

import os
import re
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM, DataCollatorForLanguageModeling
from datasets import Dataset
import matplotlib.pyplot as plt
import torch
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import ttest_ind
import os
import re
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

"""# **Fine-Tune on Union Transcripts**"""

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Define the output directory for the fine-tuned model
model_output_dir = '/content/drive/MyDrive/Colab Notebooks/ft_model_union'

# File path
union_file_path = '/content/drive/MyDrive/Colab Notebooks/no_timecode_union_running_text.txt'

# Check and load data
if not os.path.exists(union_file_path):
    raise FileNotFoundError(f"ERROR: The union data file was not found at: {union_file_path}")

with open(union_file_path, "r") as f:
    union_data = f.read()

# Data preparation and cleaning
data = []
for line in union_data.splitlines():  # Split into lines for better handling
    if (
        not line.startswith("[")
        and not line.startswith("\"[")
        and not line.isspace()
    ):
        data.append(re.sub(r"\s+", " ", line))  # Replace multiple spaces with a single space

# Tokenize data
tokenized_data = tokenizer(
    data,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=tokenizer.model_max_length
)

# Create dataset
dataset = Dataset.from_dict({
    'input_ids': tokenized_data['input_ids'],
    'attention_mask': tokenized_data['attention_mask']
})

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import TrainingArguments, Trainer

def adjust_length_to_model(length, max_sequence_length):
    if length > max_sequence_length:
        return max_sequence_length
    return length

# Get the length of the dataset
length = len(dataset)

# Load the model
model_ft = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased', output_hidden_states=True)

# Adjust the length to the model's maximum sequence length
length = adjust_length_to_model(length, max_sequence_length=model_ft.config.max_position_embeddings)

# Set the learning rate
learning_rate=2e-5

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=30,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=128,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,                # log & save weights each logging_steps
    learning_rate=learning_rate,     # learning rate
    save_steps=10_000,               # number of updates steps before two checkpoint saves
    save_total_limit=2,              # limit the total amount of checkpoints. Deletes the older checkpoints.
    remove_unused_columns=False,     # keep all columns in the dataset
)

# Initialize the Trainer
trainer = Trainer(
    model=model_ft,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Start the training
trainer.train()

# Save the fine-tuned model to the desired location
trainer.save_model(model_output_dir)

from google.colab import drive
drive.mount('/content/drive')

"""# **Fine-Tune on Employer Transcripts**"""

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Define the output directory for the fine-tuned model
model_output_dir = '/content/drive/MyDrive/Colab Notebooks/ft_model_employer'

# File path
employer_file_path = '/content/drive/MyDrive/Colab Notebooks/no_timecode_employer_running_text.txt'

# Check and load data
if not os.path.exists(union_file_path):
    raise FileNotFoundError(f"ERROR: The union data file was not found at: {union_file_path}")

with open(union_file_path, "r") as f:
    union_data = f.read()

# Data preparation and cleaning
data = []
for line in union_data.splitlines():  # Split into lines for better handling
    if (
        not line.startswith("[")
        and not line.startswith("\"[")
        and not line.isspace()
    ):
        data.append(re.sub(r"\s+", " ", line))  # Replace multiple spaces with a single space

# Tokenize data
tokenized_data = tokenizer(
    data,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=tokenizer.model_max_length
)

# Create dataset
dataset = Dataset.from_dict({
    'input_ids': tokenized_data['input_ids'],
    'attention_mask': tokenized_data['attention_mask']
})

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import TrainingArguments, Trainer

def adjust_length_to_model(length, max_sequence_length):
    if length > max_sequence_length:
        return max_sequence_length
    return length

# Get the length of the dataset
length = len(dataset)

# Load the model
model_ft = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased', output_hidden_states=True)

# Adjust the length to the model's maximum sequence length
length = adjust_length_to_model(length, max_sequence_length=model_ft.config.max_position_embeddings)

# Set the learning rate
learning_rate=2e-5

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=30,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=128,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,                # log & save weights each logging_steps
    learning_rate=learning_rate,     # learning rate
    save_steps=10_000,               # number of updates steps before two checkpoint saves
    save_total_limit=2,              # limit the total amount of checkpoints. Deletes the older checkpoints.
    remove_unused_columns=False,     # keep all columns in the dataset
)

# Initialize the Trainer
trainer = Trainer(
    model=model_ft,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Start the training
trainer.train()

# Save the fine-tuned model to the desired location
trainer.save_model(model_output_dir)



"""# **Compare**"""

import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import ttest_ind
import random

# Load fine-tuned models
model_path_union = '/content/drive/MyDrive/Colab Notebooks/ft_model_union'
model_path_employer = '/content/drive/MyDrive/Colab Notebooks/ft_model_employer'
tokenizer_union = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model_union = DistilBertModel.from_pretrained(model_path_union)
tokenizer_employer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model_employer = DistilBertModel.from_pretrained(model_path_employer)

# Load pre-trained DistilBERT model and tokenizer
tokenizer_pretrained = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model_pretrained = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_union.to(device)
model_employer.to(device)
model_pretrained.to(device)

# Define terms for young and old
young_terms = ['young', 'youth', 'teenager', 'adolescent', 'juvenile']
old_terms = ['old', 'elderly', 'senior', 'aged', 'mature']

# Define target word sets
attribute_sets = {
    'Physical Descriptors (Good vs. Bad)': (
        ['beautiful', 'pretty', 'handsome', 'attractive', 'cute', 'fit', 'slim', 'muscular', 'glowing', 'vibrant', 'tall', 'strong'],
        ['wrinkled', 'unattractive', 'not neat', 'less attractive', 'worse-looking', 'ugly', 'fat', 'scrawny', 'bald', 'short', 'weak', 'shaky', 'fragile', 'poor posture', 'slow']
    ),
    'Personality Descriptors (Good vs. Bad)': (
        ['enthusiastic', 'open-minded', 'adaptable', 'kind', 'friendly', 'calm', 'adventurous', 'warm', 'good-natured', 'benevolent', 'amicable', 'conscientious', 'self-discipline','loyal', 'trustworthy', 'stable', 'committed', 'reliable', 'adapt to change'],
        ['rigid', 'stubborn', 'mean', 'aggressive', 'anxious', 'cautious', 'grumpy', 'cranky', 'irritable', 'old-fashioned', 'resistant to change', 'dejected', 'hopeless', 'unhappy', 'lonely', 'insecure', 'complains a lot', 'grouchy', 'critical', 'miserly', 'unpleasant', 'ill-tempered', 'bitter', 'demanding', 'complaining', 'annoying', 'humorless', 'selfish', 'prejudiced', 'suspicious of strangers', 'unfriendliness', 'uncreative', 'lower creativity']
    ),
    'Competence Descriptors (Good vs. Bad)': (
        ['intelligent','useful','high performance', 'competent', 'capable', 'efficient', 'skilled', 'tech-savvy', 'innovative', 'quick learner', 'sharp mind', 'good memory', 'remembers details', 'quick recall', 'retains information', 'occupationally flexible', 'more flexibility','experienced','more experience','strong work ethic','working hard'],
        ['incompetent', 'incapable', 'inefficient', 'low performance', 'unskilled', 'slow learner', 'forgetful', 'struggles to remember', 'memory lapses', 'poor recall', 'worse memory', 'thinks people speak too softly', 'thinks others speak too fast', 'often asks to repeat', 'less flexible in doing different tasks', 'less likely to grasp new ideas','unable to communicate']
    ),
    'Health Descriptors (Good vs. Bad)': (
        ['healthy', 'fit', 'vigorous', 'active', 'energetic', 'agile', 'strong', 'good hearing', 'physically fit'],
        ['unhealthy', 'infirm', 'weak', 'tired', 'sedentary', 'physically handicapped', 'sick', 'lower physical capacity', 'worse physical capability', 'less qualified for physical jobs', 'scared of becoming sick', 'lower activity', 'less energy', 'worse health', 'less speed', 'less physically active', 'moves slowly', 'worse psychomotor speed', 'hard of hearing']
    )
}

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Get BERT embedding for a word
def get_bert_embedding_for_word(model, tokenizer, word):
    encoded_word = tokenizer.encode(word, add_special_tokens=False, return_tensors='pt').to(device)
    with torch.no_grad():
        embedding = model(encoded_word)
    vector = embedding.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    return vector

# Calculate g function
def g(c, A, B, w):
    t1 = np.mean([cosine_similarity(w[c], w[a]) for a in A])
    t2 = np.mean([cosine_similarity(w[c], w[b]) for b in B])
    return t1 - t2

# Calculate WEAT score using cosine similarity
def weat_bert(model, tokenizer, S, T, A, B):
    w = {}
    for word_list in [S, T, A, B]:
        for word in word_list:
            if word not in w:
                w[word] = get_bert_embedding_for_word(model, tokenizer, word)

    S_T = np.union1d(S, T)

    t1 = np.mean([g(s, A, B, w) for s in S])
    t2 = np.mean([g(t, A, B, w) for t in T])
    t3 = np.std([g(c, A, B, w) for c in S_T])

    weat_score = (t1 - t2) / t3 if t3 != 0 else 0

    # Calculate p-value using permutation test
    STAB = h(S, T, A, B, w)
    prs = 0
    for s in S:
        for t in T:
            STAB_i = h_i(s, t, A, B, w)
            if STAB_i > STAB:
                prs += 1
    p_value = prs / (len(S) * len(T))

    return weat_score, p_value

# Helper functions for permutation test
def h(S, T, A, B, w):
    t1 = sum(g(s, A, B, w) for s in S)
    t2 = sum(g(t, A, B, w) for t in T)
    return t1 - t2

def h_i(s, t, A, B, w):
    return g(s, A, B, w) - g(t, A, B, w)


# Compute WEAT scores for each model
weat_results = []

for target_words, (target_X, target_Y) in attribute_sets.items():


    # Union model
    weat_score_union, p_value_union = weat_bert(model_union, tokenizer_union, young_terms, old_terms, target_X, target_Y)

    # Employer model
    weat_score_employer, p_value_employer = weat_bert(model_employer, tokenizer_employer, young_terms, old_terms, target_X, target_Y)

    # Pretrained model
    weat_score_pretrained, p_value_pretrained = weat_bert(model_pretrained, tokenizer_pretrained, young_terms, old_terms, target_X, target_Y)

    weat_results.append({
        'Target words': target_words,
        'Union WEAT Score (s)': weat_score_union,
        'Union WEAT P-value': p_value_union,
        'Employer WEAT Score (s)': weat_score_employer,
        'Employer WEAT P-value': p_value_employer,
        'Pretrained WEAT Score (s)': weat_score_pretrained,
        'Pretrained WEAT P-value': p_value_pretrained
    })


# Convert results to DataFrames
df_weat = pd.DataFrame(weat_results)
df_seat = pd.DataFrame(seat_results)

# Print the WEAT scores in tables
print("WEAT Scores:")
print(tabulate(df_weat, headers='keys', tablefmt='pipe', floatfmt=".4f"))

import pandas as pd
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Assuming df_weat and df_seat are already created as per your provided code

# Concatenate the DataFrames side by side
df_combined = pd.concat([df_weat, df_seat], axis=1)

# Authenticate and create a Sheets API client
auth.authenticate_user()
service = build('sheets', 'v4')

# Create a new spreadsheet
spreadsheet = {
    'properties': {
        'title': 'WEAT Results1'
    }
}
spreadsheet = service.spreadsheets().create(body=spreadsheet, fields='spreadsheetId').execute()
spreadsheet_id = spreadsheet.get('spreadsheetId')
print(f'Spreadsheet ID: {spreadsheet_id}')

# Specify the range in the sheet where you want to write the data
range_name = 'Sheet1!A1'

# Convert the DataFrame to a list of lists
values = df_combined.reset_index(drop=True).T.reset_index().T.values.tolist()

# Write the DataFrame to the sheet
try:
    body = {
        'values': values
    }
    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id, range=range_name,
        valueInputOption='RAW', body=body).execute()
    print(f"{result.get('updatedCells')} cells updated in the spreadsheet.")
except HttpError as error:
    print(f"An error occurred: {error}")

"""#Computing confidence intervals for WEAT scores"""

import numpy as np
from scipy import stats

def bootstrap_weat(X, Y, A, B, embeddings, num_bootstraps=10000):
    weat_scores = []
    for _ in range(num_bootstraps):
        X_sample = np.random.choice(X, size=len(X), replace=True)
        Y_sample = np.random.choice(Y, size=len(Y), replace=True)
        A_sample = np.random.choice(A, size=len(A), replace=True)
        B_sample = np.random.choice(B, size=len(B), replace=True)

        weat_score = compute_weat(X_sample, Y_sample, A_sample, B_sample, embeddings)
        weat_scores.append(weat_score)

    ci_lower, ci_upper = np.percentile(weat_scores, [2.5, 97.5])
    return np.mean(weat_scores), ci_lower, ci_upper

# Assuming compute_weat is a function that calculates the WEAT score
# X, Y are the target word sets, A, B are the attribute word sets
# embeddings is your word embedding model

weat_score, ci_lower, ci_upper = bootstrap_weat(X, Y, A, B, embeddings)
print(f"WEAT Score: {weat_score:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")



import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import ttest_ind
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import ttest_ind
import random

# Load your fine-tuned models (replace paths as needed)
model_path_union = '/content/drive/MyDrive/Colab Notebooks/ft_model_union'
model_path_employer = '/content/drive/MyDrive/Colab Notebooks/ft_model_employer'
tokenizer_union = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model_union = DistilBertModel.from_pretrained(model_path_union)
tokenizer_employer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model_employer = DistilBertModel.from_pretrained(model_path_employer)

# Load pre-trained DistilBERT model and tokenizer
tokenizer_pretrained = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model_pretrained = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_union.to(device)
model_employer.to(device)
model_pretrained.to(device)

# Define terms for young and old
young_terms = ['young', 'youth', 'teenager', 'adolescent', 'juvenile']
old_terms = ['old', 'elderly', 'senior', 'aged', 'mature']

# Define target word sets
attribute_sets = {
    'Physical Descriptors (Good vs. Bad)': (
        ['beautiful', 'pretty', 'handsome', 'attractive', 'cute', 'fit', 'slim', 'muscular', 'glowing', 'vibrant', 'tall', 'strong'],
        ['wrinkled', 'unattractive', 'not neat', 'less attractive', 'worse-looking', 'ugly', 'fat', 'scrawny', 'bald', 'short', 'weak', 'shaky', 'fragile', 'poor posture', 'slow']
    ),
    'Personality Descriptors (Good vs. Bad)': (
        ['enthusiastic', 'open-minded', 'adaptable', 'kind', 'friendly', 'calm', 'adventurous', 'warm', 'good-natured', 'benevolent', 'amicable', 'conscientious', 'self-discipline','loyal', 'trustworthy', 'stable', 'committed', 'reliable', 'adapt to change'],
        ['rigid', 'stubborn', 'mean', 'aggressive', 'anxious', 'cautious', 'grumpy', 'cranky', 'irritable', 'old-fashioned', 'resistant to change', 'dejected', 'hopeless', 'unhappy', 'lonely', 'insecure', 'complains a lot', 'grouchy', 'critical', 'miserly', 'unpleasant', 'ill-tempered', 'bitter', 'demanding', 'complaining', 'annoying', 'humorless', 'selfish', 'prejudiced', 'suspicious of strangers', 'unfriendliness', 'uncreative', 'lower creativity']
    ),
    'Competence Descriptors (Good vs. Bad)': (
        ['intelligent','useful','high performance', 'competent', 'capable', 'efficient', 'skilled', 'tech-savvy', 'innovative', 'quick learner', 'sharp mind', 'good memory', 'remembers details', 'quick recall', 'retains information', 'occupationally flexible', 'more flexibility','experienced','more experience','strong work ethic','working hard'],
        ['incompetent', 'incapable', 'inefficient', 'low performance', 'unskilled', 'slow learner', 'forgetful', 'struggles to remember', 'memory lapses', 'poor recall', 'worse memory', 'thinks people speak too softly', 'thinks others speak too fast', 'often asks to repeat', 'less flexible in doing different tasks', 'less likely to grasp new ideas','unable to communicate']
    ),
    'Health Descriptors (Good vs. Bad)': (
        ['healthy', 'fit', 'vigorous', 'active', 'energetic', 'agile', 'strong', 'good hearing', 'physically fit'],
        ['unhealthy', 'infirm', 'weak', 'tired', 'sedentary', 'physically handicapped', 'sick', 'lower physical capacity', 'worse physical capability', 'less qualified for physical jobs', 'scared of becoming sick', 'lower activity', 'less energy', 'worse health', 'less speed', 'less physically active', 'moves slowly', 'worse psychomotor speed', 'hard of hearing']
    )
}

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Get BERT embedding for a word
def get_bert_embedding_for_word(model, tokenizer, word):
    encoded_word = tokenizer.encode(word, add_special_tokens=False, return_tensors='pt').to(device)
    with torch.no_grad():
        embedding = model(encoded_word)
    vector = embedding.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    return vector

# Calculate g function
def g(c, A, B, w):
    t1 = np.mean([cosine_similarity(w[c], w[a]) for a in A])
    t2 = np.mean([cosine_similarity(w[c], w[b]) for b in B])
    return t1 - t2

# Calculate WEAT score using cosine similarity
def weat_bert(model, tokenizer, S, T, A, B):
    w = {}
    for word_list in [S, T, A, B]:
        for word in word_list:
            if word not in w:
                w[word] = get_bert_embedding_for_word(model, tokenizer, word)

    S_T = np.union1d(S, T)

    t1 = np.mean([g(s, A, B, w) for s in S])
    t2 = np.mean([g(t, A, B, w) for t in T])
    t3 = np.std([g(c, A, B, w) for c in S_T])

    weat_score = (t1 - t2) / t3 if t3 != 0 else 0

    # Calculate p-value using permutation test
    STAB = h(S, T, A, B, w)
    prs = 0
    for s in S:
        for t in T:
            STAB_i = h_i(s, t, A, B, w)
            if STAB_i > STAB:
                prs += 1
    p_value = prs / (len(S) * len(T))

    return weat_score, p_value

# Helper functions for permutation test
def h(S, T, A, B, w):
    t1 = sum(g(s, A, B, w) for s in S)
    t2 = sum(g(t, A, B, w) for t in T)
    return t1 - t2

def h_i(s, t, A, B, w):
    return g(s, A, B, w) - g(t, A, B, w)

# Compute WEAT scores for each model
weat_results = []
seat_results = []

for target_words, (target_X, target_Y) in attribute_sets.items():
    target_X_sentences = create_sentences(target_X)
    target_Y_sentences = create_sentences(target_Y)

    # Union model
    weat_score_union, p_value_union = weat_bert(model_union, tokenizer_union, old_terms, young_terms, target_X, target_Y)
    seat_score_union, seat_p_value_union = compute_seat(model_union, tokenizer_union, old_terms, young_terms, target_X_sentences, target_Y_sentences)

    # Employer model
    weat_score_employer, p_value_employer = weat_bert(model_employer, tokenizer_employer, old_terms, young_terms, target_X, target_Y)
    seat_score_employer, seat_p_value_employer = compute_seat(model_employer, tokenizer_employer, old_terms, young_terms, target_X_sentences, target_Y_sentences)

    # Pretrained model
    weat_score_pretrained, p_value_pretrained = weat_bert(model_pretrained, tokenizer_pretrained, old_terms, young_terms, target_X, target_Y)
    seat_score_pretrained, seat_p_value_pretrained = compute_seat(model_pretrained, tokenizer_pretrained, old_terms, young_terms, target_X_sentences, target_Y_sentences)

    weat_results.append({
        'Target words': target_words,
        'Union WEAT Score (s)': weat_score_union,
        'Union WEAT P-value': p_value_union,
        'Employer WEAT Score (s)': weat_score_employer,
        'Employer WEAT P-value': p_value_employer,
        'Pretrained WEAT Score (s)': weat_score_pretrained,
        'Pretrained WEAT P-value': p_value_pretrained
    })


# Convert results to DataFrames
df_weat = pd.DataFrame(weat_results)

# Print the WEAT scores in tables
print("WEAT Scores:")
print(tabulate(df_weat, headers='keys', tablefmt='pipe', floatfmt=".4f"))

import pandas as pd
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Authenticate and create a Sheets API client
auth.authenticate_user()
service = build('sheets', 'v4')

# Create a new spreadsheet
spreadsheet = {
    'properties': {
        'title': 'WEAT Results2'
    }
}
spreadsheet = service.spreadsheets().create(body=spreadsheet, fields='spreadsheetId').execute()
spreadsheet_id = spreadsheet.get('spreadsheetId')
print(f'Spreadsheet ID: {spreadsheet_id}')

# Specify the range in the sheet where you want to write the data
range_name = 'Sheet1!A1'  # Adjust the sheet name and range as needed

# Prepare the DataFrame for writing to the sheet
# Convert the DataFrame to a list of lists
values = df_weat.reset_index(drop=True).T.reset_index().T.values.tolist()

# Write the DataFrame to the sheet
try:
    body = {
        'values': values
    }
    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id, range=range_name,
        valueInputOption='RAW', body=body).execute()
    print(f"{result.get('updatedCells')} cells updated in the spreadsheet.")
except HttpError as error:
    print(f"An error occurred: {error}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df_weat is your DataFrame containing WEAT scores and p-values

# Melt the dataframe to have a long format suitable for seaborn
weat_long = df_weat.melt(id_vars=['Target words'],
                         value_vars=['Union WEAT Score (s)', 'Employer WEAT Score (s)', 'Pretrained WEAT Score (s)'],
                         var_name='Test Type and Corpus', value_name='Score')

# Extract p-values for significance annotation
p_values_long = df_weat.melt(id_vars=['Target words'],
                             value_vars=['Union WEAT P-value', 'Employer WEAT P-value', 'Pretrained WEAT P-value'],
                             var_name='P-value Type', value_name='P-value')

# Ensure the order of p-values matches the scores
weat_long['P-value'] = p_values_long['P-value']

# Map the test type to the color palette
palette = {
    'Union WEAT Score (s)': 'skyblue',
    'Employer WEAT Score (s)': 'steelblue',
    'Pretrained WEAT Score (s)': 'lightcoral'
}

# Plotting the WEAT scores
plt.figure(figsize=(14, 8))

# Create a bar plot
ax = sns.barplot(
    x='Score',
    y='Target words',
    hue='Test Type and Corpus',
    data=weat_long,
    palette=palette
)

# Customize the plot
plt.title('WEAT Scores: Young vs. Old (Union and Employer Corpora)', fontsize=16)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Target Words', fontsize=12)
plt.axvline(x=0, color='black', linestyle='--')
plt.legend(title='Test Type and Corpus', fontsize=10)

# Annotate the bars with their score values and asterisks for significance
for index, row in weat_long.iterrows():
    is_significant = row['P-value'] < 0.05
    annotation = f"{row['Score']:.4f}" + ("*" if is_significant else "")

    # Find the position of the bar
    bar_position = ax.patches[index].get_y() + ax.patches[index].get_height() / 2

    ax.annotate(
        annotation,
        (row['Score'], bar_position),
        va='center',
        ha='left' if row['Score'] >= 0 else 'right',
        fontsize=8
    )

plt.tight_layout()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df_weat is your DataFrame containing WEAT scores and p-values

# Melt the dataframe to have a long format suitable for seaborn
weat_long = df_weat.melt(id_vars=['Target words'],
                         value_vars=['Union WEAT Score (s)', 'Employer WEAT Score (s)', 'Pretrained WEAT Score (s)'],
                         var_name='Test Type and Corpus', value_name='Score')

# Extract p-values for significance annotation
p_values_long = df_weat.melt(id_vars=['Target words'],
                             value_vars=['Union WEAT P-value', 'Employer WEAT P-value', 'Pretrained WEAT P-value'],
                             var_name='P-value Type', value_name='P-value')

# Ensure the order of p-values matches the scores
weat_long['P-value'] = p_values_long['P-value']

# Map the test type to the color palette
palette = {
    'Union WEAT Score (s)': 'skyblue',
    'Employer WEAT Score (s)': 'steelblue',
    'Pretrained WEAT Score (s)': 'lightcoral'
}

# Plotting the WEAT scores
plt.figure(figsize=(14, 8))

# Create a bar plot
ax = sns.barplot(
    x='Score',
    y='Target words',
    hue='Test Type and Corpus',
    data=weat_long,
    palette=palette
)

# Customize the plot
plt.title('WEAT Scores: Young vs. Old (Union and Employer Corpora)', fontsize=16)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Target Words', fontsize=12)
plt.axvline(x=0, color='black', linestyle='--')
plt.legend(title='Test Type and Corpus', fontsize=10)

# Annotate the bars with their score values and asterisks for significance
for index, row in weat_long.iterrows():
    is_significant = row['P-value'] < 0.05
    annotation = f"{row['Score']:.4f}" + ("*" if is_significant else "")

    # Find the position of the bar
    bar_position = ax.patches[index].get_y() + ax.patches[index].get_height() / 2

    ax.annotate(
        annotation,
        (row['Score'], bar_position),
        va='center',
        ha='left' if row['Score'] >= 0 else 'right',
        fontsize=8
    )

plt.tight_layout()
plt.show()

"""#visualise emdeddings

"""

# Create TSV files
with open('vectors.tsv', 'w', newline='') as vectors_file, open('metadata.tsv', 'w', newline='') as metadata_file:
    vector_writer = csv.writer(vectors_file, delimiter='\t')
    metadata_writer = csv.writer(metadata_file, delimiter='\t')

    # Write header for metadata file
    metadata_writer.writerow(['Word'])

    for word in words:
        # Get embedding for the word
        embedding = get_word_embedding(word)

        # Write embedding to vectors file
        vector_writer.writerow(embedding)

        # Write word to metadata file
        metadata_writer.writerow([word])

print("TSV files created successfully.")

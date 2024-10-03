ort torch
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

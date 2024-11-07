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
young_terms = ['young', 'youth', 'teenager', 'adolescent', 'juvenile', 'junior']
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

# Print the WEAT scores in tables
print("WEAT Scores:")
print(tabulate(df_weat, headers='keys', tablefmt='pipe', floatfmt=".4f"))

# After calculating all WEAT scores, collect the g scores:
import numpy as np
from scipy.stats import ttest_ind
from tabulate import tabulate
import pandas as pd

def collect_g_scores_per_category(model, tokenizer, young_terms, old_terms, attribute_sets):
    """
    Calculates g scores for each word in young_terms and old_terms against each attribute set,
    organized by category.

    Args:
      model: BERT model.
      tokenizer: BERT tokenizer.
      young_terms: List of words representing the "young" concept.
      old_terms: List of words representing the "old" concept.
      attribute_sets: Dictionary of attribute sets, where keys are category names 
                      and values are tuples of target sets (target_X, target_Y).

    Returns:
      A dictionary where keys are category names and values are lists of g scores 
      for all words in that category.
    """
    all_g_scores = {}
    for category, (target_X, target_Y) in attribute_sets.items():
        # Get embeddings
        w = {}
        for word_list in [young_terms, old_terms, target_X, target_Y]:
            for word in word_list:
                if word not in w:
                    w[word] = get_bert_embedding_for_word(model, tokenizer, word)
        
        # Calculate g scores for all words in the current category
        category_scores = [g(word, target_X, target_Y, w) for word in young_terms + old_terms]
        all_g_scores[category] = category_scores
    
    return all_g_scores

# Collect g scores per category for each model
union_scores_per_category = collect_g_scores_per_category(model_union, tokenizer_union, young_terms, old_terms, attribute_sets)
employer_scores_per_category = collect_g_scores_per_category(model_employer, tokenizer_employer, young_terms, old_terms, attribute_sets)
pretrained_scores_per_category = collect_g_scores_per_category(model_pretrained, tokenizer_pretrained, young_terms, old_terms, attribute_sets)

# Perform t-tests and calculate Cohen's d for each category
def cohens_d(x1, x2):
    pooled_std = np.sqrt((np.var(x1) + np.var(x2)) / 2)
    return (np.mean(x1) - np.mean(x2)) / pooled_std

comparison_tables = {} 
for category in attribute_sets.keys():
    comparisons = pd.DataFrame({
        'Comparison': ['Union vs Employer', 'Union vs Pretrained', 'Employer vs Pretrained'],
        't_statistic': [
            ttest_ind(union_scores_per_category[category], employer_scores_per_category[category]).statistic,
            ttest_ind(union_scores_per_category[category], pretrained_scores_per_category[category]).statistic,
            ttest_ind(employer_scores_per_category[category], pretrained_scores_per_category[category]).statistic
        ],
        'p_value': [
            ttest_ind(union_scores_per_category[category], employer_scores_per_category[category]).pvalue,
            ttest_ind(union_scores_per_category[category], pretrained_scores_per_category[category]).pvalue,
            ttest_ind(employer_scores_per_category[category], pretrained_scores_per_category[category]).pvalue
        ],
        'cohens_d': [
            cohens_d(union_scores_per_category[category], employer_scores_per_category[category]),
            cohens_d(union_scores_per_category[category], pretrained_scores_per_category[category]),
            cohens_d(employer_scores_per_category[category], pretrained_scores_per_category[category])
        ]
    })
    comparison_tables[category] = comparisons

# Print comparison tables for each category
for category, table in comparison_tables.items():
    print(f"\nWEAT Score Comparisons Between Models for category: {category}")
    print(tabulate(table, headers='keys', tablefmt='pipe', floatfmt=".4f"))
print(tabulate(comparisons, headers='keys', tablefmt='pipe', floatfmt=".4f"))

def format_results_into_tables(results):
    # Table 1: Within-model comparisons (young vs old)
    within_rows = []
    for category, cat_results in results.items():
        row = {
            'Category': category,
            'Union_young': f"{cat_results['young_scores']['Union']['mean']:.3f} ± {cat_results['young_scores']['Union']['std']:.3f}",
            'Union_old': f"{cat_results['old_scores']['Union']['mean']:.3f} ± {cat_results['old_scores']['Union']['std']:.3f}",
            'Union_p': cat_results['within_model']['Union']['p_value'],
            'Union_d': cat_results['within_model']['Union']['cohens_d'],
            'Employer_young': f"{cat_results['young_scores']['Employer']['mean']:.3f} ± {cat_results['young_scores']['Employer']['std']:.3f}",
            'Employer_old': f"{cat_results['old_scores']['Employer']['mean']:.3f} ± {cat_results['old_scores']['Employer']['std']:.3f}",
            'Employer_p': cat_results['within_model']['Employer']['p_value'],
            'Employer_d': cat_results['within_model']['Employer']['cohens_d'],
            'Pretrained_young': f"{cat_results['young_scores']['Pretrained']['mean']:.3f} ± {cat_results['young_scores']['Pretrained']['std']:.3f}",
            'Pretrained_old': f"{cat_results['old_scores']['Pretrained']['mean']:.3f} ± {cat_results['old_scores']['Pretrained']['std']:.3f}",
            'Pretrained_p': cat_results['within_model']['Pretrained']['p_value'],
            'Pretrained_d': cat_results['within_model']['Pretrained']['cohens_d']
        }
        within_rows.append(row)

    # Table 2: Between-model comparisons
    between_rows = []
    for category, cat_results in results.items():
        # For young terms
        row_young = {
            'Category': f"{category} (Young Terms)",
            'Union_vs_Employer_p': cat_results['between_model_young']['Union_vs_Employer']['p_value'],
            'Union_vs_Employer_d': cat_results['between_model_young']['Union_vs_Employer']['cohens_d'],
            'Union_vs_Pretrained_p': cat_results['between_model_young']['Union_vs_Pretrained']['p_value'],
            'Union_vs_Pretrained_d': cat_results['between_model_young']['Union_vs_Pretrained']['cohens_d'],
            'Employer_vs_Pretrained_p': cat_results['between_model_young']['Employer_vs_Pretrained']['p_value'],
            'Employer_vs_Pretrained_d': cat_results['between_model_young']['Employer_vs_Pretrained']['cohens_d']
        }
        between_rows.append(row_young)
        
        # For old terms
        row_old = {
            'Category': f"{category} (Old Terms)",
            'Union_vs_Employer_p': cat_results['between_model_old']['Union_vs_Employer']['p_value'],
            'Union_vs_Employer_d': cat_results['between_model_old']['Union_vs_Employer']['cohens_d'],
            'Union_vs_Pretrained_p': cat_results['between_model_old']['Union_vs_Pretrained']['p_value'],
            'Union_vs_Pretrained_d': cat_results['between_model_old']['Union_vs_Pretrained']['cohens_d'],
            'Employer_vs_Pretrained_p': cat_results['between_model_old']['Employer_vs_Pretrained']['p_value'],
            'Employer_vs_Pretrained_d': cat_results['between_model_old']['Employer_vs_Pretrained']['cohens_d']
        }
        between_rows.append(row_old)

    df_within = pd.DataFrame(within_rows)
    df_between = pd.DataFrame(between_rows)

    print("\nTable 1: Within-Model Comparisons (Young vs Old):")
    print(tabulate(df_within, headers='keys', tablefmt='pipe', floatfmt=".4f"))
    
    print("\nTable 2: Between-Model Comparisons:")
    print(tabulate(df_between, headers='keys', tablefmt='pipe', floatfmt=".4f"))
    
    return df_within, df_between

# Run and display results
df_within, df_between = format_results_into_tables(results)

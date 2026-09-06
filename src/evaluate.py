import pandas as pd
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

df=pd.read_csv("results.csv")

openai_rouge1_scores = []
gemini_rouge1_scores = []
openai_rouge2_scores = []
gemini_rouge2_scores = []
openai_rougeL_scores = []
gemini_rougeL_scores = []

for _, row in df.iterrows():
    openai_score = scorer.score(row["Reference Answer"], row["OpenAI Answer"])
    gemini_score = scorer.score(row["Reference Answer"], row["Gemini Answer"])
    openai_rouge1_scores.append(openai_score['rouge1'].fmeasure)
    gemini_rouge1_scores.append(gemini_score['rouge1'].fmeasure)
    openai_rouge2_scores.append(openai_score['rouge2'].fmeasure)
    gemini_rouge2_scores.append(gemini_score['rouge2'].fmeasure)
    openai_rougeL_scores.append(openai_score['rougeL'].fmeasure)        
    gemini_rougeL_scores.append(gemini_score['rougeL'].fmeasure)

print(f" OpenAI average rouge1 score: {sum(openai_rouge1_scores)/len(openai_rouge1_scores)}")
print(f"gemini average rouge1 score: {sum(gemini_rouge1_scores)/len(gemini_rouge1_scores)}")
print(f" OpenAI average rouge2 score: {sum(openai_rouge2_scores)/len(openai_rouge2_scores)}")
print(f"gemini average rouge2 score: {sum(gemini_rouge2_scores)/len(gemini_rouge2_scores)}")
print(f" OpenAI average rougeL score: {sum(openai_rougeL_scores)/len(openai_rougeL_scores)}")
print(f"gemini average rougeL score: {sum(gemini_rougeL_scores)/len(gemini_rougeL_scores)}")
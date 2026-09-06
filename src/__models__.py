import pandas as pd
import openai
import google.generativeai as gemini
from __data_loader__ import load_data, get_sample
from dotenv import load_dotenv
import time
import os 

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
gemini.configure(api_key=os.getenv("GEMINAI_API_KEY"))


def openai_response(question: str, context: str, model: str = "gpt-4o-mini") -> str:

    """
    Get a response from OpenAI's GPT model based on the provided question and context.
        question (str): The question to ask the model.
        context (str): The context to provide to the model.
        model (str): The specific GPT model to use (default is "gpt-4-o-mini").
        str: The response from the model.
    """
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: if the context does not contain enough information to answer, respond with exactly: cannot determine from context period."
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "if you can't find the answer in the context, respond with exactly the word abstain."},
                  {"role": "user", "content": prompt}]  
    )
    return response.choices[0].message['content'].strip()

def gemini_response(question: str, context: str, model: str = "gemini-3.6-flash") -> str:
    """
    Get a response from Google's Gemini model based on the provided question and context.
        question (str): The question to ask the model.
        context (str): The context to provide to the model.
        model (str): The specific Gemini model to use (default is "gemini-3.6.-flash").
        str: The response from the model.
    """
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: if the context does not contain enough information to answer, respond with exactly: cannot determine from context period."
    model_instance = gemini.GenerativeModel(model)
    response = model_instance.generate_content(prompt)
    return response.text.strip()


if __name__ == "__main__":
    df = load_data("data/Financial-QA-10k.csv")
    sample_df = get_sample(df, 50)
    results = []
    for _, row in sample_df.iterrows():
        question = row["question"]
        context = row["context"]
        openai_answer = openai_response(question, context)
        gemini_answer = gemini_response(question, context)
        time.sleep(30)
        print(f"OpenAI Response: {openai_answer}")
        print(f"Gemini Response: {gemini_answer}")
        openai_abstained ="cannot determine"in openai_answer.lower() 
        gemini_abstained = "cannot determine" in gemini_answer.lower()
        results.append((question, row["answer"], context, openai_answer, gemini_answer, openai_abstained, gemini_abstained))
results_df = pd.DataFrame(results, columns=["Question", "Reference Answer", "Context", "OpenAI Answer", "Gemini Answer", "OpenAI Abstained", "Gemini Abstained"])
results_df.to_csv("results.csv", index=False)
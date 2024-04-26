# Databricks notebook source
# MAGIC %pip install beautifulsoup4 langchain openai pandas seaborn scikit-learn typing_extensions==4.3.0 PyPDF2 pycryptodome

# COMMAND ----------

import json
import os
import PyPDF2

# For cost-saving, create a cache for the LLM responses
import threading

# For data analysis and visualization
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd

# For scraping
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# COMMAND ----------

# MAGIC %md
# MAGIC Set up openai to use ChatGPT

# COMMAND ----------

openai.api_key = dbutils.secrets.get(scope="my_openai_secret_scope", key="openai_api_key")
os.environ["OPENAI_API_KEY"] = openai.api_key

# COMMAND ----------

# Choose a seed for reproducible results
SEED = 2024

# To avoid re-running the scraping process, choose a path to save the scrapped docs
SCRAPPED_DATA_PATH = "docs_scraped.csv"

# Choose a path to save the generated dataset
OUTPUT_DF_PATH = "question_answer_source.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC Set chunk size

# COMMAND ----------

CHUNK_SIZE = 1500

# COMMAND ----------

# MAGIC %md
# MAGIC Scrape documents from volume

# COMMAND ----------

import os
import PyPDF2
import pandas as pd

# Set the directory where the PDF documents are stored
pdfs_dir = "/Volumes/main/rag_chatbot_callum_elder/armhub_sample"

data = []
for filename in os.listdir(pdfs_dir):
    try:
        file_path = os.path.join(pdfs_dir, filename)
        with open(file_path, "rb") as file:
            # Use PyPDF2 to read the PDF file
            pdf_reader = PyPDF2.PdfReader(file, strict=False)  # Specify strict=False to ignore unsupported encryption types
            text = ""
            # Extract text from each page and replace newline characters with white space
            for page in pdf_reader.pages:
                text += page.extract_text().replace("\\n", " ")
        # Append filename and extracted text to the data list
        data.append([filename, text])
    except Exception as e:
        print(f'Error occurred while processing {filename}: {e}')
        continue

# Create a DataFrame from the extracted data
df = pd.DataFrame(data, columns=["source", "text"])
df.head()

# COMMAND ----------

df.to_csv(SCRAPPED_DATA_PATH, index=False, escapechar='\\')
df = pd.read_csv(SCRAPPED_DATA_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC Split docs into chunks

# COMMAND ----------

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, separator=" ")

def get_chunks(input_row):
    new_rows = []
    chunks = text_splitter.split_text(str(input_row["text"]))
    for i, chunk in enumerate(chunks):
        # discard chunks >20% of chunk size
        if len(chunk) <= CHUNK_SIZE * 1.2:
            new_rows.append({"chunk": chunk, "source": input_row["source"], "chunk_index": i})
    return new_rows

expanded_df = pd.DataFrame(columns=["chunk", "source", "chunk_index"])
for index, row in df.iterrows():
    new_rows = get_chunks(row)
    expanded_df = pd.concat([expanded_df, pd.DataFrame(new_rows)], ignore_index=True)
expanded_df.head()

# COMMAND ----------

start, end = 0, -1
filtered_df = (
    expanded_df.groupby("source").apply(lambda x: x.iloc[start:end]).reset_index(drop=True)
)
filtered_df.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC Generate questions

# COMMAND ----------

print(filtered_df["chunk"][0])

# COMMAND ----------

def get_raw_response(content):
    prompt = f"""Please generate a question asking for the key information in the given paragraph.
    Also answer the questions using the information in the given paragraph.
    Please ask the specific question instead of the general question, like
    'What is the key information in the given paragraph?'.
    Please generate the answer using as much information as possible.
    If you are unable to answer it, please generate the answer as 'I don't know.'
    The answer should be informative and should be more than 3 sentences.

    Paragraph: {content}

    Please call the submit_function function to submit the generated question and answer.
    """

    messages = [{"role": "user", "content": prompt}]

    submit_function = {
        "name": "submit_function",
        "description": "Call this function to submit the generated question and answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question asking for the key information in the given paragraph.",
                },
                "answer": {
                    "type": "string",
                    "description": "The answer to the question using the information in the given paragraph.",
                },
            },
            "required": ["question", "answer"],
        },
    }

    return openai.ChatCompletion.create(
        messages=messages,
        model="gpt-3.5-turbo",
        functions=[submit_function],
        function_call="auto",
        temperature=0.0,
        seed=SEED,
    )


def generate_question_answer(content):
    if content is None or len(content) == 0:
        return "", "N/A"

    response = get_raw_response(content)
    try:
        func_args = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])
        question = func_args["question"]
        answer = func_args["answer"]
        return question, answer
    except Exception as e:
        return str(e), "N/A"

# COMMAND ----------

queries = []

# COMMAND ----------

get_raw_response(filtered_df["chunk"][0])

# COMMAND ----------

n = len(filtered_df)
for i, row in filtered_df.iterrows():
    chunk = row["chunk"]
    question, answer = generate_question_answer(chunk)
    print(f"{i+1}/{n}: {question}")
    queries.append(
        {
            "question": question,
            "answer": answer,
            "chunk": chunk,
            "chunk_id": row["chunk_index"],
            "source": row["source"],
        }
    )

# COMMAND ----------

result_df = pd.DataFrame(queries)
result_df = result_df[result_df["answer"] != "N/A"]

# COMMAND ----------

def add_to_output_df(result_df=pd.DataFrame({})):
    """
    This function adds the records in result_df to the existing records saved at OUTPUT_DF_PATH,
    remove the duplicate rows and save the new collection of records back to OUTPUT_DF_PATH.
    """
    if os.path.exists(OUTPUT_DF_PATH):
        all_result_df = pd.read_csv(OUTPUT_DF_PATH)
    else:
        all_result_df = pd.DataFrame({})
    all_result_df = (
        pd.concat([all_result_df, result_df], ignore_index=True)
        .drop_duplicates()
        .sort_values(by=["source", "chunk_id"])
        .reset_index(drop=True)
    )
    all_result_df.to_csv(OUTPUT_DF_PATH, index=False)
    return all_result_df

# COMMAND ----------

all_result_df = add_to_output_df(result_df)

# COMMAND ----------

all_result_df

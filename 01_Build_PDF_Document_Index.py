# Databricks notebook source
# MAGIC %md The purpose of this notebook is to access and prepare our data for use with the QA Bot accelerator.  This notebook is available at https://github.com/databricks-industry-solutions/diy-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC So that our qabot application can respond to user questions with relevant answers, we will provide our model with content from documents relevant to the question being asked.  The idea is that the bot will leverage the information in these documents as it formulates a response.
# MAGIC
# MAGIC For our application, we've extracted a series of documents from [Databricks documentation](https://docs.databricks.com/), [Spark documentation](https://spark.apache.org/docs/latest/), and the [Databricks Knowledge Base](https://kb.databricks.com/).  Databricks Knowledge Base is an online forum where frequently asked questions are addressed with high-quality, detailed responses.  Using these three documentation sources to provide context will allow our bot to respond to questions relevant to this subject area with deep expertise.
# MAGIC
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/bot_data_processing4.png' width=700>
# MAGIC
# MAGIC </p>
# MAGIC
# MAGIC In this notebook, we will load these documents, extracted as a series of JSON documents through a separate process, to a table in the Databricks environment.  We will retrieve those documents along with metadata about them and feed that to a vector store which will create on index enabling fast document search and retrieval.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install PyPDF2 langchain==0.0.166 tiktoken==0.4.0 openai==0.27.6 faiss-cpu==1.7.4 typing-inspect==0.8.0 typing_extensions==4.5.0 pycryptodome==3.19.0 reportlab==4.0.7 pypandoc==1.12
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC apt-get update
# MAGIC apt-get install -y pandoc
# MAGIC apt-get install -y texlive

# COMMAND ----------

# DBTITLE 1,Import Required Functions
import pyspark.sql.functions as fn
from langchain.document_loaders import PyPDFLoader

import json

from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# MAGIC %md ##Step 1: Load the Raw Data to Table
# MAGIC
# MAGIC A snapshot of the three documentation sources is made available at a publicly accessible cloud storage. Our first step is to access the extracted documents. We can load them to a table using a Spark DataReader configured for reading [JSON](https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.DataFrameReader.json.html) with the `multiLine` option.  

# COMMAND ----------

# MAGIC %md We can persist this data to a table as follows:

# COMMAND ----------

# MAGIC %md ##Step 2: Prepare Data for Indexing
# MAGIC
# MAGIC While there are many fields avaiable to us in our newly loaded table, the fields that are relevant for our application are:
# MAGIC
# MAGIC * text - Documentation text or knowledge base response which may include relevant information about user's question
# MAGIC * source - the url pointing to the online document

# COMMAND ----------

# MAGIC %md The content available within each doc varies but some documents can be quite long.  Here is an example of a large document in our dataset:

# COMMAND ----------

# MAGIC %md The process of converting a document to an index involves us translating it to a fixed-size embedding.  An embedding is a set of numerical values, kind of like a coordinate, that summarizes the content in a unit of text. While large embeddings are capable of capturing quite a bit of detail about a document, the larger the document submitted to it, the more the embedding generalizes the content.  It's kind of like asking someone to summarize a paragraph, a chapter or an entire book into a fixed number of dimensions.  The greater the scope, the more the summary must eliminate detail and focus on the higher-level concepts in the text.
# MAGIC
# MAGIC A common strategy for dealing with this when generating embeddings is to divide the text into chunks.  These chunks need to be large enough to capture meaningful detail but not so large that key elements get washed out in the generalization.  Its more of an art than a science to determine an appropriate chunk size, but here we'll use a very small chunk size to illustrate what's happening in this step:

# COMMAND ----------

import os
import pypandoc
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


def wrap_text(c, text, max_width):
    lines = []
    words = text.split()
    current_line = words[0]

    for word in words[1:]:
        if c.stringWidth(current_line + " " + word, "Helvetica", 12) < max_width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines

def convert_to_pdf(input_file, output_folder):
    try:
        # Check if the file is a Word document or a text file
        if input_file.lower().endswith((".docx", ".doc")):
            # Generate the output PDF filename
            output_pdf_name = os.path.splitext(os.path.basename(input_file))[0] + ".pdf"
            output_pdf_path = os.path.join(output_folder, output_pdf_name)

            # Convert the document to PDF using pandoc
            pypandoc.convert_file(input_file, 'pdf', outputfile=output_pdf_path)

            print(f"Conversion successful. PDF saved to {output_pdf_path}")

        # Check if the file is a text file
        elif input_file.lower().endswith(".txt"):
            with open(input_file, 'r') as file:
                input_text = file.read()

            # Generate the output PDF filename
            output_pdf_name = os.path.splitext(os.path.basename(input_file))[0] + ".pdf"
            output_pdf_path = os.path.join(output_folder, output_pdf_name)

            # Create a new PDF document using reportlab
            c = canvas.Canvas(output_pdf_path, pagesize=letter)
            c.setFont("Helvetica", 12)

            # Set starting y-coordinate for the text
            y = 750

            # Define available width for text
            available_width = 400

            # Write each line to the PDF with manual text wrapping
            for line in wrap_text(c, input_text, available_width):
                c.drawString(100, y, line)
                y -= 12  # Move to the next line

            # Save the PDF
            c.save()

            print(f"Conversion successful. PDF saved to {output_pdf_path}")

    except Exception as e:
        print(f"Error processing '{input_file}': {e}")

def convert_folder_to_pdf(input_folder, output_folder):
    try:
        for root, dirs, files in os.walk(input_folder):
            for filename in files:
                input_file_path = os.path.join(root, filename)
                convert_to_pdf(input_file_path, output_folder)

    except Exception as e:
        print(f"Error processing folder '{input_folder}': {e}")

# Replace 'input_folder_path' and 'output_folder_path' with the appropriate paths
input_folder_path = '/Volumes/prototype/callum/research_volume'
output_folder_path = '/Volumes/prototype/callum/research_volume'

try:
    convert_folder_to_pdf(input_folder_path, output_folder_path)

except FileNotFoundError:
    print(f"Error: Folder '{input_folder_path}' not found.")
except Exception as e:
    print(f"Error: {e}")


# COMMAND ----------

# DBTITLE 1,Split Text into Chunks
from PyPDF2 import PdfReader
from pathlib import Path
from Crypto.Cipher import AES

# Function to read a specific PDF and extract text
def extract_text_from_pdf(file_path):
    text = ''
    for path in Path(file_path).glob("**/*.pdf"):
        print("Found pdf file: ",path)
        with open(path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    return text

# File path of the PDF
pdf_file_path = '/Volumes/prototype/{}/{}_volume'.format(user_name,user_name)

# Extract text from the specified PDF file
extracted_text = extract_text_from_pdf(pdf_file_path)

# Create a DataFrame with the extracted text
raw = spark.createDataFrame([(extracted_text,)], ["text"],['source'])

# Display the DataFrame
display(raw)


# COMMAND ----------

# Assuming `extracted_text` contains the text from your PDF
raw_inputs = spark.createDataFrame([(extracted_text,)], ["text"])


# COMMAND ----------

# MAGIC %md Please note that we are specifying overlap between our chunks.  This is to help avoid the arbitrary separation of words that might capture a key concept. 
# MAGIC
# MAGIC We have set our overlap size very small for this demonstration but you may notice that overlap size does not neatly translate into the exact number of words that will overlap between chunks. This is because we are not splitting the content directly on words but instead on byte-pair encoding tokens derived from the words that make up the text.  You can learn more about byte-pair encoding [here](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt) but just note that its a frequently employed mechanism for compressing text in many LLM algorithms.

# COMMAND ----------

# MAGIC %md With the concept of document splitting under our belt, let's write a function to divide our documents into chunks and apply it to our data. Note that we are setting the chunk size and overlap to higher values for this step to better align with the [limits](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) specified with the Chat-GPT model we will eventually transmit this information to.  You might be able to set these values higher but please note that a fixed number of *tokens* are currently allowed with each Chat-GPT model request and that the entire user prompt (including context) and the generated response must fit within that token limit.  Otherwise, an error will be generated:

# COMMAND ----------

# DBTITLE 1,Chunking Configurations
chunk_size = 200
chunk_overlap = 100

# COMMAND ----------

# DBTITLE 1,Divide Inputs into Chunks
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from langchain.text_splitter import TokenTextSplitter

chunk_size = 300  # Set the desired chunk size
chunk_overlap = 100  # Set the desired overlap size

@udf(ArrayType(StringType()))
def get_chunks(text):
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Apply the chunking logic to the DataFrame
chunked_inputs = (
    raw_inputs
    .withColumn('chunks', get_chunks('text'))
    .drop('text')
    .withColumn('num_chunks', fn.size('chunks'))
    .withColumn('chunk', fn.explode('chunks'))
    .drop('chunks')
    .withColumnRenamed('chunk', 'text')
)

# Display the transformed data
display(chunked_inputs)


# COMMAND ----------

# MAGIC %md ##Step 4: Create Vector Store
# MAGIC
# MAGIC With our data divided into chunks, we are ready to convert these records into searchable embeddings. Our first step is to separate the content that will be converted from the content that will serve as the metadata surrounding the document:

# COMMAND ----------

# DBTITLE 1,Separate Inputs into Searchable Text & Metadata
# convert inputs to pandas dataframe
inputs = chunked_inputs.toPandas()

# extract searchable text elements
text_inputs = inputs['text'].to_list()



# COMMAND ----------

# MAGIC %md Next, we will initialize the vector store into which we will load our data.  If you are not familiar with vector stores, these are specialized databases that store text data as embeddings and enable fast searches based on content similarity.  We will be using the [FAISS vector store](https://faiss.ai/) developed by Facebook AI Research. It's fast and lightweight, characteristics that make it ideal for our scenario.
# MAGIC
# MAGIC The key to setting up the vector store is to configure it with an embedding model that it will used to convert both the documents and any searchable text to an embedding (vector). You have a wide range of choices avaialble to you as you consider which embedding model to employ.  Some popular models include the [sentence-transformer](https://huggingface.co/models?library=sentence-transformers&sort=downloads) family of models available on the HuggingFace hub as well as the [OpenAI embedding models](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings):
# MAGIC
# MAGIC **NOTE** The OpenAI API key used by the OpenAIEmbeddings object is specified in an environment variable set during the earlier `%run` call to get configuration variables.

# COMMAND ----------

# DBTITLE 1,Load Vector Store
# identify embedding model that will generate embedding vectors
embeddings = OpenAIEmbeddings(model=config['openai_embedding_model'])

# instantiate vector store object
vector_store = FAISS.from_texts(
  embedding=embeddings, 
  texts=text_inputs
  )

# COMMAND ----------

# MAGIC %md So that we make use of our vector store in subsequent notebooks, let's persist it to storage:

# COMMAND ----------

# DBTITLE 1,Persist Vector Store to Storage
vector_store.save_local(folder_path=config['vector_store_path'])

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |

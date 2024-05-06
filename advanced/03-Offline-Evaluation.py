# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # 3/ Evaluating the RAG Chat Bot with LLMs-as-a-Judge for automated evaluation
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-llm-as-a-judge.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC Now that our RAG model is deployed, we aim to evaluate its predictions correctness.
# MAGIC
# MAGIC Evaluating LLMs can be challenging as existing benchmarks and metrics can not measure them comprehensively. Humans are often involved in these tasks (see [RLHF](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback), but it doesn't scale well: humans are slow and expensive!
# MAGIC
# MAGIC ## Introducing LLM-as-a-Judge
# MAGIC
# MAGIC In this notebook, we'll automate the evaluation process with a trending approach in the LLM community: **LLMs-as-a-judge**.
# MAGIC
# MAGIC Faster and cheaper than human evaluation, LLM-as-a-Judge leverages an external agent who judges the generative model predictions given what is expected from it.
# MAGIC
# MAGIC Superior models are typically used for such evaluation (e.g. `llama2-70B` judges `llama2-7B`, or `GPT4` judges `llama2-70B`)
# MAGIC
# MAGIC We'll explore the new LLMs-as-a-judges evaluation methods introduced in MLflow 2.9, with its powerful `mlflow.evaluate()`API.+
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=5890053377436761&notebook=%2F02-advanced%2F03-Offline-Evaluation&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F02-advanced%2F03-Offline-Evaluation&version=1">

# COMMAND ----------

# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-llm-rag-chatbot-callum_elder` from the dropdown menu ([open cluster configuration](https://adb-5890053377436761.1.azuredatabricks.net/#setting/clusters/0315-001658-1toyiaqo/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('llm-rag-chatbot')` or re-install the demo: `dbdemos.install('llm-rag-chatbot')`*

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.12.0 databricks-genai-inference==0.1.1 mlflow==2.9.0 textstat==0.7.3 tiktoken==0.5.1 evaluate==0.4.1 langchain==0.1.16 databricks-vectorsearch==0.22 transformers==4.30.2 torch==2.1.2 cloudpickle==2.2.1 pydantic==2.5.2 --upgrade sqlalchemy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-init-advanced $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Creating an external model endpoint with Azure Open AI as a judge
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/create-external-endpoint.png?raw=true" style="float:right" width="500px" />
# MAGIC
# MAGIC Databricks Serving Endpoint can be of 3 types:
# MAGIC
# MAGIC - Your own models, deployed as an endpoint (a chatbot model, your custom fine tuned LLM)
# MAGIC - Fully managed, serverless Foundation Models (e.g. llama2, MPT...)
# MAGIC - An external Foundation Model (e.g. Azure OpenAI)
# MAGIC
# MAGIC Let's create a external model endpoint using Azure Open AI.
# MAGIC
# MAGIC Note that you'll need to change the values with your own Azure Open AI configuration. Alternatively, you can setup a connection to another provider like OpenAI.
# MAGIC
# MAGIC *Note: If you don't have an Azure OpenAI deployment, this demo will fallback to a Databricks managed llama 2 model. Evaluation won't be as good.* 

# COMMAND ----------

from mlflow.deployments import get_deploy_client
endpoint_name = "claude-2-chat-endpoint"
try:
    client = get_deploy_client("databricks")

    client.create_endpoint(
        name=endpoint_name,
        config={
            "served_entities": [
                {
                    "name": "claude-completions",
                    "external_model": {
                        "name": "claude-2.1",
                        "provider": "anthropic",
                        "task": "llm/v1/chat",
                        "anthropic_config": {
                            "anthropic_api_key": "{{secrets/my_anthropic_secret_scope/anthropic_api_key}}"
                        },
                    },
                }
            ],
        },
    )
except Exception as e:
    if 'RESOURCE_ALREADY_EXISTS' in str(e):
        print('Endpoint already exists')
    else:
        print(f"Couldn't create the external endpoint with Anthropic Claude: {e}. Will fallback to llama2-70-B as judge. Consider using a stronger model as a judge.")
        endpoint_name = "databricks-llama-2-70b-chat"

#Let's query our external model endpoint
answer_test = client.predict(endpoint=endpoint_name, inputs={"messages": [{"role": "user", "content": "What is Apache Spark?"}], "max_tokens": 4096})
answer_test['choices'][0]['message']['content']

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Offline LLM evaluation
# MAGIC
# MAGIC We will start with offline evaluation, scoring our model before its deployment. This requires a set of questions we want to ask to our model.
# MAGIC
# MAGIC In our case, we are fortunate enough to have a labeled training set (questions+answers)  with state-of-the-art technical answers from our Databricks support team. Let's leverage it so we can compare our RAG predictions and ground-truth answers in MLflow.
# MAGIC
# MAGIC **Note**: This is optional! We can benefit from the LLMs-as-a-Judge approach without ground-truth labels. This is typically the case if you want to evaluate "live" models answering any customer questions

# COMMAND ----------

volume_folder =  f"/Volumes/main/rag_chatbot_callum_elder/armhub_datasets"
dataset_path = volume_folder + '/question_answer_source.csv'

# COMMAND ----------

import pandas as pd

# Read the CSV file using pandas
df = pd.read_csv(dataset_path)

# COMMAND ----------

# DBTITLE 1,Preparing our evaluation dataset
# Convert the pandas DataFrame to a Spark DataFrame
spark_df = spark.createDataFrame(df)

# Create a temporary view from the Spark DataFrame
spark_df.createOrReplaceTempView("question_answer_csv")

# Create or replace the evaluation_dataset table
spark.sql('''
CREATE OR REPLACE TABLE evaluation_dataset AS
  SELECT question, answer 
  FROM question_answer_csv
''')

# Display the evaluation_dataset table
display(spark.table('evaluation_dataset'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Automated Evaluation of our chatbot model registered in Unity Catalog
# MAGIC
# MAGIC Let's retrieve the chatbot model we registered in Unity Catalog and predict answers for each questions in the evaluation set.

# COMMAND ----------

# MAGIC %pip install mlflow[databricks]

# COMMAND ----------

import mlflow
import os

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos-callum", "rag_sp_token")
model_name = f"{catalog}.{db}.gpt_advanced_chatbot_model_armhub"
model_version_to_evaluate = get_latest_model_version(model_name)
mlflow.set_registry_uri("databricks-uc")
rag_model = mlflow.langchain.load_model(f"models:/{model_name}/{model_version_to_evaluate}")

@pandas_udf("string")
def predict_answer(questions):
    def answer_question(question):
        dialog = {"messages": [{"role": "user", "content": question}]}
        return rag_model.invoke(dialog)['result']
    return questions.apply(answer_question)

# COMMAND ----------

df_qa = (spark.read.table('evaluation_dataset')
                  .selectExpr('question as inputs', 'answer as targets')
                  .where("targets is not null")
                  .sample(fraction=0.005, seed=40)) #small sample for interactive demo

df_qa_with_preds = df_qa.withColumn('preds', predict_answer(col('inputs'))).cache()

display(df_qa_with_preds)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##LLMs-as-a-judge: automated LLM evaluation with out of the box and custom GenAI metrics
# MAGIC
# MAGIC MLflow 2.8 provides out of the box GenAI metrics and enables us to make our own GenAI metrics:
# MAGIC - Mlflow will automatically compute relevant task-related metrics. In our case, `model_type='question-answering'` will add the `toxicity` and `token_count` metrics.
# MAGIC - Then, we can import out of the box metrics provided by MLflow 2.8. Let's benefit from our ground-truth labels by computing the `answer_correctness` metric. 
# MAGIC - Finally, we can define customer metrics. Here, creativity is the only limit. In our demo, we will evaluate the `professionalism` of our Q&A chatbot.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Custom correctness answer
from mlflow.metrics.genai.metric_definitions import answer_correctness
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

# Because we have our labels (answers) within the evaluation dataset, we can evaluate the answer correctness as part of our metric. Again, this is optional.
answer_correctness_metrics = answer_correctness(model=f"endpoints:/{endpoint_name}")
print(answer_correctness_metrics)

# COMMAND ----------

# DBTITLE 1,Adding custom professionalism metric
professionalism_example = EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps "
        "you track experiments, package your code and models, and collaborate with your team, making the whole ML "
        "workflow smoother. It's like your Swiss Army knife for machine learning!"
    ),
    score=2,
    justification=(
        "The response is written in a casual tone. It uses contractions, filler words such as 'like', and "
        "exclamation points, which make it sound less professional. "
    )
)

professionalism = make_genai_metric(
    name="professionalism",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is "
        "tailored to the context and audience. It often involves avoiding overly casual language, slang, or "
        "colloquialisms, and instead using clear, concise, and respectful language."
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below are the details for different scores: "
        "- Score 1: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for "
        "professional contexts."
        "- Score 2: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
        "some informal professional settings."
        "- Score 3: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
        "- Score 4: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 5: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal "
        "business or academic settings. "
    ),
    model=f"endpoints:/{endpoint_name}",
    parameters={"temperature": 0.0, "max_tokens": 4096},
    aggregations=["mean", "variance"],
    examples=[professionalism_example],
    greater_is_better=True
)

print(professionalism)

# COMMAND ----------

# DBTITLE 1,Start the evaluation run
from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")

#This will automatically log all
with mlflow.start_run(run_name="chatbot_rag") as run:
    eval_results = mlflow.evaluate(data = df_qa_with_preds.toPandas(), # evaluation data,
                                   model_type="question-answering", # toxicity and token_count will be evaluated   
                                   predictions="preds", # prediction column_name from eval_df
                                   targets = "targets",
                                   extra_metrics=[answer_correctness_metrics, professionalism])
    
eval_results.metrics

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Visualization of our GenAI metrics produced by our GPT4 judge
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-llm-as-a-judge-mlflow.png?raw=true" style="float: right; margin-left:10px" width="800px">
# MAGIC
# MAGIC You can open your MLFlow experiment runs from the Experiments menu on the right. 
# MAGIC
# MAGIC From here, you can compare multiple model versions, and filter by correctness to spot where your model doesn't answer well. 
# MAGIC
# MAGIC Based on that and depending on the issue, you can either fine tune your prompt, your model fine tuning instruction with RLHF, or improve your documentation.
# MAGIC <br style="clear: both"/>
# MAGIC
# MAGIC ### Custom visualizations
# MAGIC You can equaly plot the evaluation metrics directly from the run, or pulling the data from MLFlow:

# COMMAND ----------

df_genai_metrics = eval_results.tables["eval_results_table"]
display(df_genai_metrics)

# COMMAND ----------

import plotly.express as px
px.histogram(df_genai_metrics, x="token_count", labels={"token_count": "Token Count"}, title="Distribution of Token Counts in Model Responses")

# COMMAND ----------

# Counting the occurrences of each answer correctness score
px.bar(df_genai_metrics['answer_correctness/v1/score'].value_counts(), title='Answer Correctness Score Distribution')

# COMMAND ----------

df_genai_metrics['toxicity'] = df_genai_metrics['toxicity/v1/score'] * 100
fig = px.scatter(df_genai_metrics, x='toxicity', y='answer_correctness/v1/score', title='Toxicity vs Correctness', size=[10]*len(df_genai_metrics))
fig.update_xaxes(tickformat=".2f")

# COMMAND ----------

# MAGIC %md
# MAGIC #Retrieval System Evaluation

# COMMAND ----------

# Prepare dataframe `data` with the required format
data = pd.DataFrame({})
data["question"] = df["question"].copy(deep=True)
data["source"] = df["source"].apply(lambda x: [x])
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC #Set up embedding model endpoints

# COMMAND ----------

# MAGIC %md
# MAGIC ##Databricks Embedding Model

# COMMAND ----------

from langchain_community.embeddings import DatabricksEmbeddings

bge_embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# COMMAND ----------

# MAGIC %md
# MAGIC ##OpenAI Embedding Models

# COMMAND ----------

from langchain_community.embeddings import DatabricksEmbeddings

ada_embedding_model = DatabricksEmbeddings(endpoint="text-embedding-ada-002")

# COMMAND ----------

# MAGIC %md
# MAGIC ##OpenAI Large Embedding Model

# COMMAND ----------

from langchain_community.embeddings import DatabricksEmbeddings

openai_large_embedding_model = DatabricksEmbeddings(endpoint="text-embedding-3-large")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Get Retriever

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.chains import RetrievalQA
import os

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos-callum", "rag_sp_token")

VECTOR_SEARCH_ENDPOINT_NAME = "500_25_experimental_vector_search" # change for each experiment
index_name = "main.rag_chatbot_callum_elder.500_25_experimental_openai_large_self_managed_vs_index" # change for each embedding index
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

def get_retriever(embedding_model, persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model, columns=["basename"]
    )
    return vectorstore.as_retriever(search_kwargs={'k': 4})

retriever = get_retriever(openai_large_embedding_model)  # change for each embedding model

# COMMAND ----------

# Test the retriever with a query
retrieved_docs = retriever.get_relevant_documents(
    "What is ARM Hub?"
)
len(retrieved_docs)

# COMMAND ----------

from typing import List

# Define a function to return a list of retrieved doc ids
def retrieve_doc_ids(question: str) -> List[str]:
    docs = retriever.get_relevant_documents(question)
    return [doc.metadata["basename"] + ".pdf" for doc in docs]

data["retrieved_doc_ids"] = data["question"].apply(retrieve_doc_ids)
print(data)

# COMMAND ----------

data.to_csv("500_25_openai_large_retrieval_dataset.csv", index=False)

# COMMAND ----------

with mlflow.start_run() as run:
    evaluate_results = mlflow.evaluate(
        data=data,
        targets="source",
        predictions="retrieved_doc_ids",
        evaluators="default",
        extra_metrics=[
            mlflow.metrics.precision_at_k(1),
            mlflow.metrics.precision_at_k(2),
            mlflow.metrics.precision_at_k(3),
            mlflow.metrics.recall_at_k(1),
            mlflow.metrics.recall_at_k(2),
            mlflow.metrics.recall_at_k(3),
            mlflow.metrics.ndcg_at_k(1),
            mlflow.metrics.ndcg_at_k(2),
            mlflow.metrics.ndcg_at_k(3),
        ],
    )

# COMMAND ----------

display(evaluate_results.tables["eval_results_table"])

# COMMAND ----------

evaluate_results.tables["eval_results_table"].to_csv("500_25_openai_large_evaluate_results.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Plot Values

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

datasets = ["500_25_bge_evaluate_results.csv", "500_25_ada_evaluate_results.csv", "500_25_openai_large_evaluate_results.csv"]
metrics = ["precision", "recall", "ndcg"]
names = ["bge", "ada", "openai_large"]
k_values = [1, 2, 3]

# Create a figure with subplots for each metric
fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

for i, metric_name in enumerate(metrics):
    for j, k in enumerate(k_values):
        x = [j + dataset_index * (len(k_values) + 1) for dataset_index in range(len(datasets))]
        y = [pd.read_csv(dataset)[f"{metric_name}_at_{k}/score"].mean() for dataset in datasets]
        axes[i].bar(x, y, width=0.8, label=f"k={k}")

    axes[i].set_xlabel("Dataset")
    axes[i].set_ylabel(f"{metric_name.capitalize()} Value Mean")
    axes[i].set_title(f"{metric_name.capitalize()}@k")
    axes[i].set_xticks([i * (len(k_values) + 1) + (len(k_values) - 1) / 2 for i in range(len(datasets))])
    axes[i].set_xticklabels(names)
    axes[i].set_ylim(0, 1)
    axes[i].legend()

# Add an overall title
fig.suptitle("Evaluation Metrics for Chunking Strategy 500/25", fontsize=16)

# Adjust the spacing between subplots
plt.tight_layout()

# Save the plot to a file
plt.savefig("evaluation_metrics_plot_500_25.png")

# Display the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## This is looking good, let's tag our model as production ready
# MAGIC
# MAGIC After reviewing the model correctness and potentially comparing its behavior to your other previous version, we can flag our model as ready to be deployed.
# MAGIC
# MAGIC *Note: Evaluation can be automated and part of a MLOps step: once you deploy a new Chatbot version with a new prompt, run the evaluation job and benchmark your model behavior vs the previous version.*

# COMMAND ----------

client = MlflowClient()
client.set_registered_model_alias(name=model_name, alias="prod", version=model_version_to_evaluate)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Databricks AI makes it easy to evaluate your LLM Models, leveraging custom metrics.
# MAGIC
# MAGIC Evaluating your chatbot is key to measure your future version impact, and your Data Intelligence Platform makes it easy leveraging automated Workflow for your MLOps pipelines.
# MAGIC
# MAGIC For a production-grade GenAI application, this step should be automated and part as a job, executed everytime the model is changed and benchmarked against previous run to make sure you don't have performance regression.
# MAGIC
# MAGIC ### Next: Deploy our model as Model Serving Endpoint with Inference Tables and deploy LLM metric monitoring (live monitoring)
# MAGIC
# MAGIC Open the [04-Deploy-Model-as-Endpoint]($./04-Deploy-Model-as-Endpoint) to deploy your model and track your endpoint payload as a Delta Table.

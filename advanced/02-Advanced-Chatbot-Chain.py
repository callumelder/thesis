# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # 2/ Advanced chatbot with message history and filter using Langchain
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-flow-2.png?raw=true" style="float: right; margin-left: 10px"  width="900px;">
# MAGIC
# MAGIC Our Vector Search Index is now ready!
# MAGIC
# MAGIC Let's now create a more advanced langchain model to perform RAG.
# MAGIC
# MAGIC We will improve our langchain model with the following:
# MAGIC
# MAGIC - Build a complete chain supporting a chat history, using llama 2 input style
# MAGIC - Add a filter to only answer Databricks-related questions
# MAGIC - Compute the embeddings with Databricks BGE models within our chain to query the self-managed Vector Search Index
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=5890053377436761&notebook=%2F02-advanced%2F02-Advanced-Chatbot-Chain&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F02-advanced%2F02-Advanced-Chatbot-Chain&version=1">
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-llm-rag-chatbot-callum_elder` from the dropdown menu ([open cluster configuration](https://adb-5890053377436761.1.azuredatabricks.net/#setting/clusters/0315-001658-1toyiaqo/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('llm-rag-chatbot')` or re-install the demo: `dbdemos.install('llm-rag-chatbot')`*

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 langchain==0.1.5 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.18.0 cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC %pip install pip mlflow[databricks]==2.10.1
# MAGIC %pip install sqlalchemy --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-init-advanced $reset_all_data=false

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exploring Langchain capabilities
# MAGIC
# MAGIC Let's start with the basics and send a query to a Databricks Foundation Model using LangChain.

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage
from mlflow.deployments import get_deploy_client

# create endpoint for gpt4 model

client = get_deploy_client("databricks")

name = "callums-gpt-endpoint"  # rename when creating new endpoint
try:
  client.create_endpoint(
    name=name,
    config={
      "served_entities": [
        {
          "name": name,
          "external_model": {
            "name": "gpt-4",
            "provider": "openai",
            "task": "llm/v1/chat",
              "openai_config": {
            "openai_api_key": "{{secrets/my_openai_secret_scope/openai_api_key}}"
            }
          },
        }
      ],
    },
  )
except Exception as e:
  if 'RESOURCE_ALREADY_EXISTS' in str(e):
    print('Endpoint already exists')
  else:
    print(e)

chat_model = ChatDatabricks(
  target_uri="databricks",
  endpoint=name,
  temperature=0.1,
)

# COMMAND ----------

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

prompt = PromptTemplate(
  input_variables = ["question"],
  template = "You are an assistant. Give a short answer to this question: {question}"
)

chain = (
  prompt
  | chat_model
  | StrOutputParser()
)
print(chain.invoke({"question": "Who founded google?"}))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Adding conversation history to the prompt 

# COMMAND ----------

prompt_with_history_str = """
You are a ARM Hub's chatbot. Please answer ARM Hub Related questions only. If you don't know the answer or it is not related to ARM Hub, don't answer.

Here is a history between you and a human: {chat_history}

Now, please answer this question: {question}
"""

prompt_with_history = PromptTemplate(
  input_variables = ["chat_history", "question"],
  template = prompt_with_history_str
)

# COMMAND ----------

# MAGIC %md When invoking our chain, we'll pass history as a list, specifying whether each message was sent by a user or the assistant. For example:
# MAGIC
# MAGIC ```
# MAGIC [
# MAGIC   {"role": "user", "content": "What is Apache Spark?"}, 
# MAGIC   {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
# MAGIC   {"role": "user", "content": "Does it support streaming?"}
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC Let's create chain components to transform this input into the inputs passed to `prompt_with_history`.

# COMMAND ----------

from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

chain_with_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | prompt_with_history
    | chat_model
    | StrOutputParser()
)

print(chain_with_history.invoke({
    "messages": [
        {"role": "user", "content": "How do I engage with ARM Hub?"}, 
        {"role": "assistant", "content": "To engage with ARM Hub, you can start by visiting their website and exploring the various resources and tools available. You can also sign up for their newsletter to stay up-to-date on the latest news and developments in the ARM ecosystem. Additionally, you can participate in online communities and forums related to ARM technology to connect with other developers and users.."}, 
        {"role": "user", "content": "Where is ARM Hub?"}
    ]
}))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Use LangChain to retrieve documents from the vector store
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-model-1.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC Let's add our LangChain retriever. 
# MAGIC
# MAGIC It will be in charge of:
# MAGIC
# MAGIC * Creating the input question embeddings (with Databricks `bge-large-en`)
# MAGIC * Calling the vector search index to find similar documents to augment the prompt with
# MAGIC
# MAGIC Databricks LangChain wrapper makes it easy to do in one step, handling all the underlying logic and API call for you.

# COMMAND ----------

index_name = "main.rag_chatbot_callum_elder.250_12_experimental_openai_large_self_managed_vs_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
embedding_endpoint_name = "text-embedding-3-large"
VECTOR_SEARCH_ENDPOINT_NAME = "250_12_experimental_vector_search"
 
#Let's make sure the secret is properly setup and can access our vector search index. Check the quick-start demo for more guidance
test_demo_permissions(host, secret_scope="dbdemos-callum", secret_key="rag_sp_token", vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name, embedding_endpoint_name=embedding_endpoint_name, managed_embeddings = False)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos-callum", "rag_sp_token")

embedding_model = DatabricksEmbeddings(endpoint=embedding_endpoint_name)

def get_retriever(persist_dir: str = None):
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

retriever = get_retriever()

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | retriever
)
print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "Who is Cori Stewart?"}]}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Improve document search using LLM to generate a better sentence for the vector store, based on the chat history
# MAGIC
# MAGIC We need to retrieve documents related the the last question but also the history.
# MAGIC
# MAGIC One solution is to add a step for our LLM to summarize the history and the last question, making it a better fit for our vector search query. Let's do that as a new step in our chain:

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | RunnableBranch(  #Augment query only when there is a chat history
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

#Let's try it
output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is ARM Hub?"}
    ]
})
print(f"Test retriever query without history: {output}")

output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is ARM Hub?"}, 
        {"role": "assistant", "content": "ARM Hub is an independent, not-for-profit organization that aims to accelerate the adoption of advanced manufacturing technologies in Australia. It serves as an aggregator of research and development, connecting private industry, research institutions, and government to help uplift, upskill, and transform Australian manufacturing with a particular focus on small and medium-sized enterprises (SMEs). ARM Hub facilitates the creation and adoption of advanced manufacturing technologies and processes by providing expertise from researchers, engineers, and roboticists in priority technical areas such as automation and robotics, data science, image processing and computer vision, human-robot interaction, and process design. They also build expert teams to address the specific needs of business transformations and apply Industry 4.0 technologies to meet industry challenges."}, 
        {"role": "user", "content": "How do I engage it?"}
    ]
})
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Let's put it together
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-model-2.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC
# MAGIC Let's now merge the retriever and the full LangChain chain.
# MAGIC
# MAGIC We will use a custom LangChain template for our assistant to give a proper answer.
# MAGIC
# MAGIC Make sure you take some time to try different templates and adjust your assistant tone and personality for your requirement.
# MAGIC
# MAGIC

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough
from operator import itemgetter


# Template to handle any questions, assuming all are relevant
question_with_history_and_context_str = """
You are ARM Hub's friendly chatbot. You answer questions, based on the companies data, to the best of your ability. You are to answer the question in a professional manner. Use the discussion to understand the context of the question if relevant. You are to be respectful and conduct yourself to a high standard.
Discussion: {chat_history}

Here's some context which may be relevant to the question: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])

def extract_source_urls(docs):
    return [d.metadata["basename"] for d in docs]

# Process all questions through this chain
answer_all_questions_chain = (
    RunnablePassthrough() |
        {
            "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
            "chat_history": itemgetter("chat_history"),
            "question": itemgetter("question")
        }
    |
        {
            "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
            "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
            "chat_history": itemgetter("chat_history"),
            "question": itemgetter("question")
        }
    |
        {
            "prompt": question_with_history_and_context_prompt,
            "sources": itemgetter("sources")
        }
    |
        {
            "result": itemgetter("prompt") | chat_model | StrOutputParser(),
            "sources": itemgetter("sources")
        }
)

# Full chain that always navigates to answer_all_questions_chain
full_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | answer_all_questions_chain
)

# Example usage:
result = full_chain.invoke({ "messages": [ {"role": "user", "content": "What is Apache Spark?"}, {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, {"role": "user", "content": "Who is Cori Stewart?"} ] })
print(result)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Register the chatbot model to Unity Catalog

# COMMAND ----------

import cloudpickle
import langchain
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.gpt_optimized_chatbot_model_thesis"

dialog = {
    "messages": [
        {"role": "user", "content": "What is ARM Hub?"}, 
        {"role": "assistant", "content": "ARM Hub is an independent, not-for-profit organization that aims to accelerate the adoption of advanced manufacturing technologies in Australia. It serves as an aggregator of research and development, connecting private industry, research institutions, and government to help uplift, upskill, and transform Australian manufacturing with a particular focus on small and medium-sized enterprises (SMEs). ARM Hub facilitates the creation and adoption of advanced manufacturing technologies and processes by providing expertise from researchers, engineers, and roboticists in priority technical areas such as automation and robotics, data science, image processing and computer vision, human-robot interaction, and process design. They also build expert teams to address the specific needs of business transformations and apply Industry 4.0 technologies to meet industry challenges."}, 
        {"role": "user", "content": "How do I engage it?"}
    ]
}

with mlflow.start_run(run_name="gpt_chatbot_rag") as run:
    #Get our model signature from input/output
    output = full_chain.invoke(dialog)
    signature = infer_signature(dialog, output)

    model_info = mlflow.langchain.log_model(
        full_chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle=="+ cloudpickle.__version__
        ],
        input_example=dialog,
        signature=signature,
        example_no_conversion=True,
    )

# COMMAND ----------

# MAGIC %md Let's try loading our model

# COMMAND ----------

model = mlflow.langchain.load_model(model_info.model_uri)
model.invoke(dialog)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC We've seen how we can improve our chatbot, adding more advanced capabilities to handle a chat history.
# MAGIC
# MAGIC As you add capabilities to your model and tune the prompt, it will get harder to evaluate your model performance in a repeatable way.
# MAGIC
# MAGIC Your new prompt might work well for what you tried to fixed, but could also have impact on other questions.
# MAGIC
# MAGIC ## Next: Introducing offline model evaluation with MLflow
# MAGIC
# MAGIC To solve these issue, we need a repeatable way of testing our model answer as part of our LLMOps deployment!
# MAGIC
# MAGIC Open the next [03-Offline-Evaluation]($./03-Offline-Evaluation) notebook to discover how to evaluate your model.

# Domain Specific Large Language Model for Enterprise: Development and Evaluation

In today's data-driven enterprise landscape, the ability to effectively utilize information has 
become the key to unlocking improved decision-making and task completion. However, a 
challenge lies in the gap between technical expertise required to interact with the data and strategic 
decision-making needed to make informed decisions based on the insights. Domain-specific Large 
Language Models (LLMs) bridge this gap by allowing non-technical users to directly interact with 
the data in a natural language.  

This thesis aims to create an optimal, enterprise-grade LLM pipeline using retrieval augmented 
generation (RAG) allowing the model to refer to company data using an external database. The 
retrieval system and model output were evaluated to find the optimal chunking size, embedding 
model and model prompt. The results showed that the ideal chunking size of 250 with a 12 token 
overlap, OpenAI’s Large Embedding Model and a descriptive few-shot prompt gave the best 
results. The retrieval system was measured against three metrics: precision at k (relevance of 
retrieved documents), recall at k (coverage of relevant documents), and normalized discounted 
cumulative gain (NDCG) at k (ranked quality of retrieved documents). The model output was 
measured using an LLM-as-a-judge to evaluate performance against a synthetic dataset’s ground 
truth, assessing answer correctness and professionalism through custom rubrics.  

To conclude, the optimal parameters are highly dependent on the data within the database and the 
desired model behaviour. Further work could be used to further improve upon this research and 
the designed LLM pipeline. Work suggestions include experimenting with chunking strategies, 
number of documents to be retrieved, extra metrics to test model output, and leveraging work 
from DSPy to quantify model prompt performance. 

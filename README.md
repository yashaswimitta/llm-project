# llm-project
## Product  Recommendor & Analyzer Chatbot  (Amazon)
### Domain

This project develops a Product Recommender & Analyzer Chatbot for Amazon using advanced Large Language Models (LLMs) like GPT-4. The chatbot provides personalized product recommendations, retrieves detailed product information, and analyzes user reviews for sentiment. By integrating NLP, collaborative and content-based filtering, and sentiment analysis, it enhances the user shopping experience. The system aims to deliver a seamless, intelligent virtual assistant for e-commerce platforms.


### Program
```
!pip install langchain
!pip install sentence.transformers
!pip install faiss.gpu
import os
import json
import gzip
import pandas as pd
data= []
with gzip.open('/content/AMAZON_FASHION_5.json.gz') as f:
  for l in f:
    data.append(json.loads(l.strip()))
print(data)
df = pd.DataFrame.from_dict(data)
df=df[df['reviewText'].notna()]
df
max_text_length=500
def truncate_review(text):
  return text[:max_text_length]

df['truncated']=df.apply(lambda row: truncate_review(row['reviewText']),axis=1)
df.groupby('asin').count().sort_values('overall')
df=df.loc[df['asin']=='B0017LD0BM'].copy()
df
texts=df['truncated'].tolist()
texts
!pip install langchain_community
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings() 
db = FAISS.from_texts(texts, embeddings)
from langchain import HuggingFaceHub

repo_id = "tiiuae/falcon-7b-instruct"

key = "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"

llm=HuggingFaceHub(huggingfacehub_api_token= key,
                  repo_id=repo_id, 
                  model_kwargs={"temperature":0.9, "max_length":512})
from langchain.chains import RetrievalQA 
from langchain.schema import retriever

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
query="""These are the reviews for a fashion product.

What is the overall impression of these reviews? Give a short answer. 
Do you recommend buying this product?"""

out = chain.invoke(query)

print(out['result'])
```
### output

![llm 1](https://github.com/user-attachments/assets/f4c1fc1d-1d5d-44ed-bceb-90eceeb6d7c4)

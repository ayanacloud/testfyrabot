from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import boto3
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import BedrockChat
from langchain.llms.bedrock import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
import os

# Load environment variables (Render provides these securely)
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_REGION")

# AWS Clients
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region,
)
bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region,
)

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock
)

# Load FAISS from S3
def load_faiss_index():
    local_path = "faiss_index"
    os.makedirs(local_path, exist_ok=True)
    s3.download_file("testfyrabot", "faiss_store/index.faiss", f"{local_path}/index.faiss")
    s3.download_file("testfyrabot", "faiss_store/index.pkl", f"{local_path}/index.pkl")
    return FAISS.load_local(local_path, bedrock_embeddings, allow_dangerous_deserialization=True)

# Prompt Template
prompt_template = """
Human: You are the Testfyra Bot. Use the following context to answer the question. Be concise, detailed (min 50 words), and helpful. 
Mention that users can reach out via the 'Contact Us' page at www.testfyra.com.
<context>
{context}
</context>

Question: {question}

Assistant (Testfyra Bot):
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# FastAPI App
app = FastAPI()

class Query(BaseModel):
    query: str
    model: str = "claude"  # "claude" or "llama2"

@app.post("/ask")
def ask_bot(q: Query):
    try:
        vectorstore = load_faiss_index()

     #   if q.model == "claude":
    #        llm = BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock)
    #    elif q.model == "llama2":
    #        llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
   #     else:
    #        raise HTTPException(status_code=400, detail="Unsupported model")

        # Always use llama2
        llm = Bedrock(
            model_id="meta.llama3-70b-instruct-v1:0",
            client=bedrock,
            model_kwargs={'max_gen_len': 512}
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        result = qa({"query": q.query})
        return {"response": result['result']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "TestfyraBot is running. Use POST /ask to query."}


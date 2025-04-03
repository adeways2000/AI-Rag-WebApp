# Run a test query like http://127.0.0.1:8000/bedrock/query?text=what%20projects%20has%20the%20student%20done?

from fastapi import FastAPI, HTTPException, Query
import boto3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve AWS configuration from environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")
MODEL_ARN = os.getenv("MODEL_ARN")

app = FastAPI()

# Initialize Boto3 client for Bedrock Agent Runtime
def get_bedrock_client():
    return boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

@app.get("/")
async def root():
    return {"message": "Welcome to your RAG chatbot API!"}

@app.get("/bedrock/query")
async def query_bedrock(text: str = Query(..., description="Input text for the model")):
    client = get_bedrock_client()
    try:
        response = client.retrieve_and_generate(
            input={"text": text},
            retrieveAndGenerateConfiguration={
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ARN
                },
                "type": "KNOWLEDGE_BASE"
            }
        )
        return {"response": response["output"]["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from PIL import Image
import requests
import os
import argparse
import re

# Try to import spaCy for NER
try:
    import spacy
    NER = spacy.load("en_core_web_sm")
except ImportError:
    NER = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalAISystem:
    DEFAULT_MODEL_IDS = {
        "core_llm": "gpt2",  # Local-friendly default
        "vision_model": "Salesforce/blip-image-captioning-base",
        "image_gen": "CompVis/stable-diffusion-v1-4"
    }
    LARGE_MODEL_IDS = {
        "core_llm": "deepseek-ai/deepseek-llm-67b",
        "vision_model": "llava-hf/llava-1.5-7b-hf",
        "image_gen": "stabilityai/stable-diffusion-xl-base-1.0"
    }

    WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")  # Set as env var

    def __init__(self, model_ids=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizer = None
        self.model_ids = model_ids if model_ids else self.DEFAULT_MODEL_IDS

    def load_core_llm(self):
        try:
            if "core_llm" not in self.models:
                logger.info(f"Loading core LLM model: {self.model_ids['core_llm']} ...")
                self.models["core_llm"] = AutoModelForCausalLM.from_pretrained(
                    self.model_ids["core_llm"], torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_ids["core_llm"])
        except Exception as e:
            logger.error(f"Failed to load core LLM: {e}")
            print("Model download instructions: see top of the file docstring.")
            raise

    def load_vision_model(self):
        try:
            if "vision_model" not in self.models:
                logger.info(f"Loading vision model: {self.model_ids['vision_model']} ...")
                self.models["vision_model"] = pipeline(
                    "image-to-text", model=self.model_ids["vision_model"], device=0 if self.device == "cuda" else -1
                )
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            print("Model download instructions: see top of the file docstring.")
            raise

    def load_image_generator(self):
        try:
            if "image_gen" not in self.models:
                logger.info(f"Loading image generator: {self.model_ids['image_gen']} ...")
                self.models["image_gen"] = pipeline(
                    "text-to-image", model=self.model_ids["image_gen"], device=0 if self.device == "cuda" else -1
                )
        except Exception as e:
            logger.error(f"Failed to load image generator: {e}")
            print("Model download instructions: see top of the file docstring.")
            raise

    def handle_text_query(self, prompt):
        self.load_core_llm()
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.models["core_llm"].generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response[len(prompt):].strip()
            return answer
        except Exception as e:
            logger.error(f"Text query failed: {e}")
            return "Sorry, I could not process your request."

    def handle_image_query(self, image_path):
        self.load_vision_model()
        try:
            image = Image.open(image_path)
            result = self.models["vision_model"](image)
            return result[0]['generated_text'] if result else "No description generated."
        except Exception as e:
            logger.error(f"Image query failed: {e}")
            return "Sorry, I could not process your image."

    def generate_image(self, prompt):
        self.load_image_generator()
        try:
            result = self.models["image_gen"](prompt)
            if isinstance(result, list) and result:
                image = result[0]
                output_path = "generated_image.png"
                image.save(output_path)
                return f"Image generated and saved to {output_path}."
            elif isinstance(result, dict) and "images" in result:
                image = result["images"][0]
                output_path = "generated_image.png"
                image.save(output_path)
                return f"Image generated and saved to {output_path}."
            return "Image generation failed."
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return "Sorry, I could not generate the image."

    def access_real_time_data(self, query):
        city = self.extract_city_from_query(query)
        if not city:
            return "Please specify a city for the weather information."
        if not self.WEATHER_API_KEY:
            return "Weather API key is not set."
        try:
            response = requests.get(
                f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.WEATHER_API_KEY}&units=metric"
            )
            data = response.json()
            if response.status_code != 200 or 'main' not in data:
                return "Failed to retrieve weather data."
            return f"Current temperature in {city} is {data['main']['temp']}°C."
        except Exception as e:
            logger.error(f"Weather API failed: {e}")
            return "Failed to access real-time data."

    @staticmethod
    def extract_city_from_query(query):
        # Try spaCy NER if available
        if NER:
            doc = NER(query)
            cities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
            if cities:
                return cities[-1]
        # Fallback regex for "in <city>"
        match = re.search(r'in\s+([A-Za-z ]+)', query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def generate_safe_response(self, query):
        q_lower = query.lower()
        if "generate an image" in q_lower:
            prompt = query.replace("generate an image", "", 1).strip()
            return self.generate_image(prompt)
        elif "describe image" in q_lower:
            image_path = query.replace("describe image", "", 1).strip()
            return self.handle_image_query(image_path)
        elif "weather" in q_lower:
            return self.access_real_time_data(query)
        else:
            return self.handle_text_query(query)
... 
... def parse_args():
...     parser = argparse.ArgumentParser(description="Multimodal AI System with model size options")
...     parser.add_argument("--core_llm", type=str, default=MultimodalAISystem.DEFAULT_MODEL_IDS['core_llm'],
...                         help="Core LLM model (e.g., gpt2, deepseek-ai/deepseek-llm-67b)")
...     parser.add_argument("--vision_model", type=str, default=MultimodalAISystem.DEFAULT_MODEL_IDS['vision_model'],
...                         help="Vision model (e.g., Salesforce/blip-image-captioning-base, llava-hf/llava-1.5-7b-hf)")
...     parser.add_argument("--image_gen", type=str, default=MultimodalAISystem.DEFAULT_MODEL_IDS['image_gen'],
...                         help="Image generation model (e.g., CompVis/stable-diffusion-v1-4, stabilityai/stable-diffusion-xl-base-1.0)")
...     return parser.parse_args()
... 
... if __name__ == "__main__":
...     args = parse_args()
...     model_ids = {
...         "core_llm": args.core_llm,
...         "vision_model": args.vision_model,
...         "image_gen": args.image_gen
...     }
...     ai_system = MultimodalAISystem(model_ids=model_ids)
...     print("Multimodal AI System Initialized. Type 'quit' to exit.")
...     print("Model IDs in use:")
...     for k, v in model_ids.items():
...         print(f"  {k}: {v}")
...     try:
...         while True:
...             user_query = input("\nUser: ")
...             if user_query.lower() == 'quit':
...                 print("Goodbye!")
...                 break
...             response = ai_system.generate_safe_response(user_query)
...             print(f"Assistant: {response}")
...     except KeyboardInterrupt:

# safety_checker.py
... from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification\
... import torch
... import logging
... 
... logger = logging.getLogger(__name__)
... 
... class SafetyChecker:
...     def __init__(self):
...         self.classifier = None
...         self.tokenizer = None
...         
...     def load_model(self):\
...         """Load a dedicated safety classification model."""
...         try:
...             # Using a fine-tuned DeBERTa for safety classification\
...             # 'valurank/distilroberta-base-harmful' is a good, efficient alternative.\
...             model_id = "Meta-Llama/Llama-Guard-2-8b"  # Requires approval on Hugging Face\
...             # model_id = "valurank/distilroberta-base-harmful" # Good alternative, faster.\
... \
...             self.tokenizer = AutoTokenizer.from_pretrained(model_id)\
...             self.model = AutoModelForSequenceClassification.from_pretrained(\
...                 model_id,\
...                 torch_dtype=torch.float16,\
...                 device_map="auto"\
...             )\
...             logger.info("Safety model loaded successfully.")\
...         except Exception as e:\
...             logger.error(f"Failed to load safety model: \{e\}")\
...             # Fallback to a basic pipeline\
...             self.classifier = pipeline("text-classification", model="valurank/distilroberta-base-harmful", device_map="auto")\
... \
...     def is_unsafe(self, text: str, threshold: float = 0.7) -> bool:\
...         """\
...         Classifies text as safe or unsafe using a model.\
...         Returns True if unsafe, False if safe.\
...         """\
...         if self.model is None and self.classifier is None:\
...             self.load_model()\
... \
...         try:\
...             # Use the native model if loaded\
...             if self.model and self.tokenizer:\
...                 inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)\
...                 with torch.no_grad():\
...                     outputs = self.model(**inputs)\
...                 probs = torch.softmax(outputs.logits, dim=-1)\
...                 # Assuming label 1 is "unsafe" (check the model's labels!)\
...                 unsafe_score = probs[0][1].item()\
...                 return unsafe_score > threshold\
...             else: # Fallback to pipeline\
...                 result = self.classifier(text)[0]
...                 # Check if the harmful label is above threshold
...                 if result['label'] == 'harmful' and result['score'] > threshold:
...                     return True
...                return False
...
...       except Exception as e:
...         logger.error(f"Safety classification error: \{e\}. Blocking due to error.")
...           return True  # Block on error to be safe
...
... # Global instance
... safety_checker = SafetyChecker()
...
... # app.py
... from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
... from fastapi.middleware.cors import CORSMiddleware
... from pydantic import BaseModel
... import httpx
... from typing import Optional, List
... import logging
... import uuid
... import time
...
... # Import our modules
from safety_checker import safety_checker
# ... (Import the MultimodalAISystem class from our previous work, now refactored)

# Initialize app and our AI system
app = FastAPI(title="Multimodal AI API", version="1.0.0")
ai_system = MultimodalAISystem()

# Allow frontend requests
app.add_middleware(\
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # For keeping conversation context

class ChatResponse(BaseModel):
    response: str
    session_id: str
    is_safe: bool

class ErrorResponse(BaseModel):
    detail: str

# Global task storage (use Redis in production)
tasks = \{\}

@app.post("/chat", response_model=ChatResponse, responses=\{400: \{"model": ErrorResponse\}, 429: \{"model": ErrorResponse\}\})\
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main endpoint for text-based queries."""
    # 1. Input Safety Check
    if safety_checker.is_unsafe(request.message):
        raise HTTPException(status_code=400, detail="Query violates safety policy.")

    # 2. Generate Response (could be offloaded to a worker queue like Celery for heavy loads)
    try:
        # In a real app, you'd use the session_id to retrieve conversation history
        raw_response = ai_system.generate_safe_response(request.message)
        session_id = request.session_id or str(uuid.uuid4())

        # 3. Output Safety Check
        if safety_checker.is_unsafe(raw_response):
            raw_response = "I apologize, but I cannot generate a response to that request."

        return ChatResponse(response=raw_response, session_id=session_id, is_safe=True)

    except Exception as e:
        logging.error(f"Error generating response: \{e\}")
        raise HTTPException(status_code=500, detail="Internal server error during processing.")

@app.post("/chat-with-image")
async def chat_with_image(
    message: str = Form(...),
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Endpoint for queries with an image."""
    # Check image MIME type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Read image content
    image_content = await image.read()
    
    # Safety check on the text prompt
    if safety_checker.is_unsafe(message):
        raise HTTPException(status_code=400, detail="Query violates safety policy.")

    # Process the image query (implementation details depend on your vision model)
    # You would need to add a method like `ai_system.handle_vision_query_upload(image_content, message)`
    # ... processing logic ...

    return \{"response": "Image analysis result placeholder", "session_id": session_id\}

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return \{"status": "healthy", "timestamp": time.time()\}

if __name__ == "__main__":
    import uvicorn
    # Pre-load models on startup instead of on first request
    print("Pre-loading models...")
    ai_system.load_core_llm()
    safety_checker.load_model()
    print("Models loaded. Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# frontend.py (Optional)
import gradio as gr
import requests

# Change this to your API URL
API_URL = "http://localhost:8000"

def chat_with_ai(message, history, session_id):
    """Send a message to the FastAPI backend."""
    payload = \{"message": message, "session_id": session_id\}
    try:
        response = requests.post(f"\{API_URL\}/chat", json=payload)
        if response.status_code == 200:
            data = response.json()
            return data["response"], data["session_id"]
        else:
            return f"Error: \{response.json().get('detail', 'Unknown error')\}", session_id
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the API server. Is it running?", session_id

# Create the Gradio interface
with gr.Blocks(title="Multimodal AI Chat") as demo:
    session_id = gr.State(value="")  # Store session ID
    gr.Markdown("# \uc0\u55358 \u56598  Your Multimodal AI Assistant")
    
    with gr.Row():
        chatbot = gr.Chatbot(label="Conversation")
    
    with gr.Row():
        msg = gr.Textbox(label="Your Message", placeholder="Type your message here...")
        submit = gr.Button("Send")
    
    gr.Examples(
        examples=[
            "Explain quantum computing simply.",
            "Write a Python function to calculate Fibonacci sequence.",
            "What's the latest news on AI?"
        ],
        inputs=msg
    )

    # Handle chat function\
    def respond(message, chat_history, session_id):\
        bot_message, new_session_id = chat_with_ai(message, chat_history, session_id)
        chat_history.append((message, bot_message))\
        return "", chat_history, new_session_id\

    msg.submit(respond, [msg, chatbot, session_id], [msg, chatbot, session_id])
    submit.click(respond, [msg, chatbot, session_id], [msg, chatbot, session_id])

if __name__ == "__main__":\
    demo.launch(server_port=7860, share=False)  # Set share=True for a public link

# main.py
... from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
... from fastapi.middleware.cors import CORSMiddleware
... from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
... from pydantic import BaseModel, BaseSettings
... from redis import asyncio as aioredis
... import httpx
... import logging
... import time
... from typing import Optional, List
... import uuid
... 
... # --- Configuration (would use environment variables) ---
... class Settings(BaseSettings):
...     app_name: str = "Aegis Production API"
...     redis_url: str = "redis://localhost:6379"
...     vllm_url: str = "http://vllm-server:8000/v1"
...     safety_checker_url: str = "http://triton-server:8000"
...     api_keys: List[str] = ["your-secure-api-key-here"]  # Load from vault
... 
...     class Config:
...         env_file = ".env"
... 
... settings = Settings()
... 
... # --- Security ---
... security = HTTPBearer()
... 
... def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
...     if credentials.credentials not in settings.api_keys:
...         raise HTTPException(status_code=401, detail="Invalid API key")
...     return credentials.credentials
... 
... # --- App Setup ---
... app = FastAPI(title=settings.app_name)
... app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Connections (Async) ---
@app.on_event("startup")
async def startup_event():
    app.state.redis = await aioredis.from_url(settings.redis_url)
    app.state.http_client = httpx.AsyncClient()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.redis.close()
    await app.state.http_client.accept()

# --- Rate Limiting Middleware ---
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    api_key = request.headers.get("Authorization")
    if api_key:
        # Use API key in rate limit calculation
        key = f"rate_limit:{api_key}"
    else:
        # Use IP address as a fallback
        key = f"rate_limit:{request.client.host}"

    # Allow 10 requests per minute per key/IP
    current = await app.state.redis.get(key)
    if current and int(current) > 10:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    await app.state.redis.incr(key, 1)
    await app.state.redis.expire(key, 60)
    response = await call_next(request)
    return response

# --- Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    is_safe: bool

# --- Core Endpoint ---
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    # 1. Check cache first (for idempotent requests)
    cache_key = f"chat:{request.message}"
    cached_response = await app.state.redis.get(cache_key)
    if cached_response:
        return ChatResponse(**json.loads(cached_response))

    # 2. Safety Check - Call external Triton server
    safety_payload = {"text": request.message}
    try:
        safety_response = await app.state.http_client.post(
            f"{settings.safety_checker_url}/safety",
            json=safety_payload,
            timeout=5.0
        )
        safety_result = safety_response.json()
        if safety_result.get("is_unsafe"):
            raise HTTPException(status_code=400, detail="Query violates safety policy.")
    except httpx.RequestError:
        # Log the error but proceed? Block? Depends on safety requirements.
        # In production, you might want to fail closed.
        logging.error("Safety service unavailable")

    # 3. Call vLLM server for the LLM
    llm_payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": request.message}],
        "max_tokens": 500,
        "temperature": 0.7
    }
    try:
        llm_response = await app.state.http_client.post(
            f"{settings.vllm_url}/chat/completions",
            json=llm_payload,
            timeout=30.0
        )
        llm_result = llm_response.json()
        raw_response = llm_result['choices'][0]['message']['content']
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    # 4. Async Output Safety Check (in background to not block response)
    background_tasks.add_task(perform_async_safety_check, raw_response)

    # 5. Cache the response
    session_id = request.session_id or str(uuid.uuid4())
    final_response = ChatResponse(response=raw_response, session_id=session_id, is_safe=True)
    await app.state.redis.setex(cache_key, 300, json.dumps(final_response.dict())) # Cache for 5 minutes

    return final_response

async def perform_async_safety_check(text: str):
    # ... Async call to safety checker ...
    pass

import os
from abc import ABC, abstractmethod
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs_storage
import logging

logger = logging.getLogger(__name__)

# Define the interface all storage clients must implement
class StorageClient(ABC):
    @abstractmethod
    def download_file(self, bucket_name: str, source_blob_name: str, destination_file_name: str):
        pass

    @abstractmethod
    def upload_file(self, bucket_name: str, source_file_name: str, destination_blob_name: str):
        pass

# Concrete implementation for AWS S3 and S3-compatible APIs (like MinIO for on-prem)
class S3StorageClient(StorageClient):
    def __init__(self):
        # Credentials are loaded from environment variables via Boto3
        # (e.g., AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) or IAM role if on AWS.
        self.client = boto3.client('s3')

    def download_file(self, bucket_name: str, source_blob_name: str, destination_file_name: str):
        self.client.download_file(bucket_name, source_blob_name, destination_file_name)
        logger.info(f"Downloaded {source_blob_name} from S3 bucket {bucket_name}.")

    def upload_file(self, bucket_name: str, source_file_name: str, destination_blob_name: str):
        self.client.upload_file(source_file_name, bucket_name, destination_blob_name)
        logger.info(f"Uploaded {source_file_name} to S3 bucket {bucket_name} as {destination_blob_name}.")

# Concrete implementation for Azure Blob Storage
class AzureStorageClient(StorageClient):
    def __init__(self):
...         # Connection string is picked up from environment variable
...         connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
...         self.client = BlobServiceClient.from_connection_string(connect_str)
... 
...     def download_file(self, container_name: str, source_blob_name: str, destination_file_name: str):
...         blob_client = self.client.get_blob_client(container=container_name, blob=source_blob_name)
...         with open(destination_file_name, "wb") as download_file:
...             download_file.write(blob_client.download_blob().readall())
...         logger.info(f"Downloaded {source_blob_name} from Azure container {container_name}.")
... 
...     def upload_file(self, container_name: str, source_file_name: str, destination_blob_name: str):
...         blob_client = self.client.get_blob_client(container=container_name, blob=destination_blob_name)
...         with open(source_file_name, "rb") as data:
...             blob_client.upload_blob(data, overwrite=True)
...         logger.info(f"Uploaded {source_file_name} to Azure container {container_name} as {destination_blob_name}.")
... 
... # Factory function to create the appropriate client
... def get_storage_client() -> StorageClient:
...     provider = os.getenv('STORAGE_PROVIDER', 's3').lower() # Default to 's3'
... 
...     if provider == 's3':
...         return S3StorageClient()
...     elif provider == 'azure':
...         return AzureStorageClient()
...     elif provider == 'gcs':
...         # Implementation for Google Cloud Storage would go here
...         return GCSStorageClient()
...     else:
...         raise ValueError(f"Unsupported storage provider: {provider}")
... 
... # Example usage in another file (e.g., app.py):
... # storage_client = get_storage_client()



if __name__ == "__main__":
    import uvicorn

# orchestrator_async.py
import uuid
from redis import Redis
from celery import Celery

# Connect to message broker (Redis)
redis_client = Redis(host='redis-host', port=6379)
app = Celery('aegis_tasks', broker='redis://redis-host:6379/0')

@app.task
def process_safety_check(message_id: str, user_input: str):
    # ... safety logic here ...
    result = safety_model.check(user_input)
    redis_client.hset(f"result:{message_id}", "safety", result)
    return result

@app.task
def process_vision_model(message_id: str, image_url: str):
    # ... vision logic here ...
    result = vision_model.analyze(image_url)
    redis_client.hset(f"result:{message_id}", "vision", result)
    return result

@app.task
def process_llm(message_id: str, user_input: str, context: dict):
    # ... LLM logic here ...
    result = llm.generate(user_input, context)
    redis_client.hset(f"result:{message_id}", "llm", result)
    return result

# FastAPI Endpoint
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app_fastapi = FastAPI()

class AnalysisRequest(BaseModel):
    user_input: str
    image_url: str | None = None

@app_fastapi.post("/analyze")
async def create_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    message_id = str(uuid.uuid4())
    
    # Store initial request
    redis_client.hset(f"request:{message_id}", mapping=request.dict())
    
    # Kick off async tasks
    background_tasks.add_task(process_safety_check, message_id, request.user_input)
    if request.image_url:
        background_tasks.add_task(process_vision_model, message_id, request.image_url)
    background_tasks.add_task(process_llm, message_id, request.user_input, {})
    
    return {"message_id": message_id, "status": "processing"}

@app_fastapi.get("/results/{message_id}")
async def get_results(message_id: str):
    results = redis_client.hgetall(f"result:{message_id}")
    if not results:
        return {"status": "processing"}
    return {"status": "complete", "results": results}

# circuit_breaker.py
from redis import Redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from requests.exceptions import Timeout, ConnectionError

class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.redis = Redis(host='redis-host', port=6379)
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout

    def is_open(self, model_name: str) -> bool:
        failures = self.redis.get(f"cb_failures:{model_name}") or 0
        return int(failures) >= self.failure_threshold

    def record_failure(self, model_name: str):
        key = f"cb_failures:{model_name}"
        self.redis.incr(key)
        self.redis.expire(key, self.reset_timeout)

    def record_success(self, model_name: str):
        self.redis.delete(f"cb_failures:{model_name}")

# Decorator with retry and circuit breaker logic
def with_circuit_breaker(model_name):
    def decorator(func):
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((Timeout, ConnectionError))
        )
        def wrapper(*args, **kwargs):
            cb = CircuitBreaker()
            if cb.is_open(model_name):
                raise Exception(f"Circuit breaker for {model_name} is OPEN. Skipping call.")
            try:
                result = func(*args, **kwargs)
                cb.record_success(model_name)
                return result
            except (Timeout, ConnectionError) as e:
                cb.record_failure(model_name)
                raise e
        return wrapper
    return decorator

# Usage in a model task
@with_circuit_breaker("safety_model")
def call_safety_model(text: str):
    response = requests.post(
        "http://safety-model:8000/predict",
        json={"text": text},
        timeout=5.0  # Critical: Set a strict timeout
    )
    response.raise_for_status()
    return response.json()

# model_manager.py
import threading
from typing import Dict
from model_registry import ModelRegistry  # Hypothetical model catalog
import time

class ModelManager:
    def __init__(self):
        self.loaded_models: Dict[str, dict] = {}
        self.lock = threading.RLock()
        self.registry = ModelRegistry()

    def get_model(self, model_id: str):
        with self.lock:
            if model_id not in self.loaded_models:
                # Load the model into memory
                model_info = self.registry.get_model(model_id)
                print(f"Loading model {model_id} into GPU memory...")
                # ... Logic to load the model onto the GPU ...
                model = load_model_from_disk(model_info["path"])
                self.loaded_models[model_id] = {
                    "model": model,
                    "last_used": time.time(),
                    "load_count": 1
                }
            else:
                self.loaded_models[model_id]["last_used"] = time.time()
                self.loaded_models[model_id]["load_count"] += 1
            return self.loaded_models[model_id]["model"]

    def unload_idle_models(self, idle_time_sec=300):
        with self.lock:
            current_time = time.time()
            models_to_unload = []
            for model_id, info in self.loaded_models.items():
                if current_time - info["last_used"] > idle_time_sec:
                    models_to_unload.append(model_id)
            
            for model_id in models_to_unload:
                print(f"Unloading idle model: {model_id}")
                # ... Logic to gracefully unload model from GPU ...
                del self.loaded_models[model_id]

# Start a background thread to periodically unload idle models
def cleanup_thread(model_manager):
    while True:
        time.sleep(60)  # Check every minute
        model_manager.unload_idle_models()

model_manager = ModelManager()
threading.Thread(target=cleanup_thread, args=(model_manager,), daemon=True).start()

# semantic_cache.py
from redis import Redis
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticCache:
    def __init__(self, host='redis-host', port=6379, threshold=0.9):
        self.redis = Redis(host=host, port=port, decode_responses=True)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
        self.similarity_threshold = threshold

    def _get_key(self, text: str) -> str:
        embedding = self.encoder.encode(text)
        return f"embedding:{hash(embedding.tobytes())}"

    def get(self, user_input: str):
        input_key = self._get_key(user_input)
        # Get all keys for similar embeddings
        all_cache_keys = self.redis.keys("cache:*")
        for cache_key in all_cache_keys:
            stored_embedding_key = self.redis.hget(cache_key, 'embedding_key')
            if stored_embedding_key == input_key:
                return self.redis.hget(cache_key, 'response')
        return None

    def set(self, user_input: str, response: str):
        input_key = self._get_key(user_input)
        cache_key = f"cache:{uuid.uuid4()}"
        self.redis.hset(cache_key, mapping={
            'embedding_key': input_key,
            'response': response,
            'original_input': user_input
        })
        self.redis.expire(cache_key, 3600)  # Expire in 1 hour

# Usage in the endpoint
semantic_cache = SemanticCache()

@app_fastapi.post("/chat")
async def chat_endpoint(request: AnalysisRequest):
    # Check cache first
    cached_response = semantic_cache.get(request.user_input)
    if cached_response:
        return {"response": cached_response, "source": "cache"}
    
    # ... process request with models if not in cache ...
    semantic_cache.set(request.user_input, final_response)
    return {"response": final_response, "source": "models"}

# main.py (The Core FastAPI Server)
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from redis.asyncio import Redis, ConnectionPool
from celery import Celery
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, RetryError
from typing import Optional, Dict
import uuid
import time

# --- Models & Config ---
class InferenceRequest(BaseModel):
    prompt: str
    model: Optional[str] = "default"
    session_id: Optional[str] = None

# --- Lifespan Management (Resource Initialization/Cleanup) ---
redis_pool = ConnectionPool(host='localhost', port=6379, db=0, decode_responses=True)
celery_app = Celery('aegis_tasks', broker='redis://localhost:6379/0')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize resources (pools are lazy-loaded, but this is explicit)
    app.state.redis = Redis(connection_pool=redis_pool)
    app.state.http_client = httpx.AsyncClient() # HTTPX Connection Pool
    yield
    # Shutdown: Cleanup resources gracefully
    await app.state.redis.close()
    await app.state.http_client.accept()

app = FastAPI(lifespan=lifespan, title="Aegis Robust API")

# --- Dependencies (Dependency Injection) ---
async def get_redis() -> Redis:
    async with Redis(connection_pool=redis_pool) as redis:
        yield redis

# --- Core Services ---

# **1. Caching Service (Semantic Cache)**
class SemanticCache:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def get(self, prompt: str) -> Optional[str]:
        """Simple non-semantic key for demo. For a real semantic cache, you'd use a vector DB or generate a semantic key."""
        key = f"cache:prompt:{hash(prompt)}"
        return await self.redis.get(key)

    async def set(self, prompt: str, response: str, ttl: int = 3600):
        key = f"cache:prompt:{hash(prompt)}"
        await self.redis.setex(key, ttl, response)

# **2. Resilience & Circuit Breaker Service**
class CircuitBreaker:
    def __init__(self, redis: Redis, name: str, failure_threshold: int = 5, reset_timeout: int = 60):
        self.redis = redis
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout

    async def is_open(self) -> bool:
        key = f"circuit_breaker:{self.name}:failures"
        failures = await self.redis.get(key)
        return int(failures or 0) >= self.failure_threshold

    async def record_failure(self):
        key = f"circuit_breaker:{self.name}:failures"
        await self.redis.incr(key)
        await self.redis.expire(key, self.reset_timeout)

    async def record_success(self):
        key = f"circuit_breaker:{self.name}:failures"
        await self.redis.delete(key)

# **3. Async Model Client with Retry & Circuit Breaker**
class ModelClient:
    def __init__(self, http_client: httpx.AsyncClient, redis: Redis, model_base_url: str):
        self.http_client = http_client
        self.cb = CircuitBreaker(redis, name=f"model_{model_base_url}")
        self.model_base_url = model_base_url

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception((httpx.RequestError, HTTPException))
    )
    async def generate(self, prompt: str) -> str:
        # Check circuit breaker first
        if await self.cb.is_open():
            raise HTTPException(503, detail=f"Circuit breaker for {self.model_base_url} is open. Service unavailable.")

        try:
            response = await self.http_client.post(
                f"{self.model_base_url}/generate",
                json={"prompt": prompt},
                timeout=10.0 # Critical timeout
            )
            response.raise_for_status()
            await self.cb.record_success() # Success! Reset CB.
            return response.json()["text"]
        except httpx.RequestError as e:
            await self.cb.record_failure()
            raise e

# --- API Endpoints ---

@app.post("/generate")
async def generate_text(
    request: InferenceRequest,
    background_tasks: BackgroundTasks,
    redis: Redis = Depends(get_redis)
):
    # 1. Check Cache First
    cache = SemanticCache(redis)
    cached_response = await cache.get(request.prompt)
    if cached_response:
        return JSONResponse(
            content={"response": cached_response, "source": "cache"},
            status_code=200
        )

    # 2. Initiate Async Processing
    task_id = str(uuid.uuid4())
    # In a real app, you'd store the task_id and prompt in Redis for status tracking
    background_tasks.add_task(process_inference_task, request.prompt, task_id)

    return JSONResponse(
        content={"message": "Processing started asynchronously", "task_id": task_id, "status": "accepted"},
        status_code=202 # Accepted
    )

@app.get("/task/{task_id}")
async def get_task_status(task_id: str, redis: Redis = Depends(get_redis)):
    """Endpoint to poll for the result of an async task."""
    result = await redis.get(f"task:result:{task_id}")
    if not result:
        return {"status": "processing"}
    return {"status": "complete", "result": result}

# --- Celery Task (Async Worker) ---
@celery_app.task
def process_inference_task(prompt: str, task_id: str):
    """
    This task runs in the background Celery worker pool.
    It uses a synchronous Redis client and HTTP client.
    """
    # ... (Synchronous implementations of Cache, CircuitBreaker, and model calls would go here) ...
    # For simplicity, let's simulate work.
    time.sleep(5) # Simulate model inference time
    result = f"Processed: {prompt}"

    # Store the result in Redis so the /task endpoint can find it.
    # Use a synchronous redis client
    sync_redis = Redis(host='localhost', port=6379, db=0, decode_responses=True)
    sync_redis.setex(f"task:result:{task_id}", 300, result) # Result expires in 5 min
    sync_redis.close()

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# This is a concept for a core Aegis management API
class AegisModelManager:
    def __init__(self, model_hub_url=None, trusted_registry_only=True):
        # Prong 1: Sovereignty - Can run entirely offline
        self.offline_mode = model_hub_url is None
        if self.offline_mode:
            self.model_registry = LocalModelRegistry() # Uses a local, air-gapped catalog
        else:
            # Prong 2: Open Ecosystem - Connects to the curated hub
            self.model_registry = CertifiedModelHub(model_hub_url, trusted_registry_only)

    def deploy_model(self, model_id, version="latest", deployment_config):
        """Deploys a model from a trusted source with verified checksums."""
        # 1. Download from trusted source or verify local copy
        model_path, config = self.model_registry.get_model(model_id, version)

        # 2. Verify integrity (Prong 1: Security)
        if not self._verify_model_signature(model_path, config['signature']):
            raise SecurityException("Model integrity check failed! Potential tampering.")

        # 3. Load model into optimized runtime (e.g., vLLM, TensorRT)
        engine = self._load_model_engine(model_path, deployment_config)

        # 4. Register model in the orchestration router
        self.orchestrator.register_model(
            model_id=model_id,
            engine=engine,
            capabilities=config['capabilities'] # e.g., ["text-gen", "reasoning", "code"]
        )

        # 5. (Optional) Start A/B testing cohort if configured
        if deployment_config.get('ab_test_group'):
            self.ab_tester.add_model(model_id, deployment_config['ab_test_group'])

    def _verify_model_signature(self, model_path, expected_signature):
        # Uses cryptographic signing to ensure model bits haven't been altered
        import hashlib
        with open(model_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return file_hash == expected_signature

# Example Usage for a highly secure environment:
# This runs without any internet connection, using pre-vetted models.
manager = AegisModelManager() # No URL = offline mode
manager.deploy_model(
    model_id="meta-llama-3-70b-instruct",
    version="aegis-verified-v1.0", # Uses a certified, tested version
    deployment_config={"gpu_memory": "40GB", "ab_test_group": "group_a"}
)

# hallucination_guard.py
import logging
from typing import Dict, List, Optional, Tuple
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class HallucinationGuard:
    def __init__(self, web_search_api_key: Optional[str] = None):
        self.web_search_api_key = web_search_api_key
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()

    async def _call_model(self, prompt: str, model_endpoint: str) -> str:
        """Helper function to call a model endpoint."""
        try:
            response = await self.http_client.post(
                model_endpoint,
                json={"prompt": prompt},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json().get("text", "")
        except httpx.RequestError as e:
            self.logger.error(f"Error calling model at {model_endpoint}: {e}")
            raise

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _web_search_verify(self, claim: str) -> List[Dict]:
        """Use a web search API to verify a factual claim. Returns snippets."""
        if not self.web_search_api_key:
            return []

        try:
            # Example using a hypothetical SerpAPI or Bing Search API
            params = {
                "q": claim,
                "api_key": self.web_search_api_key,
                "num": 3  # Get top 3 results
            }
            response = await self.http_client.get(
                "https://serpapi.com/search",
                params=params
            )
            data = response.json()
            # Extract snippets from search results
            snippets = [result.get("snippet") for result in data.get("organic_results", []) if "snippet" in result]
            return snippets
        except Exception as e:
            self.logger.error(f"Web search verification failed: {e}")
            return []

    async def _check_against_sources(self, claim: str, sources: List[str]) -> float:
        """
        Simple heuristic to check if a claim is supported by source text.
        Returns a confidence score between 0 and 1.
        """
        if not sources:
            return 0.0  # No sources to verify against

        claim_lower = claim.lower()
        supporting_sources = 0

        for source in sources:
            if source and source.lower() in claim_lower:
                supporting_sources += 1
            # A more advanced version would use embedding similarity here

        return supporting_sources / len(sources)

    async def generate_with_guardrails(self, user_query: str, context: Optional[str] = None) -> Dict:
        """
        The main method to generate a response with anti-hallucination guardrails.
        Returns a dict containing the response, confidence, and sources.
        """

        # Step 1: Generate the initial response with a primary model
        initial_response = await self._call_model(
            f"Answer the following query based on the provided context. If the answer isn't in the context, say 'I don't know'.\nQuery: {user_query}\nContext: {context}",
            "http://llm-model-server:8000/generate"
        )

        # Step 2: Perform self-critique
        critique_prompt = f"""
        You are a fact-checker. Analyze the following response for inaccuracies or unsupported claims.

        Original Query: {user_query}
        Response to Critique: {initial_response}

        List any potential hallucinations or inaccuracies. If everything seems accurate, say 'ALL_CHECKS_PASSED'.
        """
        critique = await self._call_model(critique_prompt, "http://llm-model-server:8000/generate")

        # Step 3: If critique found issues, correct the response
        if "ALL_CHECKS_PASSED" not in critique:
            self.logger.warning(f"Self-critique flagged response: {critique}")
            correction_prompt = f"""
            The previous response was flagged for potential inaccuracies: {critique}
            Original Query: {user_query}
            Please rewrite the response to be accurate and avoid any unsupported claims.
            """
            initial_response = await self._call_model(correction_prompt, "http://llm-model-server:8000/generate")

        # Step 4: For factual claims, perform web search verification
        verification_snippets = []
        confidence = 0.5  # Default neutral confidence

        # Extract a factual claim from the response (simplified)
        if context is None and "is" in initial_response and "I don't know" not in initial_response:
            verification_snippets = await self._web_search_verify(initial_response)
            confidence = await self._check_against_sources(initial_response, verification_snippets)

        # Step 5: Final truthfulness check with a dedicated guardrail model
        guardrail_prompt = f"""
        Is the following response supported by the context and factually accurate?
        Respond only with 'YES' or 'NO' and a confidence score between 0-1.

        Query: {user_query}
        Response: {initial_response}
        Context: {context}
        """
        guardrail_check = await self._call_model(guardrail_prompt, "http://guardrail-model:8001/check")
        guardrail_result = guardrail_check.strip().split()

        if guardrail_result and guardrail_result[0] == 'NO':
            self.logger.error("Guardrail model blocked the response.")
            initial_response = "I cannot provide a verified answer to that question at this time."

        # Compile the final output with metadata
        return {
            "response": initial_response,
            "confidence": confidence,
            "sources": verification_snippets[:2],  # Return top 2 sources
            "was_verified": len(verification_snippets) > 0,
            "guardrail_result": guardrail_check
        }

from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from hallucination_guard import HallucinationGuard
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.hallucination_guard = HallucinationGuard(
        web_search_api_key=os.getenv("WEB_SEARCH_API_KEY")
    )
    yield
    # Shutdown
    await app.state.hallucination_guard.http_client.aclose()

app = FastAPI(lifespan=lifespan)

@app.post("/query")
async def safe_query_endpoint(user_query: str, context: str = None):
    """
    Endpoint for generating responses with anti-hallucination guardrails.
    """
    try:
        async with app.state.hallucination_guard as guard:
            result = await guard.generate_with_guardrails(user_query, context)
        
        # Apply a confidence threshold
        if result["confidence"] < 0.7 and result["was_verified"]:
            result["response"] = "I'm not entirely confident about this. Based on my search, I found: " + result["response"]
        
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


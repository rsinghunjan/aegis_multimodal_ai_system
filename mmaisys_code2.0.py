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
...                 result = self.classifier(text)[0]\
...                 # Check if the harmful label is above threshold\
...                 if result['label'] == 'harmful' and result['score'] > threshold:\
...                     return True\
                return False\
\
        except Exception as e:\
            logger.error(f"Safety classification error: \{e\}. Blocking due to error.")\
            return True  # Block on error to be safe\
\
# Global instance\
safety_checker = SafetyChecker()\
\
# app.py\
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks\
from fastapi.middleware.cors import CORSMiddleware\
from pydantic import BaseModel\
import httpx\
from typing import Optional, List\
import logging\
import uuid\
import time\

# Import our modules
from safety_checker import safety_checker
# ... (Import the MultimodalAISystem class from our previous work, now refactored)

# Initialize app and our AI system\
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


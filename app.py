import os
import logging
import weaviate
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from weaviate.auth import AuthApiKey
from spellchecker import SpellChecker
from difflib import get_close_matches

# -------------------
# Setup logging
# -------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------
# Load environment
# -------------------
load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------
# Initialize clients
# -------------------
auth_config = AuthApiKey(api_key=WEAVIATE_API_KEY)
weaviate_client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=auth_config,
    timeout_config=(5, 30)
)
openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=30)
spell = SpellChecker()

# -------------------
# FastAPI app
# -------------------
app = FastAPI(title="Product Search API", version="5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, Dict[str, Any]] = {}

# -------------------
# Predefined options
# -------------------
CATEGORIES = ["sunscreen", "toner", "moisturizer", "cleanser"]
SKIN_TYPE_MAP = {
    "oily": "oily",
    "dry": "dry",
    "dehydrated": "dry",
    "sensitive": "sensitive",
    "normal": "combination",
    "combination": "combination"
}
CONCERNS = ["acne", "pigmentation", "dark spots", "sensitivity", "blemishes", "redness"]

# -------------------
# Helper functions
# -------------------
def correct_query(query: str) -> str:
    words = query.split()
    corrected_words = []
    for word in words:
        if word not in spell:
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

def embed_query(query: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding

def retrieve_products(query: str, top_k: int = 10):
    embedding = embed_query(query)
    result = (
        weaviate_client.query
        .get("Product", ["product_id", "name", "brand", "category", "price", "currency", "description", "skin_type", "concerns"])
        .with_near_vector({"vector": embedding})
        .with_additional(["certainty"])
        .with_limit(top_k)
        .do()
    )
    products = result.get("data", {}).get("Get", {}).get("Product", [])
    # Reduce certainty filter to 0.0 so nothing is dropped prematurely
    filtered = [p for p in products if p["_additional"]["certainty"] >= 0.0]
    return filtered

def product_matches(product, collected):
    # Skin type match (partial/fuzzy)
    if "skin_type" in collected:
        product_skin = [s.lower() for s in product.get("skin_type", [])]
        if collected["skin_type"].lower() not in product_skin:
            return False
    # Category match
    if product["category"].lower() != collected["category"].lower():
        return False
    return True

def rag_response(user_query: str, retrieved_products: List[dict]) -> str:
    context = "\n".join(
        [f"- {p['name']} ({p['brand']}), {p['currency']} {p['price']}: {p['description']}" for p in retrieved_products]
    )
    prompt = f"""
You are a helpful skincare assistant.
User asked: "{user_query}"

Here are some relevant products from the database:
{context}

Provide a short, clear recommendation (2-3 sentences max).
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise skincare shopping assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def parse_budget(budget_text: str):
    budget_text = budget_text.lower().replace("$", "").strip()
    numbers = [int(n) for n in re.findall(r"\d+", budget_text)]
    
    if "under" in budget_text or "below" in budget_text:
        return {"max": numbers[0]} if numbers else None
    elif "above" in budget_text or "over" in budget_text:
        return {"min": numbers[0]} if numbers else None
    elif "between" in budget_text and len(numbers) >= 2:
        return {"min": numbers[0], "max": numbers[1]}
    elif numbers:  # just a single number means "up to"
        return {"max": numbers[0]}
    else:
        return None
# -------------------
# Request / Response Models
# -------------------
class Product(BaseModel):
    product_id: str
    name: str
    brand: str
    category: str
    price: float
    currency: str
    description: str
    certainty: float

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    collected_info: Dict[str, Any]
    ready_to_search: bool
    products: Optional[List[Product]] = None
    rag_answer: Optional[str] = None

# -------------------
# Chat Endpoint
# -------------------
@app.post("/chat", response_model=ChatResponse)
def chat_with_user(body: ChatRequest):
    session = sessions.get(body.session_id, {"collected_info": {}, "history": []})
    collected = session["collected_info"]
    user_msg = body.message.lower().strip()
    session["history"].append({"user": body.message})

    if not collected:
        collected["awaiting_category"] = True
        sessions[body.session_id] = session
        return ChatResponse(
            reply=f"Hi! Let’s find the right skincare for you. First, what type of product are you looking for? Choose one of: {', '.join(CATEGORIES)}",
            collected_info=collected,
            ready_to_search=False
        )
    # Step 1: Ask for category
    if collected.get("awaiting_category"):
        chosen_category = None
        # 🔹 First check for exact match
        for cat in CATEGORIES:
            if cat.lower() == user_msg:
                chosen_category = cat
                break

        # 🔹 If not exact, try fuzzy match (handles typos like "moistrizer")
        if not chosen_category:
            matches = get_close_matches(user_msg, CATEGORIES, n=1, cutoff=0.5)
            if matches:
                chosen_category = matches[0]

        # 🔹 If still nothing, ask again
        if not chosen_category:
            logger.debug(f"Unrecognized category input: {user_msg}")
            return ChatResponse(
                reply=f"Sorry, I didn’t understand that category. Please choose one of: {', '.join(CATEGORIES)}",
                collected_info=collected,
                ready_to_search=False
            )

        # ✅ Save chosen category and continue
        collected["category"] = chosen_category
        collected.pop("awaiting_category", None)
        sessions[body.session_id] = session
        return ChatResponse(
            reply=f"Great! You chose '{chosen_category}'. What’s your skin type? (oily, dry, sensitive, combination)",
            collected_info=collected,
            ready_to_search=False
        )

    # Step 3: Capture skin type
    if "skin_type" not in collected:
        found = False
        for k, v in SKIN_TYPE_MAP.items():
            if k in user_msg:
                collected["skin_type"] = v
                found = True
                break

        if not found:
            matches = get_close_matches(user_msg, SKIN_TYPE_MAP.keys(), n=1, cutoff=0.5)  # stricter cutoff
            if matches:
                collected["skin_type"] = SKIN_TYPE_MAP[matches[0]]
            else:
                logger.debug(f"Unrecognized skin type input: {user_msg}")
                return ChatResponse(
                    reply="Sorry, I didn’t understand your skin type. Please choose one of: oily, dry, sensitive, combination.",
                    collected_info=collected,
                    ready_to_search=False
                )

        sessions[body.session_id] = session
        return ChatResponse(
            reply=f"Got it, your skin type is '{collected['skin_type']}'. What’s your budget? (or type 'no preference')",
            collected_info=collected,
            ready_to_search=False
        )

    # Step 4: Capture budget
    if "budget" not in collected:
        collected["budget"] = body.message if body.message else "no preference"
        sessions[body.session_id] = session
        return ChatResponse(
            reply="Do you have any specific skin concerns? (acne, pigmentation, dark spots, sensitivity)'",
            collected_info=collected,
            ready_to_search=False
        )

    # Step 5: Capture concerns
    if "concerns" not in collected:
        selected = []
        for concern in CONCERNS:
            if concern in user_msg:
                selected.append(concern)
        if not selected:
            matches = get_close_matches(user_msg.split(), CONCERNS, n=2, cutoff=0.5)
            selected.extend(matches)
        collected["concerns"] = selected if selected else []
        sessions[body.session_id] = session

    # Step 6: Retrieve and filter products
    query_text = f"best {collected['category']} for {collected['skin_type']} skin"
    if collected.get("concerns"):
        query_text += f" with concerns {', '.join(collected['concerns'])}"
    if collected.get("budget") and collected["budget"].lower() != "no preference":
        query_text += f" within {collected['budget']}"

    all_products = retrieve_products(query_text, top_k=10)
    products = [p for p in all_products if product_matches(p, collected)]

    budget_range = parse_budget(collected.get("budget", ""))
    if budget_range:
        if "min" in budget_range:
            products = [p for p in products if p["price"] >= budget_range["min"]]
        if "max" in budget_range:
            products = [p for p in products if p["price"] <= budget_range["max"]]

    if not products:
        products = all_products  # relax budget filter
        fallback_message = "I couldn’t find anything exactly in that price range, but here are some close alternatives."
    else:
        fallback_message = None

    products.sort(key=lambda x: x["_additional"]["certainty"], reverse=True)
    products = products[:5]

    rag_answer = rag_response(query_text, products)

    product_objs = [
        Product(
            product_id=p["product_id"],
            name=p["name"],
            brand=p["brand"],
            category=p["category"],
            price=p["price"],
            currency=p["currency"],
            description=p["description"],
            certainty=p["_additional"]["certainty"]
        )
        for p in products
    ]

    sessions[body.session_id] = session
    reply_text = f"{fallback_message} {rag_answer}" if fallback_message else f"I found some great options for you. {rag_answer}"
    return ChatResponse(
        reply=reply_text,
        collected_info=collected,
        ready_to_search=True,
        products=product_objs,
        rag_answer=rag_answer
    )

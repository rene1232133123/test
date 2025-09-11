import os
import weaviate
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from weaviate.auth import AuthApiKey
from spellchecker import SpellChecker
import uuid

load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

auth_config = AuthApiKey(api_key=WEAVIATE_API_KEY)
weaviate_client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=auth_config)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
spell = SpellChecker()

app = FastAPI(title="Product Search API", version="4.0")
sessions: Dict[str, Dict[str, Any]] = {}

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

def retrieve_products(query: str, top_k: int = 5):
    embedding = embed_query(query)
    result = (
        weaviate_client.query
        .get("Product", ["product_id", "name", "brand", "category", "price", "currency", "description"])
        .with_near_vector({"vector": embedding})
        .with_additional(["certainty"])
        .with_limit(top_k)
        .do()
    )
    products = result.get("data", {}).get("Get", {}).get("Product", [])
    return [p for p in products if p["_additional"]["certainty"] > 0.7]

def rag_response(user_query: str, retrieved_products: List[dict]) -> str:
    context = "\n".join(
        [f"- {p['name']} ({p['brand']}), {p['currency']} {p['price']}: {p['description']}"
         for p in retrieved_products]
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

def gpt_chit_chat(message: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly skincare assistant who can chat naturally."},
            {"role": "user", "content": message}
        ],
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class Product(BaseModel):
    product_id: str
    name: str
    brand: str
    category: str
    price: float
    currency: str
    description: str
    certainty: float

class SearchResponse(BaseModel):
    corrected_query: str
    products: List[Product]
    rag_answer: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    collected_info: Dict[str, Any]
    ready_to_search: bool
    products: Optional[List[Product]] = None
    rag_answer: Optional[str] = None

@app.post("/search", response_model=SearchResponse)
def search_products(body: SearchRequest):
    corrected_query = correct_query(body.query)
    products = retrieve_products(corrected_query, top_k=body.top_k)
    rag_answer = rag_response(corrected_query, products)

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

    return SearchResponse(
        corrected_query=corrected_query,
        products=product_objs,
        rag_answer=rag_answer
    )

@app.post("/chat", response_model=ChatResponse)
def chat_with_user(body: ChatRequest):
    # Create session if not exists
    session = sessions.get(body.session_id, {"collected_info": {}, "history": []})
    collected = session["collected_info"]
    user_msg = body.message.lower()
    session["history"].append({"user": body.message})

    structured_keywords = [
        "sunscreen", "toner", "moisturizer", "cleanser",
        "oily", "dry", "sensitive", "combination",
        "$", "dollar", "under", "less than", "between",
        "acne", "pigmentation", "dark spots", "sensitivity",
        "none", "no"
    ]

    if not any(kw in user_msg for kw in structured_keywords):
        reply_text = gpt_chit_chat(body.message)
        sessions[body.session_id] = session
        return ChatResponse(
            reply=reply_text,
            collected_info=collected,
            ready_to_search=False
        )

    if "category" not in collected:
        for kw in ["sunscreen", "toner", "moisturizer", "cleanser"]:
            if kw in user_msg:
                collected["category"] = kw
                break
        if "category" not in collected:
            sessions[body.session_id] = session
            return ChatResponse(
                reply="What kind of product are you looking for? (e.g., sunscreen, toner, moisturizer)",
                collected_info=collected,
                ready_to_search=False
            )

    if "skin_type" not in collected:
        for kw in ["oily", "dry", "sensitive", "combination"]:
            if kw in user_msg:
                collected["skin_type"] = kw
                break
        if "skin_type" not in collected:
            if any(kw in user_msg for kw in ["no", "not sure", "don't know", "unsure"]):
                collected["skin_type"] = "unspecified"
            else:
                sessions[body.session_id] = session
                return ChatResponse(
                    reply="Can you tell me your skin type? (oily, dry, sensitive, etc.)",
                    collected_info=collected,
                    ready_to_search=False
                )

    if "budget" not in collected:
        if any(sym in user_msg for sym in ["$", "dollar", "under", "less than", "between"]):
            collected["budget"] = body.message
        elif any(kw in user_msg for kw in ["no", "not sure", "don't want to say", "i don't have a range", "no budget"]):
            collected["budget"] = "no preference"
        else:
            sessions[body.session_id] = session
            return ChatResponse(
                reply="What's your budget range? (or type 'no preference')",
                collected_info=collected,
                ready_to_search=False
            )

    if "concerns" not in collected:
        if any(kw in user_msg for kw in ["acne", "pigmentation", "dark spots", "sensitivity"]):
            collected["concerns"] = [body.message.lower()]
        elif any(kw in user_msg for kw in ["none", "no", "not really"]):
            collected["concerns"] = []
        else:
            sessions[body.session_id] = session
            return ChatResponse(
                reply="Do you have any specific skin concerns (acne, pigmentation, sensitivity)? (or type 'none')",
                collected_info=collected,
                ready_to_search=False
            )

    query_text = f"best {collected['category']} for {collected['skin_type']} skin"
    if collected.get("concerns"):
        query_text += f" with concerns {', '.join(collected['concerns'])}"
    if collected.get("budget") and collected["budget"] != "no preference":
        query_text += f" within {collected['budget']}"

    products = retrieve_products(query_text, top_k=5)
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
    return ChatResponse(
        reply=f"I found some great options for you. {rag_answer}",
        collected_info=collected,
        ready_to_search=True,
        products=product_objs,
        rag_answer=rag_answer
    )

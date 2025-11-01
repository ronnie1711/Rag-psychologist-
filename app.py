from fastapi import FastAPI, Request
import faiss, pickle, numpy as np, os
from openai import OpenAI


app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


index = faiss.read_index("data/faiss_index_new.index")
with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


RAG_TRIGGERS = [
    "how", "what can i do", "what should i do", "how can i",
    "techniques", "methods", "ways to", "exercises", "strategies",
    "what are the ways", "how to deal", "how to fix", "what helps",
    "what would you suggest", "how do i", "help me with", "tips for"
]

def needs_rag(text):
    return any(kw in text.lower() for kw in RAG_TRIGGERS)

def embed(text):
    emb = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(emb.data[0].embedding, dtype="float32").reshape(1, -1)

def search_chunks(query, top_k=4):
    query_emb = embed(query)
    D, I = index.search(query_emb, top_k)
    return [chunks[i] for i in I[0] if 0 <= i < len(chunks)]


sessions = {}  # структура: { session_id: {history: [...], memory_summary: "..."} }

def get_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = {"history": [], "memory_summary": ""}
    return sessions[session_id]


def rag_psychologist(user_input, session_id="default"):
    session = get_session(session_id)
    history = session["history"]
    memory_summary = session["memory_summary"]

    use_rag = needs_rag(user_input)
    retrieved_texts = search_chunks(user_input, top_k=4) if use_rag else []
    context = "\n\n".join(retrieved_texts)

    dialogue_context = "\n".join([f"{role}: {msg}" for role, msg in history[-5:]])
    memory_section = f"\n\nMemory summary:\n{memory_summary}" if memory_summary else ""

    prompt = f"""
You are an empathetic AI psychologist.
Always respond in English, even if the user speaks another language.
Your tone is calm, compassionate, reflective, and professional.

User message: {user_input}
{memory_section}

Recent conversation:
{dialogue_context}

{"Relevant psychological insights:\n" + context if use_rag else ""}
Respond in English with empathy and psychological insight.
Use techniques or exercises only if relevant.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an empathetic AI psychologist who gives thoughtful, professional responses in English."},
            {"role": "user", "content": prompt}
        ]
    )
    reply = completion.choices[0].message.content.strip()

  
    history.append(("User", user_input))
    history.append(("AI", reply))
    session["history"] = history
    sessions[session_id] = session

    return reply


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default")  # можно передавать с фронта
    response = rag_psychologist(user_message, session_id=session_id)
    return {"response": response, "session_id": session_id}



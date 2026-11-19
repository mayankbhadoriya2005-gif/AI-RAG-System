
import time
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from app.observability import trace
from app.aws import embed, answer, answer_from_history
from app.vector import search
from app.config import settings
from app.guardrails import check_input, check_output


class State(TypedDict, total=False):
    question: str
    principal: Any
    passages: list
    answer: str
    sources: list
    status: str
    history: list


# -----------------------------------
# 1. Detect conversation-related questions
# -----------------------------------

def is_memory_question(question: str) -> bool:

    q = question.lower().strip()

    patterns = [
        "what were we discussing",
        "what were we talking about",
        "what did we discuss",
        "what did we talk about",
        "what was discussed earlier",
        "what were we discussing earlier",
        "what did i ask earlier",
        "what was my previous question",
        "what did you tell me earlier",
        "summarize our conversation",
        "summarize what we discussed",
        "summarize everything we discussed",
        "what have we discussed",
        "remind me what we discussed",
        "what was our previous discussion"
    ]

    return any(pattern in q for pattern in patterns)


# -----------------------------------
# 2. Input Guardrails
# -----------------------------------

@trace("guard")
def guard(s):

    start = time.perf_counter()

    result = {
        "question": check_input(s["question"])
    }

    print(
        f"[TIMING] Input guard: {time.perf_counter() - start:.2f}s",
        flush=True
    )

    return result


# -----------------------------------
# 3. Document Retrieval
# -----------------------------------

@trace("retrieve")
def retrieve(s):

    start = time.perf_counter()

    p = s["principal"]

    prior_questions = [
        m["content"]
        for m in s.get("history", [])
        if m["role"] == "user"
    ]

    query = (
        prior_questions[-1] + "\nFollow-up: " + s["question"]
        if prior_questions
        else s["question"]
    )

    # Generate query embedding
    t1 = time.perf_counter()

    embedding = embed(query)

    print(
        f"[TIMING] Embedding: {time.perf_counter() - t1:.2f}s",
        flush=True
    )

    # Search ChromaDB
    t2 = time.perf_counter()

    hits = search(
        p.tenant,
        embedding,
        p.groups,
        p.subject
    )

    print(
        f"[TIMING] ChromaDB search: {time.perf_counter() - t2:.2f}s",
        flush=True
    )

    # Filter irrelevant passages
    hits = [
        h for h in hits
        if h["score"] >= settings().min_relevance
    ]

    sources = [
        {
            "index": i + 1,
            "document_id": h["metadata"].get("document_id", ""),
            "title": h["metadata"].get("title", ""),
            "page": h["metadata"].get("page", 1),
            "excerpt": h["metadata"].get("text", "")[:320]
        }
        for i, h in enumerate(hits)
    ]

    print(
        f"[TIMING] Total retrieval: {time.perf_counter() - start:.2f}s",
        flush=True
    )

    return {
        "passages": hits,
        "sources": sources
    }


# -----------------------------------
# 4. Route After Input Guardrails
# -----------------------------------

def route_after_guard(s):

    if s.get("history") and is_memory_question(s["question"]):
        return "memory"

    return "retrieve"


# -----------------------------------
# 5. Route After Document Retrieval
# -----------------------------------

def route_after_retrieval(s):

    if s.get("passages"):
        return "generate"

    if s.get("history"):
        return "memory"

    return "no_evidence"


# -----------------------------------
# 6. Memory Node
# -----------------------------------

@trace("memory")
def memory_node(s):

    start = time.perf_counter()

    response = answer_from_history(
        s["question"],
        s.get("history", [])
    )

    print(
        f"[TIMING] Memory answer: {time.perf_counter() - start:.2f}s",
        flush=True
    )

    return {
        "answer": response,
        "status": "memory_answered",
        "sources": []
    }


# -----------------------------------
# 7. No Evidence Node
# -----------------------------------

@trace("no_evidence")
def no_evidence(s):

    return {
        "answer": (
            "I could not find supporting evidence in "
            "documents you are authorized to access."
        ),
        "status": "no_evidence",
        "sources": []
    }


# -----------------------------------
# 8. RAG Answer Generation
# -----------------------------------

@trace("generate")
def generate(s):

    start = time.perf_counter()

    evidence = "\n\n".join(
        f'[{i + 1}] {h["metadata"].get("title", "Document")}, '
        f'page {h["metadata"].get("page", 1)}: '
        f'{h["metadata"].get("text", "")}'
        for i, h in enumerate(s["passages"])
    )

    t1 = time.perf_counter()

    response = answer(
        s["question"],
        evidence,
        s.get("history", [])
    )

    print(
        f"[TIMING] Gemini answer: {time.perf_counter() - t1:.2f}s",
        flush=True
    )

    print(
        f"[TIMING] Total generation: {time.perf_counter() - start:.2f}s",
        flush=True
    )

    return {
        "answer": response,
        "status": "answered"
    }


# -----------------------------------
# 9. Output Validation
# -----------------------------------

@trace("validate")
def validate(s):

    start = time.perf_counter()

    result = {
        "answer": check_output(s["answer"])
    }

    print(
        f"[TIMING] Output validation: {time.perf_counter() - start:.2f}s",
        flush=True
    )

    return result


# -----------------------------------
# 10. Build LangGraph Workflow
# -----------------------------------

def make_graph():

    g = StateGraph(State)

    nodes = [
        ("guard", guard),
        ("retrieve", retrieve),
        ("memory", memory_node),
        ("no_evidence", no_evidence),
        ("generate", generate),
        ("validate", validate)
    ]

    for name, fn in nodes:
        g.add_node(name, fn)

    g.add_edge(START, "guard")

    g.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            "memory": "memory",
            "retrieve": "retrieve"
        }
    )

    g.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate": "generate",
            "memory": "memory",
            "no_evidence": "no_evidence"
        }
    )

    g.add_edge("generate", "validate")
    g.add_edge("memory", "validate")

    g.add_edge("validate", END)
    g.add_edge("no_evidence", END)

    return g.compile()


# Compile graph
graph = make_graph()


# -----------------------------------
# 11. Main Chat Function
# -----------------------------------

@trace("chat")
def chat(p, question, history=None):

    start = time.perf_counter()

    r = graph.invoke({
        "principal": p,
        "question": question,
        "history": history or []
    })

    print(
        f"[TIMING] Complete chat pipeline: {time.perf_counter() - start:.2f}s",
        flush=True
    )

    return {
        "answer": r["answer"],
        "sources": r.get("sources", []),
        "status": r.get("status", "answered")
    }

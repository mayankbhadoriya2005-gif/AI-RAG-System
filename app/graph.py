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
    return {
        "question": check_input(s["question"])
    }


# -----------------------------------
# 3. Document Retrieval
# -----------------------------------

@trace("retrieve")
def retrieve(s):

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

    hits = search(
        p.tenant,
        embed(query),
        p.groups,
        p.subject
    )

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

    response = answer_from_history(
        s["question"],
        s.get("history", [])
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

    evidence = "\n\n".join(
        f'[{i + 1}] {h["metadata"].get("title", "Document")}, '
        f'page {h["metadata"].get("page", 1)}: '
        f'{h["metadata"].get("text", "")}'
        for i, h in enumerate(s["passages"])
    )

    response = answer(
        s["question"],
        evidence,
        s.get("history", [])
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

    return {
        "answer": check_output(s["answer"])
    }


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

    # Starting point
    g.add_edge(START, "guard")

    # Memory intent detection
    g.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            "memory": "memory",
            "retrieve": "retrieve"
        }
    )

    # Document retrieval routing
    g.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate": "generate",
            "memory": "memory",
            "no_evidence": "no_evidence"
        }
    )

    # Answer validation
    g.add_edge("generate", "validate")
    g.add_edge("memory", "validate")

    # End nodes
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

    r = graph.invoke({
        "principal": p,
        "question": question,
        "history": history or []
    })

    return {
        "answer": r["answer"],
        "sources": r.get("sources", []),
        "status": r.get("status", "answered")
    }

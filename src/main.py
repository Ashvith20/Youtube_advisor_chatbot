# main.py

from utils.chroma import ChromaDBClient
from utils.generator import generate_response
import argparse

def build_prompt(user_query: str, documents: list, metadatas: list) -> str:
    context_parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        context_parts.append(
            f"Snippet {i}:\n"
            f"{doc}\n"
            f"(source: {meta.get('source', 'unknown')}, start: {meta.get('start', 0)}, end: {meta.get('end', 0)})\n"
        )

    context = "\n---\n".join(context_parts)

    return f"""You are an assistant helping answer questions based only on the given context.
Do not use any external knowledge.

Context:
{context}

Question: {user_query}

Answer (based strictly on the above snippets, and include references if helpful):
"""

def main():
    parser = argparse.ArgumentParser(description="Ask a question with grounded answers.")
    parser.add_argument("question", type=str, help="Your prompt or question")
    args = parser.parse_args()

    chroma_db = ChromaDBClient()
    results = chroma_db.query(args.question, top_k=3)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    prompt = build_prompt(args.question, documents, metadatas)
    response = generate_response(prompt)

    print("\n Answer:\n")
    print(response)

if __name__ == "__main__":
    main()

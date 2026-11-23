from google import genai
from search_embeddings import load_chunks, search, DOCS_FOLDER

# The client reads the GEMINI_API_KEY environment variable automatically.
client = genai.Client()


def answer_question(question):
    """Retrieve the best sentences, then ask Gemini to answer from them."""
    chunks = load_chunks(DOCS_FOLDER)
    results = search(question, chunks, top_k=2)

    # Each result is (score, source_file, sentence). We only need the sentences.
    context = "\n".join(f"- {text}" for _, _, text in results)

    prompt = f"""Answer the question using ONLY the passages below.
If the passages do not contain the answer, say "I don't know from these documents."

Passages:
{context}

Question: {question}"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )
    return response.text, results


if __name__ == "__main__":
    question = input("Ask a question: ")
    answer, sources = answer_question(question)

    print(f"\nAnswer:\n{answer}")
    print("\nSources used:")
    for score, source, text in sources:
        print(f"  [{score:.2f}] {source}: {text}")
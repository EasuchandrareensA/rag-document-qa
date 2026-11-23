import os
from sentence_transformers import SentenceTransformer, util

DOCS_FOLDER = "docs"

# Downloads a small model the first time you run it (about 90 MB).
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_chunks(folder):
    """Return a list of (source_filename, sentence) pairs from every .txt file."""
    chunks = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                for sentence in f.read().split("."):
                    sentence = sentence.strip()
                    if sentence:
                        chunks.append((filename, sentence))
    return chunks


def search(question, chunks, top_k=2):
    """Return the top_k chunks whose meaning is closest to the question."""
    sentences = [text for _, text in chunks]

    # Turn the question and every sentence into a vector (a list of numbers).
    chunk_vectors = model.encode(sentences, convert_to_tensor=True)
    question_vector = model.encode(question, convert_to_tensor=True)

    # Cosine similarity: 1.0 means same meaning, values near 0 mean unrelated.
    scores = util.cos_sim(question_vector, chunk_vectors)[0]

    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    results = []
    for i in ranked[:top_k]:
        results.append((float(scores[i]), chunks[i][0], chunks[i][1]))
    return results


if __name__ == "__main__":
    chunks = load_chunks(DOCS_FOLDER)
    question = input("Ask a question: ")

    for score, source, text in search(question, chunks):
        print(f"\n[{score:.2f}] {source}: {text}")
import os

DOCS_FOLDER = "docs"


def load_documents(folder):
    """Read every .txt file in the folder and return a dict of {filename: text}."""
    documents = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                documents[filename] = f.read()
    return documents


def split_into_chunks(text):
    """Split text into sentences. Each sentence becomes one chunk."""
    chunks = []
    for sentence in text.split("."):
        sentence = sentence.strip()
        if sentence:
            chunks.append(sentence)
    return chunks


STOP_WORDS = {"a", "an", "the", "is", "in", "of", "to", "do", "not", "who", "can", "and", "with", "for", "on", "by", "it", "i", "my"}


def normalize(word):
    """Lowercase the word, remove punctuation, and drop a trailing 's'."""
    word = word.lower().strip("?.,!:;")
    if word.endswith("s") and len(word) > 3:
        word = word[:-1]
    return word


def score(question, chunk):
    """Count shared meaningful words between the question and the chunk."""
    question_words = {normalize(w) for w in question.split()} - STOP_WORDS
    chunk_words = {normalize(w) for w in chunk.split()} - STOP_WORDS
    return len(question_words & chunk_words)


def find_best_chunk(question, documents):
    """Check every chunk in every document and return the best match."""
    best_score = -1
    best_chunk = None
    best_source = None

    for filename, text in documents.items():
        for chunk in split_into_chunks(text):
            current = score(question, chunk)
            if current > best_score:
                best_score = current
                best_chunk = chunk
                best_source = filename

    return best_source, best_chunk, best_score


if __name__ == "__main__":
    documents = load_documents(DOCS_FOLDER)
    question = input("Ask a question: ")
    source, chunk, points = find_best_chunk(question, documents)

    print(f"\nBest match from: {source}")
    print(f"Matching words: {points}")
    print(f"Text: {chunk}")
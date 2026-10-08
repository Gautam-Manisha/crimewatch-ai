from pathlib import Path
import pickle

import faiss
from sentence_transformers import SentenceTransformer


KNOWLEDGE_FILE = Path(
    "data/knowledge/crimewatch_knowledge.txt"
)

VECTOR_DB_DIR = Path(
    "data/knowledge/vector_db"
)

MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents():

    text = KNOWLEDGE_FILE.read_text(
        encoding="utf-8"
    )

    sections = [
        section.strip()
        for section in text.split("\n\n\n")
        if section.strip()
    ]

    return sections


def build_vector_database():

    print("=" * 60)
    print("BUILDING CRIMEWATCH VECTOR DATABASE")
    print("=" * 60)

    documents = load_documents()

    print("Documents:", len(documents))

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Creating embeddings...")

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    dimension = embeddings.shape[1]

    print("Embedding dimension:", dimension)

    # FAISS similarity index
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings.astype("float32"))

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(
            VECTOR_DB_DIR / "crimewatch.index"
        )
    )

    with open(
        VECTOR_DB_DIR / "documents.pkl",
        "wb",
    ) as file:

        pickle.dump(
            documents,
            file
        )

    print()
    print("Vector database created successfully.")
    print(
        "Index:",
        VECTOR_DB_DIR / "crimewatch.index"
    )
    print(
        "Documents:",
        VECTOR_DB_DIR / "documents.pkl"
    )


if __name__ == "__main__":
    build_vector_database()
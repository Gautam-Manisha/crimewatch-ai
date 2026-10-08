from pathlib import Path
import pickle

import faiss
from sentence_transformers import SentenceTransformer


VECTOR_DB_DIR = Path(
    "data/knowledge/vector_db"
)

MODEL_NAME = "all-MiniLM-L6-v2"


class CrimeWatchRetriever:

    def __init__(self):

        print("Loading CrimeWatch RAG system...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        self.index = faiss.read_index(
            str(
                VECTOR_DB_DIR / "crimewatch.index"
            )
        )

        with open(
            VECTOR_DB_DIR / "documents.pkl",
            "rb",
        ) as file:

            self.documents = pickle.load(file)

        print("RAG system loaded.")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
        )

        distances, indices = self.index.search(
            query_embedding.astype("float32"),
            top_k,
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0],
        ):

            if index < 0:
                continue

            results.append({
                "document": self.documents[index],
                "distance": float(distance),
            })

        return results


if __name__ == "__main__":

    retriever = CrimeWatchRetriever()

    query = input(
        "\nEnter your incident query: "
    )

    results = retriever.retrieve(
        query,
        top_k=3,
    )

    print()
    print("=" * 60)
    print("SEMANTICALLY RETRIEVED CONTEXT")
    print("=" * 60)

    for index, result in enumerate(
        results,
        start=1
    ):

        print()
        print(f"RESULT #{index}")
        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )
        print("-" * 40)
        print(result["document"])
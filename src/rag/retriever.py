from pathlib import Path


KNOWLEDGE_FILE = Path(
    "data/knowledge/crimewatch_knowledge.txt"
)


def load_knowledge():

    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_FILE}"
        )

    text = KNOWLEDGE_FILE.read_text(
        encoding="utf-8"
    )

    return text


def retrieve_context(label: str):

    knowledge = load_knowledge()

    sections = knowledge.split("\n\n\n")

    label_upper = label.upper()

    matching_sections = []

    for section in sections:

        if label_upper in section.upper():

            matching_sections.append(section)

    if not matching_sections:

        return (
            "No specific knowledge-base context "
            f"was found for {label}."
        )

    return "\n\n".join(
        matching_sections
    )


if __name__ == "__main__":

    context = retrieve_context("Fighting")

    print("=" * 60)
    print("RETRIEVED CONTEXT")
    print("=" * 60)
    print(context)
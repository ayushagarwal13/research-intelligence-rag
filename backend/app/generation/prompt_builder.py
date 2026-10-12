SYSTEM_INSTRUCTION = """
You are a research assistant.

Answer the user's question using only the
provided research context.

Rules:
1. Use the provided context as the source of truth.
2. Do not invent facts that are not supported by the context.
3. If the context does not contain enough information
   to answer the question, clearly say that the available
   research documents do not provide enough information.
4. Keep the answer focused on the user's question.
5. Cite the relevant sources using their source numbers.
""".strip()


def build_prompt(
    query: str,
    context: str,
) -> str:
    """
    Build the final prompt sent to the generation model.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not context.strip():
        raise ValueError("Context cannot be empty.")

    return f"""
{SYSTEM_INSTRUCTION}

Research Context:
----------------
{context}
----------------

User Question:
{query.strip()}

Answer the question using the research context above.
Include source references such as [Source 1] when
making claims supported by a particular source.
""".strip()

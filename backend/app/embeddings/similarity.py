import numpy as np


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """
    a = np.array(vector_a)
    b = np.array(vector_b)

    similarity = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    return float(similarity)

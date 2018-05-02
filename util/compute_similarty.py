import numpy as np


def cosine_similarity(a, b):
    tmp1 = sum(a * b)
    tmp2 = (sum(a ** 2) ** 0.5) * (sum(b ** 2) ** 0.5)
    similarity = tmp1 / (tmp2 + 1e-12)
    return similarity

if __name__ == '__main__':
    A = np.array([-1, 2])
    B = np.array([2, 3])
    sim = cosine_similarity(A, B)
    print sim

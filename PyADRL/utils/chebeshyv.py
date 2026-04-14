# Chebyshev distance is used as the distance metric for rewards
def chebyshev_distance(x1: int, y1: int, x2: int, y2: int):
    return max(abs(x1 - x2), abs(y1 - y2))

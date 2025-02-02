import math
import matplotlib.pyplot as plt

# Class for representing the terrain
class Terrain:
    def __init__(self, vertices):
        self.vertices = vertices
        self.edges = [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]

# Helper functions

def almost_equal(a, b, eps=1e-9):
    return abs(a - b) < eps

def line_support(p, q):
    if almost_equal(p[0], q[0]):
        return None, p[0]
    m = (q[1] - p[1]) / (q[0] - p[0])
    b = p[1] - m * p[0]
    return m, b

def compute_upper_envelope(lines):
    sorted_lines = sorted(lines, key=lambda ln: ln[0] if ln[0] is not None else float('inf'))
    hull = []

    def intersect_x(L1, L2):
        m1, b1 = L1
        m2, b2 = L2
        if m1 is None:  # Vertical line
            return b1
        if m2 is None:
            return b2
        if almost_equal(m1, m2):
            return None
        return (b2 - b1) / (m1 - m2)

    for line in sorted_lines:
        while len(hull) >= 2:
            x1 = intersect_x(hull[-2], hull[-1])
            x2 = intersect_x(hull[-1], line)
            if x2 is None or x1 is None or x2 > x1:
                break
            hull.pop()
        hull.append(line)
    return hull

def eval_upper_envelope(hull, x):
    if not hull:
        return -float('inf')

    def line_value(line, x):
        m, b = line
        return m * x + b if m is not None else float('inf')

    best_y = -float('inf')
    for line in hull:
        best_y = max(best_y, line_value(line, x))
    return best_y

# Parametric search algorithm

def parametric_search(terrain, u_idx, min_tower=1.0, eps=1e-6):
    low, high = min_tower, 1000  # Initial search bounds

    def feasible(h1):
        u_base = terrain.vertices[u_idx]
        u_top = (u_base[0], u_base[1] + h1)
        lines = []
        for edge in terrain.edges:
            m, b = line_support(*edge)
            lines.append((m, b))
        hull = compute_upper_envelope(lines)
        for p in terrain.vertices:
            px, py = p
            if px != u_base[0]:
                if eval_upper_envelope(hull, px) > py + h1:
                    return False
        return True

    best_h1 = high
    while high - low > eps:
        mid = (low + high) / 2
        if feasible(mid):
            best_h1 = mid
            high = mid
        else:
            low = mid

    return best_h1

def solve_discrete_two_watchtowers(terrain):
    best_val = float('inf')
    best_sol = None
    for u_idx in range(len(terrain.vertices)):
        h1 = parametric_search(terrain, u_idx)
        u_base = terrain.vertices[u_idx]
        u_top = (u_base[0], u_base[1] + h1)

        # Compute height and placement for the second tower
        lines = []
        for edge in terrain.edges:
            m, b = line_support(*edge)
            lines.append((m, b))
        hull = compute_upper_envelope(lines)

        best_h2 = float('inf')
        best_p = None
        for p_idx, p in enumerate(terrain.vertices):
            if p_idx != u_idx:
                px, py = p
                env_y = eval_upper_envelope(hull, px)
                needed_h2 = max(env_y - py, 1.0)
                if needed_h2 < best_h2:
                    best_h2 = needed_h2
                    best_p = p_idx

        max_height = max(h1, best_h2)
        if max_height < best_val:
            best_val = max_height
            best_sol = (u_idx, h1, best_p, best_h2)

    return best_val, best_sol

# Visualization and testing

def demo_run():
    vertices = [(0, 0), (1, 3), (2, 1), (3, 3), (4, 2), (5, 4)]
    terrain = Terrain(vertices)

    best_val, (u_idx, h1, p_idx, h2) = solve_discrete_two_watchtowers(terrain)

    print(f"Optimal max height: {best_val}")
    print(f"First tower at index {u_idx}, height: {h1}")
    print(f"Second tower at index {p_idx}, height: {h2}")

    plt.figure(figsize=(10, 6))
    plt.title("Two Watchtowers with Parametric Search")
    for edge in terrain.edges:
        plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'k-')

    u_base = terrain.vertices[u_idx]
    p_base = terrain.vertices[p_idx]

    plt.plot([u_base[0], u_base[0]], [u_base[1], u_base[1] + h1], 'r-', label="First Tower")
    plt.plot([p_base[0], p_base[0]], [p_base[1], p_base[1] + h2], 'b-', label="Second Tower")

    plt.legend()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    demo_run()
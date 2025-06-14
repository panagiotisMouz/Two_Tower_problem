from math import isclose, sqrt
from typing import List, Tuple, Optional
import time
import numpy as np
from matplotlib import pyplot as plt

# Define a Point type for clarity.
Point = Tuple[float, float]

def line_from_points(p: Point, q: Point) -> Tuple[float, float]:
    if isclose(q[0], p[0], abs_tol=1e-12):
        raise ValueError("Vertical line encountered")
    m = (q[1] - p[1]) / (q[0] - p[0])
    b = p[1] - m * p[0]
    return m, b

def intersection_line(m1: float, b1: float, m2: float, b2: float) -> Optional[Point]:
    if isclose(m1, m2, abs_tol=1e-12):
        return None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return (x, y)

def compute_SPT_upper_envelope(terrain: List[Point], u: int) -> List[float]:
    n = len(terrain)
    u_point = terrain[u]
    S = [0.0] * n

    # Process vertices to the right of u
    envelope = []  # Each element is (x_start, m, b)
    for w in range(u + 1, n):
        xw = terrain[w][0]
        m_direct, b_direct = line_from_points(u_point, terrain[w])
        extra = 0.0
        # Check existing envelope segments
        for seg in envelope:
            x_start, m_seg, b_seg = seg
            if xw >= x_start:
                y_seg = m_seg * xw + b_seg
                y_direct = m_direct * xw + b_direct
                extra = max(extra, y_seg - y_direct)
        S[w] = extra
        # Update envelope with the new line (from u_point shifted by S[w] to terrain[w])
        new_line = line_from_points((u_point[0], u_point[1] + S[w]), terrain[w])
        # Prune segments dominated by new_line
        while envelope:
            x_start, m_last, b_last = envelope[-1]
            inter = intersection_line(new_line[0], new_line[1], m_last, b_last)
            if inter is None or inter[0] <= x_start + 1e-9:
                envelope.pop()
            else:
                break
        envelope.append((xw, new_line[0], new_line[1]))

    # Process vertices to the LEFT of u using a reversed convex hull approach
    envelope_left = []  # Each element is (x_start, m, b), valid for x <= x_start
    for w in range(u - 1, -1, -1):
        xw = terrain[w][0]
        m_direct, b_direct = line_from_points(u_point, terrain[w])
        extra = 0.0
        # Check existing envelope_left segments
        for seg in envelope_left:
            x_start, m_seg, b_seg = seg
            if xw <= x_start:
                y_seg = m_seg * xw + b_seg
                y_direct = m_direct * xw + b_direct
                extra = max(extra, y_seg - y_direct)
        S[w] = extra
        # Update envelope_left with the new line (from u_point shifted by S[w] to terrain[w])
        new_line = line_from_points((u_point[0], u_point[1] + S[w]), terrain[w])
        # Prune segments dominated by new_line (reverse logic)
        while envelope_left:
            x_start, m_last, b_last = envelope_left[-1]
            inter = intersection_line(new_line[0], new_line[1], m_last, b_last)
            if inter is None or inter[0] >= x_start - 1e-9:
                envelope_left.pop()
            else:
                break
        envelope_left.append((xw, new_line[0], new_line[1]))

    return S

def compute_all_SPTs(terrain: List[Point]) -> List[List[float]]:
    n = len(terrain)
    SPT = []
    for u in range(n):
        SPT.append(compute_SPT_upper_envelope(terrain, u))
    return SPT

def compute_type1_from_SPT(SPT: List[List[float]]) -> List[List[float]]:
    n = len(SPT)
    type1 = []
    for u in range(n):
        vals = set()
        for w in range(n):
            if u == w:
                continue
            vals.add(SPT[u][w])
        vals.add(0.0)
        type1.append(sorted(vals))
    return type1

def solve_joint_visibility_exact(h_a: float, h_b: float, h_pa: float, h_pb: float) -> Optional[float]:
    """
    Solve the quadratic equation for the joint critical height h where the sight-lines
    from tower u (with endpoints requiring h_a and h_b) and tower v (with endpoints requiring h_pa and h_pb)
    meet on an edge.
    """
    A = (h_b - h_a) - (h_pb - h_pa)
    B = 2 * (h_a * (h_pb - h_pa) - h_pa * (h_b - h_a))
    C = h_a**2 - h_pa**2
    if isclose(A, 0, abs_tol=1e-12):
        # Fall back to a linear solution.
        if isclose((h_b - h_a) - (h_pb - h_pa), 0, abs_tol=1e-12):
            return None
        t = (h_pa - h_a) / ((h_b - h_a) - (h_pb - h_pa))
        candidate = h_a + t * (h_b - h_a)
        return candidate if 0 < t < 1 else None
    disc = B * B - 4 * A * C
    if disc < 0:
        return None
    r1 = (-B + sqrt(disc)) / (2 * A)
    r2 = (-B - sqrt(disc)) / (2 * A)
    candidate = max(r1, r2)
    if h_b < candidate < h_a:
        return candidate
    return None

# -------------------- UPDATED TYPE 2 CRITICAL HEIGHTS --------------------
def compute_type2_for_edge_exact(H_u_a: float, H_u_b: float, H_v_a: float, H_v_b: float) -> Optional[float]:
    """
    Compute the critical height for the edge between two consecutive terrain vertices
    when considering towers at u and v. This function returns a critical joint height only
    when one tower covers the left endpoint better and the other covers the right endpoint better.
    Otherwise, if one tower alone covers the edge, no joint critical height is needed.
    """
    # If tower u covers both endpoints at least as well as tower v, no joint height is needed.
    if H_u_a <= H_v_a and H_u_b <= H_v_b:
        return None
    # Similarly, if tower v covers both endpoints better than tower u.
    if H_u_a >= H_v_a and H_u_b >= H_v_b:
        return None
    # Otherwise, compute the joint critical height.
    return solve_joint_visibility_exact(H_u_a, H_u_b, H_v_a, H_v_b)
# -------------------------------------------------------------------------

def compute_type2_heights_exact(SPT: List[List[float]], u: int, v: int) -> List[float]:
    crits = set()
    if u > v:
        u, v = v, u  # Ensure u is to the left of v.
    for p in range(u, v):
        H_u_a = SPT[u][p]
        H_u_b = SPT[u][p + 1]
        H_v_a = SPT[v][p]
        H_v_b = SPT[v][p + 1]
        crit = compute_type2_for_edge_exact(H_u_a, H_u_b, H_v_a, H_v_b)
        if crit is not None:
            crits.add(crit)
    return sorted(crits)

#############################################
# Step 4: Enhanced Decision Procedure Using Upper Envelopes
#############################################
def is_guarded_envelope(terrain: List[Point], SPT: List[List[float]], u: int, v: int, h: float) -> bool:
    n = len(terrain)
    for p in range(n - 1):
        # For edges entirely to the left of tower u:
        if p + 1 <= u:
            if max(SPT[u][p], SPT[u][p + 1]) > h:
                return False
        # For edges entirely to the right of tower v:
        elif p >= v:
            if max(SPT[v][p], SPT[v][p + 1]) > h:
                return False
        else:
            # For edges between u and v:
            cover_u = max(SPT[u][p], SPT[u][p + 1]) <= h
            cover_v = max(SPT[v][p], SPT[v][p + 1]) <= h
            if cover_u or cover_v:
                continue
            # Neither tower alone covers the edge. Check for joint visibility.
            joint = compute_type2_for_edge_exact(
                SPT[u][p], SPT[u][p + 1],
                SPT[v][p], SPT[v][p + 1]
            )
            if joint is None or h < joint - 1e-9:
                return False
    return True

def find_min_height_for_pair_envelope(terrain: List[Point], SPT: List[List[float]], type1: List[List[float]], u: int, v: int) -> float:
    crits = set(type1[u] + type1[v])
    type2 = compute_type2_heights_exact(SPT, u, v)
    for c in type2:
        crits.add(c)
    sorted_crits = sorted(crits)
    lo, hi = 0, len(sorted_crits) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = sorted_crits[mid]
        if is_guarded_envelope(terrain, SPT, u, v, candidate):
            best = candidate
            hi = mid - 1
        else:
            lo = mid + 1
    return best if best is not None else float('inf')

def two_watchtowers_min_height_envelope(terrain: List[Point]) -> Tuple[float, Tuple[int, int]]:
    n = len(terrain)
    SPT = compute_all_SPTs(terrain)
    type1 = compute_type1_from_SPT(SPT)
    best_h = float('inf')
    best_pair = (-1, -1)
    for u in range(n):
        for v in range(u + 1, n):
            h_pair = find_min_height_for_pair_envelope(terrain, SPT, type1, u, v)
            if h_pair < best_h:
                best_h = h_pair
                best_pair = (u, v)
    return best_h, best_pair

#############################################
# Step 5: Visualization
#############################################
def plot_solution(terrain: List[Point], best_pair: Tuple[int, int], best_height: float):
    plt.figure(figsize=(12, 6))
    xs = [p[0] for p in terrain]
    ys = [p[1] for p in terrain]
    plt.plot(xs, ys, 'ko-', label="Terrain")

    u, v = best_pair
    tower_u = terrain[u]
    tower_v = terrain[v]
    plt.plot([tower_u[0], tower_u[0]], [tower_u[1], tower_u[1] + best_height],
             'r-', linewidth=2, label="Tower u")
    plt.scatter([tower_u[0]], [tower_u[1] + best_height], color="red", s=100)
    plt.plot([tower_v[0], tower_v[0]], [tower_v[1], tower_v[1] + best_height],
             'b-', linewidth=2, label="Tower v")
    plt.scatter([tower_v[0]], [tower_v[1] + best_height], color="blue", s=100)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(f"Two Watchtowers Guarding Terrain (Height = {best_height})")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    # Sample terrain: an x-monotone chain.
    terrain: List[Point] = [(0, 5), (1, 5), (2, 4), (4, 0), (6, 7), (7, 0), (8, 6), (9, 4)]
    best_height, best_pair = two_watchtowers_min_height_envelope(terrain)
    print("Optimal minimum height for towers:", best_height)
    print("Best pair of vertices (tower positions):", best_pair)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    plot_solution(terrain, best_pair, best_height)

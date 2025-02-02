import math
import matplotlib.pyplot as plt

class Terrain:
    def __init__(self, vertices):
        self.vertices = vertices
        self.n = len(vertices) - 1
        self.edges = []
        for i in range(self.n):
            self.edges.append((vertices[i], vertices[i+1]))
        
        self.non_visible_chains = {}  

def almost_equal(a, b, eps=1e-9):
    return abs(a - b) < eps

def line_support(p, q):
    if almost_equal(p[0], q[0]):
        return (None, p[0])
    m = (q[1] - p[1]) / (q[0] - p[0])
    b = p[1] - m * p[0]
    return (m, b)

def line_intersect(m1, b1, m2, b2):
    if m1 is None and m2 is None:
        return None
    if m1 is None:
        x_val = b1
        y_val = m2 * x_val + b2
        return (x_val, y_val)
    if m2 is None:
        x_val = b2
        y_val = m1 * x_val + b1
        return (x_val, y_val)
    if almost_equal(m1, m2):
        return None
    x_val = (b2 - b1) / (m1 - m2)
    y_val = m1 * x_val + b1
    return (x_val, y_val)

def sees_point(terrain, tower_top, pt):
    (xT, yT) = tower_top
    (xp, yp) = pt
    if almost_equal(xT, xp):
        return True
    dx = xp - xT
    dy = yp - yT
    x_low, x_high = min(xT, xp), max(xT, xp)
    for (A, B) in terrain.edges:
        (x1, y1) = A
        (x2, y2) = B
        if x2 < x_low or x1 > x_high:
            continue
        
        if almost_equal(x1, xp) and almost_equal(y1, yp):
            continue
        if almost_equal(x2, xp) and almost_equal(y2, yp):
            continue
        xm = 0.5 * (max(x1, x_low) + min(x2, x_high))
        t = (xm - xT) / dx if not almost_equal(dx, 0) else 0.0
        if 0 < t < 1:
            line_y = yT + t * dy
            (mE, bE) = line_support(A, B)
            if mE is not None:
                edge_y = mE * xm + bE
                if line_y < edge_y - 1e-12:
                    return False
    return True

def edge_non_visible(terrain, u_top, edge):
    (A, B) = edge
    sample_ts = [0.25, 0.5, 0.75]
    for t in sample_ts:
        xm = A[0] + t * (B[0] - A[0])
        ym = A[1] + t * (B[1] - A[1])
        if not sees_point(terrain, u_top, (xm, ym)):
            return True
    return False

def build_line_sets(terrain, u_top, min_tower=1.0):
    Le = []
    Lr = []
    invis_edges = []
    non_visible_groups = {}
    current_group = []
    group_id = 0
    for idx, e in enumerate(terrain.edges):
        is_non_visible = edge_non_visible(terrain, u_top, e)
        if is_non_visible:
            invis_edges.append(e)
            current_group.append(e)
        else:
            if current_group:
                non_visible_groups[group_id] = current_group.copy()
                group_id += 1
                current_group.clear()
    if current_group:
        non_visible_groups[group_id] = current_group.copy()
    terrain.non_visible_chains = non_visible_groups  
    for e in invis_edges:
        (mE, bE) = line_support(e[0], e[1])
        Le.append((mE, bE))
        (A, B) = e
        r_end = B if B[0] > A[0] else A
        (rx, ry) = r_end
        for p in terrain.vertices:
            if p[0] > rx + 1e-12:
                (mT, bT) = line_support(r_end, p)
                Lr.append((mT, bT))
    return (Le, Lr)

def slope_val(m, big=1e9):
    if m is None:
        return big
    return m

def compute_upper_envelope(lines):
    Ls = sorted(lines, key=lambda ln: slope_val(ln[0]))
    hull = []
    def intersect_x(L1, L2):
        (m1, b1) = L1
        (m2, b2) = L2
        ipt = line_intersect(m1, b1, m2, b2)
        return ipt[0] if ipt else None
    for ln in Ls:
        if not hull:
            hull.append((ln, -math.inf))
            continue
        while True:
            if len(hull) == 0:
                hull.append((ln, -math.inf))
                break
            x_new = intersect_x(hull[-1][0], ln)
            if x_new is None:
                (topm, topb) = hull[-1][0]
                if abs(slope_val(topm) - slope_val(ln[0])) < 1e-12:
                    if ln[1] > topb:
                        hull[-1] = (ln, hull[-1][1])
                break
            else:
                x_prev = hull[-1][1]
                if len(hull) == 1:
                    hull.append((ln, x_new))
                    break
                if x_new <= x_prev + 1e-12:
                    hull.pop()
                else:
                    hull.append((ln, x_new))
                    break
    return hull

def eval_upper_envelope(hull, x):
    if not hull:
        return -1e9
    low = 0
    high = len(hull) - 1
    if len(hull) == 1:
        (m, b) = hull[0][0]
        if m is None:
            return 1e9
        return m * x + b
    while low < high:
        mid = (low + high + 1) // 2
        xm = hull[mid][1]
        if x < xm:
            high = mid - 1
        else:
            low = mid
    (m, b) = hull[low][0]
    if m is None:
        return 1e9
    return m * x + b

def discrete_2nd_tower(terrain, u_idx, h1, min_tower=0.1):
    u_base = terrain.vertices[u_idx]
    u_top = (u_base[0], u_base[1] + h1)
    (Le, Lr) = build_line_sets(terrain, u_top)
    if not Le and not Lr:
        return (min_tower, 0)
    lines = Le + Lr
    hull = compute_upper_envelope(lines)
    best_h2 = float('inf')
    best_p = 0
    for p_idx in range(len(terrain.vertices)):
        if p_idx == u_idx:
            continue
        p = terrain.vertices[p_idx]
        px, py = p
        envY = eval_upper_envelope(hull, px)
        needed = envY - py
        if needed < min_tower:
            needed = min_tower
        if needed < best_h2:
            best_h2 = needed
            best_p = p_idx
    return (best_h2, best_p)

def semi_continuous_2nd_tower(terrain, u_idx, h1, min_tower=0.1):
    u_base = terrain.vertices[u_idx]
    u_top = (u_base[0], u_base[1] + h1)
    (Le, Lr) = build_line_sets(terrain, u_top)
    if not Le and not Lr:
        return (min_tower, u_base)  
    
    lines = Le + Lr
    hull = compute_upper_envelope(lines)
    
    best_h2 = float('inf')
    best_p = None

    non_visible_edges = []
    for e in terrain.edges:
        if edge_non_visible(terrain, u_top, e):
            non_visible_edges.append(e)
    
    if not non_visible_edges:
        return (min_tower, u_base)  

    def is_covered(px, py, h2):
        tower2_top = (px, py + h2)
        for e in non_visible_edges:
            (A, B) = e
            if sees_point(terrain, tower2_top, A) or sees_point(terrain, tower2_top, B):
                continue
            if not edge_non_visible(terrain, tower2_top, e):
                continue
            return False
        return True

    for edge in non_visible_edges:
        A, B = edge
        
        left = 0.0
        right = 1.0
        for _ in range(30):  
            m1 = left + (right - left) / 3
            m2 = right - (right - left) / 3
            px1 = A[0] + m1 * (B[0] - A[0])
            py1 = A[1] + m1 * (B[1] - A[1])
            px2 = A[0] + m2 * (B[0] - A[0])
            py2 = A[1] + m2 * (B[1] - A[1])
            
            envY1 = eval_upper_envelope(hull, px1)
            needed1 = envY1 - py1
            envY2 = eval_upper_envelope(hull, px2)
            needed2 = envY2 - py2
            
            needed1 = max(needed1, min_tower)
            needed2 = max(needed2, min_tower)
            
            if needed1 < needed2:
                right = m2
            else:
                left = m1
        
        m = (left + right) / 2
        px = A[0] + m * (B[0] - A[0])
        py = A[1] + m * (B[1] - A[1])
        envY = eval_upper_envelope(hull, px)
        needed = envY - py
        needed = max(needed, min_tower)
        
        if is_covered(px, py, needed):
            if needed < best_h2:
                best_h2 = needed
                best_p = (px, py)
    
    return (best_h2, best_p)

def build_shortest_path_edges(terrain, u_idx, direction="left"):
    if direction == "left":
        return terrain.edges[:u_idx]
    else:
        return terrain.edges[u_idx:]

def gather_critical_heights(terrain, u_idx):
    u = terrain.vertices[u_idx]
    xU, yU = u
    leftE = build_shortest_path_edges(terrain, u_idx, "left")
    rightE = build_shortest_path_edges(terrain, u_idx, "right")
    H = set()
    H.add(1.0)
    def handle_edge(e):
        (A, B) = e
        (m, b) = line_support(A, B)
        if m is None:
            return
        hval = (m * xU + b) - yU
        if hval >= 1.0:
            H.add(hval)
    for e in leftE:
        handle_edge(e)
    for e in rightE:
        handle_edge(e)
    out = sorted(H)
    return out

def solve_discrete_two_watchtowers(terrain, H):
    best_val = float('inf')
    best_sol = None
    for u_idx in range(len(terrain.vertices)):
        cH = gather_critical_heights(terrain, u_idx)
        for h1 in cH:
            if h1 > H:
                continue
            h2, p_idx = discrete_2nd_tower(terrain, u_idx, h1)
            if h2 > H:
                continue
            val = max(h1, h2)
            if val < best_val:
                best_val = val
                best_sol = (u_idx, h1, p_idx, h2)
    return best_sol

def solve_semi_continuous_two_watchtowers(terrain, H):
    best_val = float('inf')
    best_sol = None
    for u_idx in range(len(terrain.vertices)):
        cH = gather_critical_heights(terrain, u_idx)
        for h1 in cH:
            if h1 > H:
                continue
            h2, p_info = semi_continuous_2nd_tower(terrain, u_idx, h1)
            if h2 > H:
                continue
            val = max(h1, h2)
            if val < best_val:
                best_val = val
                best_sol = (u_idx, h1, p_info, h2)
    return best_sol

def coverage_fill(ax, terrain, base, h, color='orange', alpha=0.3, subdiv=30):
    tower_top = (base[0], base[1] + h)
    for (A, B) in terrain.edges:
        Ax, Ay = A
        Bx, By = B
        for i in range(subdiv):
            t1 = i / float(subdiv)
            t2 = (i + 1) / float(subdiv)
            S1 = (Ax + t1 * (Bx - Ax), Ay + t1 * (By - Ay))
            S2 = (Ax + t2 * (Bx - Ax), Ay + t2 * (By - Ay))
            mx = 0.5 * (S1[0] + S2[0])
            my = 0.5 * (S1[1] + S2[1])
            if sees_point(terrain, tower_top, (mx, my)):
                ax.fill([tower_top[0], S1[0], S2[0]],
                        [tower_top[1], S1[1], S2[1]],
                        color=color, alpha=alpha)

def is_feasible_discrete(terrain, H):
    sol = solve_discrete_two_watchtowers(terrain, H)
    return sol is not None and max(sol[1], sol[3]) <= H

def is_feasible_semi_continuous(terrain, H):
    sol = solve_semi_continuous_two_watchtowers(terrain, H)
    return sol is not None and max(sol[1], sol[3]) <= H

def find_optimal_max_height_discrete(terrain):
    lower = 0.0
    upper = max(y for (_, y) in terrain.vertices) + 1000.0
    precision = 1e-6
    optimal_H = upper
    while upper - lower > precision:
        mid = (lower + upper) / 2
        if is_feasible_discrete(terrain, mid):
            optimal_H = mid
            upper = mid
        else:
            lower = mid
    return optimal_H

def find_optimal_max_height_semi_continuous(terrain):
    lower = 0.0
    upper = max(y for (_, y) in terrain.vertices) + 1000.0
    precision = 1e-6
    optimal_H = upper
    while upper - lower > precision:
        mid = (lower + upper) / 2
        if is_feasible_semi_continuous(terrain, mid):
            optimal_H = mid
            upper = mid
        else:
            lower = mid
    return optimal_H

def demo_run():
    test_terrains = {
        "simple_mountain_terrain": [
            (0, 0),    
            (2, 1),    
            (4, 3),    
            (6, 1),    
            (8, 2),    
            (10, 0)    
        ],
        "Regular Terrain": [(1, 1), (2, 0.5), (3, 2), (4, 1.8), (5, 3.5), (6, 3), (7, 4.2), (8, 3.8), (9, 5),
            (10, 4.5), (11, 4.7), (12, 6), (13, 5.5), (14, 6.8), (15, 5), (16, 4), (17, 3.7), (18, 4.2)
        ],
        "Steep Slopes": [
            (0, 0), (1, 5), (2, 10), (3, 15), (4, 20), (5, 25), (6, 30), (7, 35), (8, 40), (9, 45), (10, 50)
        ],
        "Long Terrain": [ (0, 0), (1, 1), (2, 0.5), (3, 2), (4, 1.8), (5, 3.5), (6, 3), (7, 4.2), (8, 3.8), (9, 5),
            (10, 4.5), (11, 4.7), (12, 6), (13, 5.5), (14, 6.8), (15, 5), (16, 4), (17, 3.7), (18, 4.2),(19, 2.5),
            (20, 3), (21, 1.8), (22, 2.2), (23, 3), (24, 2.7), (25, 1.5), (26, 2), (27, 0.8),
            (28, 1.2), (29, 0.5), (30, 1.7), (31, 1), (32, 3)
        ],
        "Long Round Terrain": [(x, math.sin(x * 0.1) * 5) for x in range(0, 50)],
        "Nearly Vertical Edges": [
            (0, 0), (1, 1), (2, 2), (3, 100), (4, 101), (5, 102), (6, 200), (7, 201), (8, 202)
        ]
    }

    for terrain_name, vertices in test_terrains.items():
        print(f"\n=== Testing Terrain: {terrain_name} ===")
        ter = Terrain(vertices)
        
        # Discrete Two Watchtowers with Parametric Search
        optimal_H_discrete = find_optimal_max_height_discrete(ter)
        best_sol_discrete = solve_discrete_two_watchtowers(ter, optimal_H_discrete)
        if best_sol_discrete:
            u_idx, h1, p_idx, h2 = best_sol_discrete
            print(f"Discrete Solution => Optimal max height = {optimal_H_discrete:.6f}")
            print(f"  Tower1 at index {u_idx} (Position: {ter.vertices[u_idx]}), height h1 = {h1:.6f}")
            print(f"  Tower2 at index {p_idx} (Position: {ter.vertices[p_idx]}), height h2 = {h2:.6f}")
        else:
            print("Discrete Solution => No feasible solution found.")
        
        # Plotting Discrete Solution
        if best_sol_discrete:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"Discrete 2-Towers on {terrain_name}")
            for e in ter.edges:
                (A, B) = e
                ax.plot([A[0], B[0]], [A[1], B[1]], color='black', linewidth=2)
            baseU = ter.vertices[u_idx]
            ax.plot([baseU[0], baseU[0]], [baseU[1], baseU[1] + h1], color='red', linewidth=3, label='Tower 1')
            ax.plot(baseU[0], baseU[1] + h1, 'o', color='red')
            baseP = ter.vertices[p_idx]
            ax.plot([baseP[0], baseP[0]], [baseP[1], baseP[1] + h2], color='green', linewidth=3, label='Tower 2')
            ax.plot(baseP[0], baseP[1] + h2, 'o', color='green')
            coverage_fill(ax, ter, baseU, h1, color='red', alpha=0.3)
            coverage_fill(ax, ter, baseP, h2, color='green', alpha=0.3)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            ax.grid(True)
            plt.show()
        
        # Semi-Continuous Two Watchtowers with Parametric Search
        optimal_H_semi = find_optimal_max_height_semi_continuous(ter)
        best_sol_semi = solve_semi_continuous_two_watchtowers(ter, optimal_H_semi)
        if best_sol_semi:
            u_idx, h1, p_info, h2 = best_sol_semi
            print(f"Semi-Continuous Solution => Optimal max height = {optimal_H_semi:.6f}")
            print(f"  Tower1 at index {u_idx} (Position: {ter.vertices[u_idx]}), height h1 = {h1:.6f}")
            print(f"  Tower2 at position {p_info}, height h2 = {h2:.6f}")
        else:
            print("Semi-Continuous Solution => No feasible solution found.")
        
        # Plotting Semi-Continuous Solution
        if best_sol_semi:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"Semi-Continuous 2-Towers on {terrain_name}")
            for e in ter.edges:
                (A, B) = e
                ax.plot([A[0], B[0]], [A[1], B[1]], color='black', linewidth=2)
            baseU = ter.vertices[u_idx]
            ax.plot([baseU[0], baseU[0]], [baseU[1], baseU[1] + h1], color='blue', linewidth=3, label='Tower 1')
            ax.plot(baseU[0], baseU[1] + h1, 'o', color='blue')
            baseP = p_info
            ax.plot([baseP[0], baseP[0]], [baseP[1], baseP[1] + h2], color='purple', linewidth=3, label='Tower 2')
            ax.plot(baseP[0], baseP[1] + h2, 'o', color='purple')
            coverage_fill(ax, ter, baseU, h1, color='blue', alpha=0.3)
            coverage_fill(ax, ter, baseP, h2, color='purple', alpha=0.3)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            ax.grid(True)
            plt.show()

def is_feasible_discrete(terrain, H):
    sol = solve_discrete_two_watchtowers(terrain, H)
    return sol is not None and max(sol[1], sol[3]) <= H

def is_feasible_semi_continuous(terrain, H):
    sol = solve_semi_continuous_two_watchtowers(terrain, H)
    return sol is not None and max(sol[1], sol[3]) <= H

def find_optimal_max_height_discrete(terrain):
    lower = 0.0
    upper = max(y for (_, y) in terrain.vertices) + 1000.0  # Adjust as needed
    precision = 1e-6
    optimal_H = upper
    while upper - lower > precision:
        mid = (lower + upper) / 2
        if is_feasible_discrete(terrain, mid):
            optimal_H = mid
            upper = mid
        else:
            lower = mid
    return optimal_H

def find_optimal_max_height_semi_continuous(terrain):
    lower = 0.0
    upper = max(y for (_, y) in terrain.vertices) + 1000.0
    precision = 1e-6
    optimal_H = upper
    while upper - lower > precision:
        mid = (lower + upper) / 2
        if is_feasible_semi_continuous(terrain, mid):
            optimal_H = mid
            upper = mid
        else:
            lower = mid
    return optimal_H

def solve_discrete_two_watchtowers(terrain, H):
    best_val = float('inf')
    best_sol = None
    for u_idx in range(len(terrain.vertices)):
        cH = gather_critical_heights(terrain, u_idx)
        for h1 in cH:
            if h1 > H:
                continue
            h2, p_idx = discrete_2nd_tower(terrain, u_idx, h1)
            if h2 > H:
                continue
            val = max(h1, h2)
            if val < best_val:
                best_val = val
                best_sol = (u_idx, h1, p_idx, h2)
    return best_sol

def solve_semi_continuous_two_watchtowers(terrain, H):
    best_val = float('inf')
    best_sol = None
    for u_idx in range(len(terrain.vertices)):
        cH = gather_critical_heights(terrain, u_idx)
        for h1 in cH:
            if h1 > H:
                continue
            h2, p_info = semi_continuous_2nd_tower(terrain, u_idx, h1)
            if h2 > H:
                continue
            val = max(h1, h2)
            if val < best_val:
                best_val = val
                best_sol = (u_idx, h1, p_info, h2)
    return best_sol

if __name__ == "__main__":
    demo_run() 
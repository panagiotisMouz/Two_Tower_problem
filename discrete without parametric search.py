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

def solve_discrete_two_watchtowers(terrain):
    best_val = float('inf')
    best_sol = None
    visibility_data = {
        "Tower1": {"visible": [], "non_visible": {}},
        "Tower2": {"visible": [], "non_visible": {}},
    }

    for u_idx in range(len(terrain.vertices)):
        cH = gather_critical_heights(terrain, u_idx)
        if not cH:
            h2, p_idx = discrete_2nd_tower(terrain, u_idx, 1.0)
            val = max(1.0, h2)
            if val < best_val:
                best_val = val
                best_sol = (u_idx, 1.0, p_idx, h2)
                
                visibility_data["Tower1"]["non_visible"] = terrain.non_visible_chains.copy()
                terrain.non_visible_chains = {}
                discrete_2nd_tower(terrain, p_idx, h2)
                visibility_data["Tower2"]["non_visible"] = terrain.non_visible_chains.copy()
            continue
        for h1 in cH:
            h2, p_idx = discrete_2nd_tower(terrain, u_idx, h1)
            val = max(h1, h2)
            if val < best_val:
                best_val = val
                best_sol = (u_idx, h1, p_idx, h2)
                
                visibility_data["Tower1"]["non_visible"] = terrain.non_visible_chains.copy()
                terrain.non_visible_chains = {}
                discrete_2nd_tower(terrain, p_idx, h2)
                visibility_data["Tower2"]["non_visible"] = terrain.non_visible_chains.copy()
        bigH = cH[-1] + 1.0
        h2b, p_idx2 = discrete_2nd_tower(terrain, u_idx, bigH)
        val2 = max(bigH, h2b)
        if val2 < best_val:
            best_val = val2
            best_sol = (u_idx, bigH, p_idx2, h2b)
            
            visibility_data["Tower1"]["non_visible"] = terrain.non_visible_chains.copy()
            terrain.non_visible_chains = {}
            discrete_2nd_tower(terrain, p_idx2, h2b)
            visibility_data["Tower2"]["non_visible"] = terrain.non_visible_chains.copy()

    terrain.non_visible_chains = visibility_data  
    return best_val, best_sol

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

def demo_run():
    test_terrains = {
        "Regular Terrain": [(1, 1), (2, 0.5), (3, 2), (4, 1.8), (5, 3.5), (6, 3), (7, 4.2), (8, 3.8), (9, 5),
            (10, 4.5), (11, 4.7), (12, 6), (13, 5.5), (14, 6.8), (15, 5), (16, 4), (17, 3.7), (18, 4.2)
        ],
        "Steep Slopes": [
            (0, 0), (1, 5), (2, 10), (3, 15), (4, 20), (5, 25), (6, 30), (7, 35), (8, 40), (9, 45), (10, 50)
        ],
        "Long Terrain": [ (0, 0), (1, 1), (2, 0.5), (3, 2), (4, 1.8), (5, 3.5), (6, 3), (7, 4.2), (8, 3.8), (9, 5),
            (10, 4.5), (11, 4.7), (12, 6), (13, 5.5), (14, 6.8), (15, 5), (16, 4), (17, 3.7), (18, 4.2),(19, 2.5),
            (20, 3), (21, 1.8), (22, 2.2), (23, 3), (24, 2.7), (25, 1.5), (26, 2), (27, 0.8),
            (28, 1.2), (29, 0.5), (30, 1.7), (31, 1), (32, 3), (33, 2.5), (34, 4.2), (35, 3.8), (36, 5),
            (37, 4), (38, 5.5), (39, 6.2), (40, 5.8), (41, 7), (42, 6.5), (43, 7.5), (44, 7), (45, 6.8),
            (46, 6.2), (47, 5.5), (48, 5.8), (49, 5.2), (50, 4.7), (51, 3.5), (52, 3.8), (53, 3.2),
            (54, 2.5), (55, 2.8), (56, 1.2), (57, 0.7), (58, 1.5), (59, 0.8), (60, 0.2)
        ],
        "Long Round Terrain": [(x, math.sin(x * 0.1) * 5) for x in range(0, 50)],
        "Nearly Vertical Edges": [
            (0, 0), (1, 1), (2, 2), (3, 100), (4, 101), (5, 102), (6, 200), (7, 201), (8, 202)
        ]
    }

    for terrain_name, vertices in test_terrains.items():
        print(f"\n=== Testing Terrain: {terrain_name} ===")
        ter = Terrain(vertices)
        best_val, (u_idx, h1, p_idx, h2) = solve_discrete_two_watchtowers(ter)
        print(f"Best solution => max height = {best_val:.3f}")
        print(f"  Tower1 at index {u_idx} (Position: {ter.vertices[u_idx]}), height h1 = {h1:.3f}")
        print(f"  Tower2 at index {p_idx} (Position: {ter.vertices[p_idx]}), height h2 = {h2:.3f}")
        
        print("Visibility Status and Grouped Non-Visible Chains:")
        for tower, data in ter.non_visible_chains.items():
            print(f"  {tower}:")
            if not data["non_visible"]:
                print("    No non-visible chains.")
            else:
                for group_id, chain in data["non_visible"].items():
                    print(f"    Group {group_id}:")
                    for edge in chain:
                        print(f"      Edge from {edge[0]} to {edge[1]}")
        
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

if __name__ == "__main__":
    demo_run()

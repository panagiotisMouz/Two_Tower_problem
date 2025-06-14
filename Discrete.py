from math import isclose
from typing import List, Tuple
import time
import numpy as np
from matplotlib import pyplot as plt

Point = Tuple[float, float]
Segment = Tuple[Point, Point]

def split_and_reverse(points, x_split):
    left = []
    right = []
    towerone = ()

    found = False
    for x, y in points:
        if x == x_split:
            found = True
            towerone = (x, y)
            continue
        if not found:
            left.append((x, y))
        else:
            right.append((x, y))

    return left[::-1], right, towerone

def find_non_visible_chains(terrain: List[Point], u_idx: int, h1: float) -> Tuple[List[Segment], List[Tuple[float, float]]]:
    global previous, towerOne, non_visible, general, half_visible
    previous = ()
    general = ()
    non_visible = []
    half_visible = ()
    left, right, towerOne = split_and_reverse(terrain, u_idx)
    (x, y) = towerOne
    y = y + h1
    towerOne = (x, y)
    if len(left) > 1:
        for i in range(len(left)):
            (x, y) = left[i]
            if left[0] == (x, y):
                previous = (x, y)
                general = left[i+1]
            else:
                if cross_product(towerOne, previous, left[i]) <= 0:
                    if half_visible == left[i-1]:
                        non_visible.append((left[i], half_visible))
                        previous=(x,y)
                    else:
                        previous = (x, y)
                else:
                    non_visible.append(((x, y), general))
                    half_visible = left[i]
            general = left[i]
    previous = ()
    general = ()
    half_visible = ()
    if len(right) > 1:
        for i in range(len(right)):
            (x, y) = right[i]
            if right[0] == (x, y):
                previous = (x, y)
            else:
                evaluation=cross_product(towerOne, previous, right[i])
                if  evaluation>= 0:

                    if half_visible == right[i-1]:

                        non_visible.append((half_visible,right[i]))
                        previous=(x,y)
                    else:
                        previous = (x, y)

                elif evaluation<0:


                    non_visible.append((general,(x, y)))
                    half_visible=(x,y)

            general = right[i]


    L_r,right_endpoints=compute_upper_tangents(non_visible, terrain)
    return non_visible, L_r


def compute_upper_tangents(non_visible, terrain):
    right_dict = {}
    for seg in non_visible:
        pt = seg[1]
        if pt[0] in right_dict:
            if pt[1] > right_dict[pt[0]][1]:
                right_dict[pt[0]] = pt
        else:
            right_dict[pt[0]] = pt

    right_endpoints = list(right_dict.values())

    # Ensure the terrain's last point is included:
    p_end = terrain[-1]
    if not any(isclose(p_end[0], pt[0], abs_tol=1e-9) and isclose(p_end[1], pt[1], abs_tol=1e-9)
               for pt in right_endpoints):
        right_endpoints.append(p_end)

    # Sort right endpoints by x (and if equal, by descending y)
    right_endpoints.sort(key=lambda p: (p[0], -p[1]))

    L_r = []
    # Compute tangents between consecutive right endpoints.
    for i in range(len(right_endpoints) - 1):
        r_i = right_endpoints[i]
        r_next = right_endpoints[i + 1]
        # Skip vertical segments (if x's are essentially the same)
        if not np.isclose(r_i[0], r_next[0], atol=1e-9):
            m = (r_next[1] - r_i[1]) / (r_next[0] - r_i[0])
            b = r_i[1] - m * r_i[0]
            L_r.append((m, b))

    # Get the last right endpoint:
    if right_endpoints:
        r_last = right_endpoints[-1]
        p_end = terrain[-1]
        # Instead of checking arbitrary atol differences, we compare the y–values.
        # If the terrain's end y is lower than r_last's y, then extend horizontally.
        if p_end[1] < r_last[1]:
            # Extend horizontally at the last y value (bridge)
            L_r.append((0.0, r_last[1]))
            print("Added horizontal bridge: slope=0.0, intercept =", r_last[1])
        else:
            # Otherwise, add a tangent connecting r_last to the terrain end
            if not np.isclose(r_last[0], p_end[0], atol=1e-9):
                m = (p_end[1] - r_last[1]) / (p_end[0] - r_last[0])
                b = r_last[1] - m * r_last[0]
                L_r.append((m, b))
                print("Added tangent from r_last to terrain end: slope =", m, "intercept =", b)
    return L_r, right_endpoints

def cross_product(a, b, c):
    (ax, ay) = a
    (bx, by) = b
    (cx,cy) = c
    return (ax*by)-(ay*bx)+(ay*cx)-(ax*cy)+(bx*cy)-(by*cx)

def graham_scan(points):
    if len(points) <= 2:
        return points
    points.sort(key=lambda p: (p[0], p[1]))
    lower = []
    for p in points:
        while len(lower) >= 2 and (
                (lower[-1][0] - lower[-2][0]) * (p[1] - lower[-2][1]) - (lower[-1][1] - lower[-2][1]) * (
                p[0] - lower[-2][0])) <= 0:
            lower.pop()
        lower.append(p)
    return lower

def compute_upper_envelope_discrete_graham_modified(non_visible_segments, last, terrain, L_r):
    from math import isclose  # make sure to import isclose
    domain_min, domain_max = -1.0, last + 1.0


    group = non_visible_segments

    all_env_pts = []
    lines = []
    # For each segment in our grouped segments, compute its line (slope, intercept)
    for (x1, y1), (x2, y2) in group:
        if isclose(x2, x1, abs_tol=1e-9):
            continue  # skip vertical segments
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        lines.append((m, b))
    if not lines:
        return []


    lines.extend(L_r)

    # Convert lines to dual points (each line y = m*x + b becomes (m, -b))
    dual_points = [(m, -b) for (m, b) in lines]
    dual_points.sort(key=lambda p: p[0])
    hull_dual = graham_scan(dual_points)
    hull_lines = [(m, -negb) for (m, negb) in hull_dual]

    env_pts = []
    def line_intersection(m1, b1, m2, b2):
        denom = (m1 - m2)
        if isclose(denom, 0, abs_tol=1e-9):
            return None
        x_int = (b2 - b1) / denom
        y_int = m1 * x_int + b1

        return (x_int, y_int)

    for i in range(len(hull_lines) - 1):
        m1, b1 = hull_lines[i]
        m2, b2 = hull_lines[i+1]
        pt = line_intersection(m1, b1, m2, b2)
        print("Intersection point between lines:", pt)
        if pt is not None:
            env_pts.append(pt)

    print("Hull lines:", hull_lines)
    # Create envelope endpoints at the left and right ends of the group domain
    group_min = min(seg[0][0] for seg in group)
    group_max = max(seg[1][0] for seg in group)
    left_pt = (domain_min, hull_lines[0][0] * domain_min + hull_lines[0][1])
    right_pt = (domain_max, hull_lines[-1][0] * domain_max + hull_lines[-1][1])
    group_env = [left_pt] + env_pts + [right_pt]
    all_env_pts.extend(group_env)
    if not all_env_pts:
        return []
    all_env_pts.sort(key=lambda p: p[0])
    unique_x = sorted(set(x for (x, _) in all_env_pts))
    merged = []
    for x in unique_x:
        y_max = max(y for (xx, y) in all_env_pts if isclose(xx, x, abs_tol=1e-9) or xx == x)
        merged.append((x, y_max))
    if merged[0][0] > domain_min:
        merged.insert(0, (domain_min, merged[0][1]))
    if merged[-1][0] < domain_max:
        merged.append((domain_max, merged[-1][1]))
    return merged

def evaluate_envelope(envelope, x):

    if x <= envelope[0][0]:
        return envelope[0][1]
    if x >= envelope[-1][0]:
        return envelope[-1][1]

    for i in range(len(envelope) - 1):
        x1, y1 = envelope[i]
        x2, y2 = envelope[i + 1]
        if x1 <= x <= x2:
            # Linear interpolation:
            t = (x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
    # Fallback (should not happen if envelope is non-empty)
    return envelope[-1][1]

def parametric_search(terrain, envelope):

    min_h2 = float('inf')
    best_location = None

    # Iterate over each terrain vertex (discrete candidate location for the second tower)
    for vx, ground in terrain:
        # Evaluate the envelope at this x-coordinate:
        env_y = evaluate_envelope(envelope, vx)
        # The extra height required is the gap between envelope and ground:
        required_h2 = max(0, env_y - ground)


        if required_h2 <= min_h2:
            min_h2 = required_h2
            best_location = (vx, ground)
    return round(min_h2), best_location

def plot_solution(terrain: List[Point], first_tower: int, h1: float, upper_envelope: List[Point], h2: float, best_location: Point,non:list[Segment]):

    plt.figure(figsize=(12, 6))
    x_vals, y_vals = zip(*terrain)
    plt.plot(x_vals, y_vals, 'ko-', label="Terrain")


    u_x= first_tower
    u_top =0
    for x,y in terrain:
        if u_x==x:
            u_top=y
            break

    plt.plot([u_x, u_x], [u_top, u_top+h1], 'r-', linewidth=2, label="First Tower")
    plt.scatter([u_x], [h1+u_top], color="red", s=100, zorder=10)

    u=0
    for (x,y),(x2,y2) in non:
        if u==0:
            u=1

            plt.plot([x, x2], [y, y2], linestyle='--', color='blue', linewidth=2, label="Non-visible Chains")
        else:
            plt.plot([x, x2], [y, y2], linestyle='--', color='blue', linewidth=2)


    x_upper, y_upper = zip(*upper_envelope)
    plt.plot(x_upper, y_upper, 'g--', alpha=0.7, linewidth=2, label="Upper Envelope")


    if best_location is not None:
       v_x, v_y = best_location
       plt.plot([v_x, v_x], [v_y, v_y + h2], 'b-', linewidth=2, label="Second Tower")
       plt.scatter([v_x], [v_y + h2], color="blue", s=100, zorder=10)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Decision Procedure")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    #terrain=[(0,3),(1,5),(2,4),(3,0),(5,5)]
    #terrain = [(0, 15), (1, 5), (2, 4), (3, 0), (5, 0), (7, 5), (8, 6), (9, 4)]
    #terrain = [(0, 2), (1, 3), (2, 0), (3, 4), (4, 2), (5, 8), (7, 6), (9, 1), (10, 10)]
    #terrain = [(0, 2), (1, 3), (2, 0), (3, 4), (4, 2), (5, 8), (7, 6), (9, 1), (10, 10), (12, 0), (14, 10), (16, 5),(17,7),(18,5),(19,5),(20,5),(40,5)]# >2 k <7 tote sazi to upper envelope
    #terrain = [(0, 7), (1, 3), (2, 8), (3, 3), (4, 0), (5, 8), (7, 0), (9, 1), (10, 0)]
    terrain=[(0, 0), (1, 5), (2, 4), (3, 0),(4,5),(6,0) ,(7, 5), (8, 6), (9, 4),(10,8),(11,0),(12,0)]
    u_idx = 1
    h1 = 5

    non_visible,lr = find_non_visible_chains(terrain, u_idx, h1)

    if not non_visible:
        print("Entire terrain is visible. Second watchtower height: 0 at any vertex.")
    else:

        domain_max = terrain[-1][0]
        upper_envelope = compute_upper_envelope_discrete_graham_modified(non_visible,domain_max,terrain,lr)
        temp=sorted(upper_envelope)

        h2, best_location = parametric_search(terrain, temp)

        print("Non-visible chains:", non_visible)
        print("Upper envelope:", temp)
        print(f"Optimal height of second tower: {h2}, Location: {best_location}")

        end_time = time.time()  # End the timer
        execution_time = end_time - start_time

        print(f"Execution time: {execution_time:.6f} seconds")


        plot_solution(terrain, u_idx, h1, temp, h2, best_location, non_visible)


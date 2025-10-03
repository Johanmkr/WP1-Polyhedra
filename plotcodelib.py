import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def extreme_rays(A, tol=1e-9, device="cpu"):
    """
    Find candidate extreme rays for the recession cone of {x | A x <= c}.
    Only uses A (c is irrelevant).
    Works reliably in 2D.
    
    A: (m, n) array-like (should be 2D with n=2).
    tol: feasibility tolerance
    device: torch device
    """
    m, n = A.shape
    rays = []
    for i in range(m):
        ai = A[i]
        perp = np.array([ai[1], -ai[0]])
        for r in [perp,-perp]:
            r = r / np.linalg.norm(r)
            if np.all(A@r <= tol):
                rays.append(tuple(np.round(r, 12))) # Reduce numerical duplicates by rounding
    rays = list(set(rays))
    return np.array(rays)

def constraint_plot_limits(A, c, margin=0.5):
    points = []
    m = A.shape[0]
    for i in range(m):
        for j in range(i+1, m):
            Ai, ci = A[i], c[i][0]
            Aj, cj = A[j], c[j][0]
            M = np.vstack([Ai, Aj])
            if np.linalg.matrix_rank(M) == 2:  # avoid parallel lines
                b = np.array([ci, cj])
                x = np.linalg.solve(M, b)
                points.append(x)
    if not points:
        return (-1, 1), (-1, 1)  # default
    pts = np.array(points)
    xmin, xmax = pts[:,0].min() - margin, pts[:,0].max() + margin
    ymin, ymax = pts[:,1].min() - margin, pts[:,1].max() + margin
    return (xmin, xmax), (ymin, ymax)


def plot_region(region, ax=None, feasible_pt=None, anchor=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    
    # Compute global limits from constraints
    plot_xlim, plot_ylim = constraint_plot_limits(region.A, region.c, margin=0.5)
    
    arrow_scaling = ((plot_xlim[1] - plot_xlim[0])+plot_ylim[1] - plot_ylim[0]) / 2 * 0.10
    
    xs = np.linspace(plot_xlim[0], plot_xlim[1], 400)
    ys = np.linspace(plot_ylim[0], plot_ylim[1], 400)
    XX, YY = np.meshgrid(xs, ys)

    for i in range(region.A.shape[0]):
        a1, a2 = region.A[i]
        ci = region.c[i][0]
        
        # Boundary line
        if abs(a2) > 1e-12:
            line_ys = (ci - a1 * xs) / a2
            line_handle, = ax.plot(xs, line_ys, label=f"{a1:.2f} x1 + {a2:.2f} x2 <= {ci:.2f}")
            color = line_handle.get_color()
        else:
            x_vert = ci / a1
            line_handle = ax.axvline(x_vert, label=f"{a1:.2f} x1 <= {ci:.2f}")
            color = line_handle.get_color()
        
        # Half-plane shading
        mask = (a1 * XX + a2 * YY) <= ci
        ax.contourf(XX, YY, mask, levels=[0.5, 1], colors=[color], alpha=0.08)
        
        normal = np.array([a1, a2], dtype=float)
        if feasible_pt is not None:
            if a1 * feasible_pt[0] + a2 * feasible_pt[1] <= ci:
                feasible_dir = normal
            else:
                feasible_dir = -normal
        else:
            feasible_dir = normal
        feasible_dir /= np.linalg.norm(feasible_dir)
        
        # Midpoint on line in plot range
        xm_center = 0.5 * (plot_xlim[0] + plot_xlim[1])
        if abs(a2) > 1e-12:
            ym_center = (ci - a1 * xm_center) / a2
        else:
            xm_center = ci / a1
            ym_center = 0.5 * (plot_ylim[0] + plot_ylim[1])
        
        
        arrow_start = np.array([xm_center, ym_center]) + 0.05 * feasible_dir
        arrow_end   = arrow_start + arrow_scaling * feasible_dir
        ax.annotate("", xy=arrow_end, xytext=arrow_start,
                    arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=6))
    
    # Polygon of feasible region
    if region.V_representation.shape[0] > 0:
        poly = Polygon(region.V_representation, closed=True, alpha=0.3, facecolor='red', edgecolor='black', hatch=r"//")
        ax.add_patch(poly)
        ax.scatter(region.V_representation[:, 0], region.V_representation[:, 1],
                   color='red', zorder=5, label="Vertices")
    
    
    ### RAYS
    rays = extreme_rays(region.A)
    # Draw rays
    for r in rays:
        r = r / np.linalg.norm(r)  # normalize
        for v in region.V_representation:
            ax.arrow(v[0], v[1], arrow_scaling*r[0], arrow_scaling*r[1],
                    head_width=0.02, head_length=0.01, fc='blue', ec='blue', lw=5,
                    length_includes_head=True)

    
    ax.set_xlim(*plot_xlim)
    ax.set_ylim(*plot_ylim)
    # ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Feasible Half-planes in $\\mathbb{R}^2$"+f"for q={region.q}")
    return ax
    

# plot_region(feasible_regions[0], anchor=None, ray_len=0.4)
def best_subplot_shape(N):
    n = math.ceil(math.sqrt(N))  # number of columns
    m = math.ceil(N / n)         # number of rows
    return (m, n)


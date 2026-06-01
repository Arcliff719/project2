"""
Generate synthetic graph images as test dataset for Project 2.
Creates various chart/plot images simulating "graph sets" to be
encoded and transmitted over the wireless communication system.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import os


def generate_graph_1(save_path: str, size: tuple = (512, 512)):
    """Line plot with multiple series — typical scientific graph."""
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    x = np.linspace(0, 4 * np.pi, 200)
    ax.plot(x, np.sin(x), "b-", linewidth=2, label="sin(x)")
    ax.plot(x, np.cos(x), "r--", linewidth=2, label="cos(x)")
    ax.plot(x, 0.5 * np.sin(2 * x), "g-.", linewidth=1.5, label="0.5·sin(2x)")
    ax.set_xlabel("x (radians)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Trigonometric Functions")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def generate_graph_2(save_path: str, size: tuple = (512, 512)):
    """Bar chart — categorical data."""
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    categories = ["A", "B", "C", "D", "E", "F", "G", "H"]
    values = [23, 45, 56, 78, 32, 67, 43, 51]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(categories)))
    ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Category")
    ax.set_ylabel("Value")
    ax.set_title("Performance by Category")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def generate_graph_3(save_path: str, size: tuple = (512, 512)):
    """Scatter plot with regression."""
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    np.random.seed(42)
    x = np.random.uniform(0, 10, 80)
    y = 2.5 * x + 5 + np.random.normal(0, 3, 80)
    ax.scatter(x, y, alpha=0.7, c="#2196F3", edgecolors="#1565C0", linewidths=0.3)
    # Regression line
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(0, 10, 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), "r-", linewidth=2, label=f"y={coeffs[0]:.2f}x+{coeffs[1]:.2f}")
    ax.set_xlabel("X Variable")
    ax.set_ylabel("Y Variable")
    ax.set_title("Correlation Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def generate_graph_4(save_path: str, size: tuple = (512, 512)):
    """Pie chart."""
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    labels = ["System A", "System B", "System C", "System D", "Others"]
    sizes = [35, 25, 20, 12, 8]
    explode = (0.05, 0.05, 0, 0, 0)
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct="%1.1f%%", shadow=True, startangle=90)
    ax.set_title("Market Share Distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def generate_graph_5(save_path: str, size: tuple = (512, 512)):
    """Heatmap / 2D data."""
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    np.random.seed(123)
    data = np.random.rand(12, 12)
    im = ax.imshow(data, cmap="hot", aspect="auto", interpolation="bilinear")
    ax.set_xlabel("X Index")
    ax.set_ylabel("Y Index")
    ax.set_title("Signal Intensity Map")
    plt.colorbar(im, ax=ax, label="Intensity")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def generate_graph_6(save_path: str, size: tuple = (512, 512)):
    """Subplots with multiple chart types."""
    fig, axes = plt.subplots(2, 2, figsize=(size[0]/100, size[1]/100), dpi=100)

    # Subplot 1: Line
    x = np.linspace(0, 10, 100)
    axes[0, 0].plot(x, np.exp(-x/3) * np.sin(4*x), "teal", linewidth=1.5)
    axes[0, 0].set_title("Damped Oscillation")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2: Histogram
    data = np.random.normal(0, 1, 500)
    axes[0, 1].hist(data, bins=25, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0, 1].set_title("Distribution")
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].set_ylabel("Frequency")

    # Subplot 3: Stem
    n = np.arange(0, 30)
    axes[1, 0].stem(n, np.abs(np.sinc(n/5)), linefmt="C1-", markerfmt="C1o", basefmt="k-")
    axes[1, 0].set_title("Sampled Sinc")
    axes[1, 0].set_xlabel("n")
    axes[1, 0].set_ylabel("|sinc(n/5)|")

    # Subplot 4: Errorbar
    means = [3.2, 4.8, 5.1, 6.0, 5.5]
    stds = [0.5, 0.8, 0.4, 0.7, 0.6]
    axes[1, 1].errorbar(range(len(means)), means, yerr=stds, fmt="o-",
                        capsize=5, color="darkorange", ecolor="gray")
    axes[1, 1].set_title("Measurement Results")
    axes[1, 1].set_xlabel("Trial")
    axes[1, 1].set_ylabel("Value")

    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def generate_graph_7(save_path: str, size: tuple = (512, 512)):
    """Box plot with multiple groups."""
    rng = np.random.default_rng(2026)
    data = [
        rng.normal(62, 8, 120),
        rng.normal(71, 7, 120),
        rng.normal(68, 10, 120),
        rng.normal(77, 6, 120),
        rng.normal(73, 9, 120),
    ]
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    box = ax.boxplot(data, patch_artist=True, tick_labels=["A", "B", "C", "D", "E"])
    for patch, color in zip(box["boxes"], plt.cm.Pastel1(np.linspace(0, 1, len(data)))):
        patch.set_facecolor(color)
    ax.set_xlabel("Experiment Group")
    ax.set_ylabel("Score")
    ax.set_title("Distribution Comparison")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def generate_graph_8(save_path: str, size: tuple = (512, 512)):
    """Filled contour plot."""
    x = np.linspace(-3, 3, 160)
    y = np.linspace(-3, 3, 160)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X**2 + Y**2) / (1 + 0.15 * (X**2 + Y**2))
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    contour = ax.contourf(X, Y, Z, levels=24, cmap="coolwarm")
    ax.contour(X, Y, Z, levels=10, colors="black", linewidths=0.35, alpha=0.55)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_title("Contour Field")
    plt.colorbar(contour, ax=ax, label="Value")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def generate_sine_grating(save_path: str, size: tuple = (512, 512)):
    """Sinusoidal grating — classic test pattern for image coding."""
    x = np.linspace(0, 4 * np.pi, size[1])
    y = np.linspace(0, 4 * np.pi, size[0])
    X, Y = np.meshgrid(x, y)
    img = (np.sin(X) * np.sin(Y) * 127 + 128).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def generate_zone_plate(save_path: str, size: tuple = (512, 512)):
    """Zone plate (Fresnel pattern) — tests frequency response of codec."""
    x = np.linspace(-1, 1, size[1])
    y = np.linspace(-1, 1, size[0])
    X, Y = np.meshgrid(x, y)
    r2 = X**2 + Y**2
    img = (0.5 * (1 + np.cos(np.pi * r2 * 80)) * 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def generate_all(directory: str = "graph"):
    """Generate all test graphs."""
    os.makedirs(directory, exist_ok=True)

    generators = [
        ("graph_line_plot", generate_graph_1),
        ("graph_bar_chart", generate_graph_2),
        ("graph_scatter", generate_graph_3),
        ("graph_pie_chart", generate_graph_4),
        ("graph_heatmap", generate_graph_5),
        ("graph_subplots", generate_graph_6),
        ("graph_box_plot", generate_graph_7),
        ("graph_contour", generate_graph_8),
    ]

    for name, gen_func in generators:
        path = os.path.join(directory, f"{name}.png")
        gen_func(path)
        print(f"  Generated: {path}")

    # Test patterns for codec evaluation
    sine_path = os.path.join(directory, "test_sine_grating.png")
    generate_sine_grating(sine_path)
    print(f"  Generated: {sine_path}")

    zp_path = os.path.join(directory, "test_zone_plate.png")
    generate_zone_plate(zp_path)
    print(f"  Generated: {zp_path}")


if __name__ == "__main__":
    generate_all()

"""Lab 1: activation functions. Run with the ai4beg Python environment."""

from pathlib import Path
import argparse
import csv

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


eps = 1e-3


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.maximum(0, z)


def slope(f, z):
    # Central difference, using the same step size as the supplied skeleton.
    return (f(z + eps) - f(z - eps)) / (2 * eps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-8, 8, 401)
    probe = np.array([-6., -1., 1., 6.])
    functions = (sigmoid, tanh, relu)
    styles = ('-', '--', ':')

    for filename, title, ylabel, derivative in (
        ('outputs.png', 'Activation functions', 'Output', False),
        ('slopes.png', 'Slopes (central difference)', 'Slope', True),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for f, style in zip(functions, styles):
            y = slope(f, x) if derivative else f(x)
            ax.plot(x, y, style, label=f.__name__, linewidth=2)
        ax.set(xlabel='x', ylabel=ylabel, title=title, xlim=(-8, 8))
        ax.axhline(0, color='gray', linewidth=0.6)
        ax.axvline(0, color='gray', linewidth=0.6)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(args.output_dir / filename, dpi=200)
        plt.close(fig)

    rows = [[f.__name__, *slope(f, probe)] for f in functions]
    with (args.output_dir / 'slope_table.csv').open(
        'w', encoding='utf-8-sig', newline=''
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(['Function / x', '-6', '-1', '1', '6'])
        for name, *values in rows:
            writer.writerow([name, *[f'{v:.6f}' for v in values]])

    # Six decimals keep the small, positive tanh slope at x=+/-6 visible.
    print(f"{'Function':<12}{'-6':>12}{'-1':>12}{'1':>12}{'6':>12}")
    for name, *values in rows:
        print(f'{name:<12}' + ''.join(f'{v:12.6f}' for v in values))

    explanation = (
        '在 x=6 时，ReLU 的斜率最大，约为 1，而 sigmoid 和 tanh 已接近饱和，斜率接近 0，因此对输入变化的响应很弱。\n'
        'ReLU 在 x=0 处左右两侧斜率分别为 0 和 1，导数不存在，而中心差分跨过这个拐点会得到 0.5，这只是数值估计值。\n'
    )
    (args.output_dir / 'explanation.txt').write_text(explanation, encoding='utf-8')
    print('\n' + explanation)
    print('Results:', args.output_dir.resolve())


if __name__ == '__main__':
    main()

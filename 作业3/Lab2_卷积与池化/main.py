"""Lab 2: valid cross-correlation and max pooling, with zero-based columns."""

from pathlib import Path
import argparse
import csv
import numpy as np


K = np.array([[-1, 1, 0]] * 3)


def maps(col):
    x = np.zeros((8, 8), dtype=int)
    x[:, col] = 1
    y = np.zeros((6, 6), dtype=int)
    p = np.zeros((3, 3), dtype=int)
    for i in range(6):
        for j in range(6):
            # Valid cross-correlation: multiply directly, without flipping K.
            y[i, j] = np.sum(x[i:i + 3, j:j + 3] * K)
    for i in range(3):
        for j in range(3):
            p[i, j] = np.max(y[2*i:2*i + 2, 2*j:2*j + 2])
    return y, p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    output = []
    for col in (3, 4, 5):
        y, p = maps(col)
        feature_col = int(np.argmax(y[0]))
        pooled_col = int(np.argmax(p[0]))
        records.append([col, feature_col, pooled_col])
        output.append(f'Line column = {col} (zero-based)\n'
                      f'Feature map y (6 x 6):\n{y}\n'
                      f'Pooled map p (3 x 3):\n{p}\n')
    arrays = '\n'.join(output)
    print(arrays)
    (args.output_dir / 'arrays.txt').write_text(arrays, encoding='utf-8')
    with (args.output_dir / 'peak_table.csv').open(
        'w', encoding='utf-8-sig', newline=''
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(['Line column', 'Feature-peak column', 'Pooled-peak column'])
        writer.writerows(records)

    check = (
        '所有行号、列号均从 0 开始。\n'
        '当竖线在第 3 列时，取 x[0:3, 2:5]，计算 y[0, 2]。\n'
        '窗口 W = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]\n'
        '滤波器 K = [[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]]\n'
        '逐元素相乘后求和：\n'
        'y[0, 2] = (0×(-1)+1×1+0×0)\n'
        '         + (0×(-1)+1×1+0×0)\n'
        '         + (0×(-1)+1×1+0×0) = 3\n'
    )
    explanation = (
        '竖线从第 3 列移到第 4 列时，特征图的正峰从第 2 列移到第 3 列，两者都在同一个池化窗口的列范围 [2, 3] 内，因此池化峰仍在第 1 列，池化输出不变。\n'
        '竖线从第 4 列移到第 5 列时，特征图的正峰从第 3 列移到第 4 列，跨过池化窗口的边界，池化峰从第 1 列移到第 2 列，因此池化输出发生变化。\n'
    )
    (args.output_dir / 'manual_check.txt').write_text(check, encoding='utf-8')
    (args.output_dir / 'explanation.txt').write_text(explanation, encoding='utf-8')
    print('Line | Feature peak | Pooled peak (zero-based)')
    for row in records:
        print(*row, sep=' | ')
    print('\n' + check + '\n' + explanation)
    print('Results:', args.output_dir.resolve())


if __name__ == '__main__':
    main()

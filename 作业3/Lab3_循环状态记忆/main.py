"""Lab 3: memory in a scalar recurrent state."""

from pathlib import Path
import csv
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


delays = np.array([0, 4, 12, 32])
threshold = 0.1


def final_state(first, L, rho):
    h = 0.0
    for x in [first] + [0.0] * int(L):
        h = rho * h + x
    return h


def main():
    output = Path(__file__).resolve().parent / 'results'
    output.mkdir(exist_ok=True)
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rho in (0.5, 0.9, 1.0):
        gaps = []
        for L in delays:
            a = final_state(1., L, rho)
            b = final_state(-1., L, rho)
            gaps.append(abs(a - b))
        passed = delays[np.asarray(gaps) >= threshold]
        longest = int(passed.max()) if len(passed) else None
        rows.append((rho, gaps, longest))
        print(f'rho={rho}: gaps={gaps}; longest tested detectable delay={longest}')
        ax.semilogy(delays, gaps, 'o-', label=str(rho))
    ax.axhline(threshold, color='gray', ls='--', label='threshold = 0.1')
    ax.set(xlabel='Delay L', ylabel='Final-state gap',
           title='Final-state gap versus delay', xticks=delays)
    ax.grid(alpha=0.25)
    ax.legend(title='rho')
    fig.tight_layout()
    fig.savefig(output / 'gap_vs_delay.png', dpi=200)
    plt.close(fig)

    with (output / 'gap_table.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['rho', 'L=0', 'L=4', 'L=12', 'L=32',
                         'longest tested detectable delay'])
        for rho, gaps, longest in rows:
            writer.writerow([rho, *[f'{g:.10g}' for g in gaps], longest])

    explanation = (
        '以最终状态差值至少 0.1 为可检测标准，rho=0.5、0.9、1.0 的最长合格测试延迟分别为 4、12、32。\n'
        '只看最终状态的正负号会得到不同答案：这三种 rho 都大于 0，两个状态在测试延迟内始终一正一负，即使差值已经小于 0.1。\n'
        '这个标量模型只有一个线性状态，不能表示更复杂输入之间的相互作用。\n'
    )
    (output / 'explanation.txt').write_text(explanation, encoding='utf-8')
    print(explanation)


if __name__ == '__main__':
    main()

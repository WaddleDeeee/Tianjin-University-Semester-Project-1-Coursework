"""随堂作业三：读取 task2.py 的 PCA 矩阵，比较 K-means 和 DBSCAN。

直接运行：python task3.py
指定输入：python task3.py --input output/task2/A01T_task2.xlsx
手动 DBSCAN：python task3.py --eps 6.6 --min-samples 5
安装依赖：python -m pip install numpy pandas scipy scikit-learn matplotlib openpyxl python-docx pillow

默认输出至 output/task3/，重复运行自动编号。提交文档为 A01T_task3.docx，
内含方法、二维图、参数选择、混淆矩阵、分析和全部 trial 的簇标签。
簇标签与 DBSCAN 参数搜索另存 CSV，便于核对。

聚类函数只接收 PCA 数值矩阵；真实标签在聚类完成后才用于评价。
DBSCAN 自动选择仅依赖非噪声覆盖率与内部轮廓系数，不使用真实标签。
原始 PCA 得分不逐列再次标准化，保留任务二的方差权重和欧氏几何。

方法文档：https://scikit-learn.org/stable/modules/clustering.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

LABEL_NAMES = {1: "左手", 2: "右手", 3: "双脚", 4: "舌头"}


def load_science():
    try:
        global np, pd, plt, KMeans, DBSCAN, NearestNeighbors
        global silhouette_score, linear_sum_assignment
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.optimize import linear_sum_assignment
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise RuntimeError("缺少依赖，请执行：python -m pip install numpy pandas scipy scikit-learn "
                           "matplotlib openpyxl python-docx pillow") from exc


def read_pca(path):
    """PC 列名是输入契约，标签和各类 trial 编号永远不会进入聚类矩阵。"""
    table = pd.read_excel(path, sheet_name="pca_features")
    columns = [c for c in table if re.fullmatch(r"PC\d+", str(c))]
    columns.sort(key=lambda name: int(name[2:]))
    if len(columns) < 2 or not {"trial_id", "label"}.issubset(table):
        raise ValueError("需有 pca_features 表，含至少两列 PC 得分、trial_id 和 label。")
    x = table[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if len(x) < 5 or not np.isfinite(x).all() or len(np.unique(x, axis=0)) < 4:
        raise ValueError("PCA 数据须有限、至少五行，且至少四个不同样本。请检查任务二输出。")
    if table.trial_id.isna().any() or not table.trial_id.is_unique:
        raise ValueError("trial_id 必须非空且唯一。")
    metadata = table.drop(columns=columns).copy()
    # 此处只检查数据契约；实际聚类与参数选择不接收这些标签。
    if not np.isin(metadata.label, list(LABEL_NAMES)).all():
        raise ValueError("真实标签只能为 1、2、3、4。")
    if set(metadata.label) != set(LABEL_NAMES):
        raise ValueError("输入必须保留四种运动想象类别。")
    return x, columns, metadata


def internal_metrics(x, cluster):
    assigned = cluster >= 0
    n_clusters = len(np.unique(cluster[assigned]))
    sil = None
    if 2 <= n_clusters < int(assigned.sum()):
        sil = float(silhouette_score(x[assigned], cluster[assigned], metric="euclidean"))
    return {"n_clusters": n_clusters, "n_noise": int((~assigned).sum()),
            "coverage": float(assigned.mean()), "silhouette_assigned": sil}


def cluster_features(x, seed=42, eps=None, min_samples=5):
    """无监督阶段：没有 y 参数，也不通过标签决定 DBSCAN 的 eps/minPts。"""
    kmeans = KMeans(n_clusters=4, init="k-means++", n_init=30,
                    max_iter=500, random_state=seed, algorithm="lloyd")
    km_labels = kmeans.fit_predict(x)
    candidates, labels_by_candidate = [], []
    distance_curves = {}
    sizes = [min_samples] if eps is not None else sorted({5, 10, max(5, x.shape[1] + 1), 20})
    for size in sizes:
        if not 2 <= size <= len(x):
            continue
        # 传入 x，近邻中包含自身；这与 DBSCAN min_samples 包含自身一致。
        distances = NearestNeighbors(n_neighbors=size, metric="euclidean").fit(x).kneighbors(x)[0][:, -1]
        distance_curves[size] = np.sort(distances)
        radii = [eps] if eps is not None else np.unique(np.quantile(
            distances, [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]))
        for radius in radii:
            if not np.isfinite(radius) or radius <= 0:
                continue
            cluster = DBSCAN(eps=float(radius), min_samples=size, metric="euclidean").fit_predict(x)
            row = {"candidate_id": len(candidates), "eps": float(radius), "min_samples": size,
                   **internal_metrics(x, cluster)}
            candidates.append(row)
            labels_by_candidate.append(cluster)
    if not candidates:
        raise ValueError("没有有效的 DBSCAN 参数。请提供正的 eps 和合法的 min-samples。")
    # 先要求至少一半样本有簇标签，避免用极少数高密度点得到虚高轮廓系数。
    eligible = [r for r in candidates if r["silhouette_assigned"] is not None and r["coverage"] >= .5]
    if eps is not None:
        chosen, reason = candidates[0], "使用命令行指定的参数，未使用标签调参。"
    elif eligible:
        chosen = max(eligible, key=lambda r: (r["silhouette_assigned"], r["coverage"], -r["eps"], -r["min_samples"]))
        reason = "在至少两个非噪声簇且覆盖率不低于 50% 的候选中，选非噪声轮廓系数最高者。"
    else:
        multiple = [r for r in candidates if r["silhouette_assigned"] is not None]
        if multiple:
            chosen = max(multiple, key=lambda r: (r["coverage"], r["silhouette_assigned"], -r["eps"]))
            reason = "无至少两个簇且覆盖率达到 50% 的候选，改选多簇候选中覆盖率最高者，再按轮廓系数打破并列。"
        else:
            chosen = max(candidates, key=lambda r: (r["coverage"], -r["eps"], -r["min_samples"]))
            reason = "搜索范围内未得到可计算轮廓系数的多簇划分，保留覆盖率最高结果，如实报告退化情况。"
    for row in candidates:
        row["selected"] = row["candidate_id"] == chosen["candidate_id"]
    return {"kmeans": km_labels, "dbscan": labels_by_candidate[chosen["candidate_id"]],
            "kmeans_centers": kmeans.cluster_centers_, "chosen": chosen,
            "selection_reason": reason, "candidates": candidates,
            "distance_curve": distance_curves[chosen["min_samples"]]}


def evaluate(x, y, cluster):
    """后验核对：Hungarian 一一对应匹配；噪声和未匹配簇计为未分配。"""
    classes = [1, 2, 3, 4]
    clusters = sorted(int(v) for v in np.unique(cluster) if v >= 0)
    counts = np.zeros((len(clusters), 4), dtype=int)
    for i, c in enumerate(clusters):
        for j, label in enumerate(classes):
            counts[i, j] = np.sum((cluster == c) & (y == label))
    mapping = {}
    if len(clusters):
        row, col = linear_sum_assignment(-counts)
        mapping = {clusters[int(i)]: classes[int(j)] for i, j in zip(row, col)}
    mapped = np.array([mapping.get(int(c), -1) for c in cluster], dtype=int)
    confusion = np.zeros((4, 5), dtype=int)
    for label, pred in zip(y, mapped):
        confusion[int(label) - 1, int(pred) - 1 if pred >= 1 else 4] += 1
    assigned = cluster >= 0
    internal = internal_metrics(x, cluster)
    result = {**internal, "accuracy_all": float(np.mean(mapped == y)),
              "accuracy_assigned": float(np.mean(mapped[assigned] == y[assigned])) if assigned.any() else None,
              "mapping": mapping, "confusion": confusion.tolist(),
              "contingency": {str(c): counts[i].tolist() for i, c in enumerate(clusters)},
              "class_noise": {str(label): int(np.sum((y == label) & (cluster == -1))) for label in classes}}
    return result, mapped


def draw_figures(x, clustering, metadata, evaluations, folder):
    """仅绘制作业要求的两个聚类结果，颜色用于区分不同簇。"""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, method in zip(axes, ["kmeans", "dbscan"]):
        cluster = clustering[method]
        for c in sorted(np.unique(cluster)):
            mask = cluster == c
            ax.scatter(x[mask, 0], x[mask, 1], s=18,
                       color="#888888" if c == -1 else plt.get_cmap("tab10")(int(c) % 10),
                       marker="x" if c == -1 else "o",
                       label="noise" if c == -1 else f"cluster {c}")
        ax.set(title="K-means" if method == "kmeans" else "DBSCAN", xlabel="PC1", ylabel="PC2")
        ax.set_xlim(x[:, 0].min() - 1, x[:, 0].max() + 1)
        ax.set_ylim(x[:, 1].min() - 1, x[:, 1].max() + 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "cluster_comparison.png", dpi=180)
    plt.close(fig)


def format_value(value, percent=False):
    if value is None:
        return "不适用"
    if percent:
        return f"{value:.2%}"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def build_report(payload_path):
    """按作业三要求输出普通黑白 Word，不加入其他评价指标和参考资料。"""
    try:
        from docx import Document
        from docx.shared import Cm, Pt, RGBColor
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    except ImportError as exc:
        raise RuntimeError("生成 Word 需要 python-docx：python -m pip install python-docx") from exc
    data = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    folder = Path(payload_path).parent
    ev, chosen = data["evaluations"], data["chosen"]
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2)
    section.left_margin = section.right_margin = Cm(2)
    for name in ["Normal", "Title", "Heading 1", "Caption"]:
        style = doc.styles[name]
        style.font.name = "Arial Unicode MS"
        fonts = style._element.get_or_add_rPr().get_or_add_rFonts()
        for key in ["ascii", "hAnsi", "eastAsia", "cs"]:
            fonts.set(qn("w:" + key), "Arial Unicode MS")
        for key in ["asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"]:
            fonts.attrib.pop(qn("w:" + key), None)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.size = Pt(11)
        style.font.bold = False
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.space_before = Pt(0)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.05
    doc.styles["Title"].font.size = Pt(14)
    doc.styles["Heading 1"].font.size = Pt(12)
    doc.styles["Heading 1"].paragraph_format.space_before = Pt(8)
    doc.styles["Caption"].font.size = Pt(10)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    page = OxmlElement("w:fldSimple"); page.set(qn("w:instr"), "PAGE")
    footer._p.append(page)

    def p(text):
        return doc.add_paragraph(text)

    def heading(text, newpage=False):
        para = doc.add_heading(text, level=1)
        para.paragraph_format.page_break_before = newpage
        return para

    def table(headers, rows, widths, size=10):
        tab = doc.add_table(rows=1, cols=len(headers))
        tab.autofit = False
        for col, width in zip(tab.columns, widths):
            col.width = Cm(width)
        for cell, value in zip(tab.rows[0].cells, headers):
            cell.text = str(value)
        for values in rows:
            for cell, value in zip(tab.add_row().cells, values):
                cell.text = str(value)
        borders = OxmlElement("w:tblBorders")
        for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
            line = OxmlElement("w:" + edge)
            for key, value in [("val", "single"), ("sz", "4"), ("color", "000000")]:
                line.set(qn("w:" + key), value)
            borders.append(line)
        tab._tbl.tblPr.append(borders)
        header = OxmlElement("w:tblHeader")
        tab.rows[0]._tr.get_or_add_trPr().append(header)
        for row in tab.rows:
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for i, cell in enumerate(row.cells):
                cell.width = Cm(widths[i])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(1)
                    para.paragraph_format.space_before = Pt(1)
                    para.paragraph_format.line_spacing = 1
                    for run in para.runs:
                        run.font.size = Pt(size)
                        run.font.bold = False
                        run.font.color.rgb = RGBColor(0, 0, 0)
        return tab

    title = doc.add_paragraph("随堂作业三", "Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    heading("一 方法")
    p(f"使用作业二输出的 PCA 特征矩阵，共 {data['n_samples']} 次试验、{data['n_features']} 个特征。"
      "聚类时不输入类别标签，得到簇标签以后再与真实类别核对。")
    p("距离采用欧氏距离。作业二已对原始特征做标准化和 PCA，直接使用 PCA 得分可以保留主成分的方差权重。"
      "本次使用全部主成分进行聚类，二维图只展示 PC1 和 PC2。")
    p(f"K-means 设 k=4，使用 k-means++ 初始化，重复初始化 30 次，最多迭代 500 次，随机种子 {data['seed']}。")
    p(f"DBSCAN 调整 eps 和 minPts。minPts 的候选为 {data['min_samples_grid']}，"
      f"eps 参考近邻距离的分位数，共尝试 {data['n_candidates']} 组参数。"
      f"最终 eps={chosen['eps']:.6f}，minPts={chosen['min_samples']}。{data['selection_reason']}")
    heading("二 聚类结果")
    metrics = [("簇数（不含噪声）", "n_clusters", False), ("噪声数", "n_noise", False),
               ("匹配准确率", "accuracy_all", True), ("轮廓系数", "silhouette_assigned", False)]
    table(["项目", "K-means", "DBSCAN"],
          [[name, format_value(ev['kmeans'][key], pct), format_value(ev['dbscan'][key], pct)]
           for name, key, pct in metrics], [7, 5, 5])
    p("准确率按簇与真实类别的一一最优匹配计算，簇号不能直接当作类别编号。"
      "DBSCAN 噪声在准确率中算错，轮廓系数只计算非噪声样本。")
    doc.add_picture(str(folder / "cluster_comparison.png"), width=Cm(17))
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    p("图1 两种方法的二维聚类结果，灰色叉号为 DBSCAN 噪声。")
    heading("三 结果分析")
    for method, name in [("kmeans", "K-means"), ("dbscan", "DBSCAN")]:
        result = ev[method]
        matrix = result["confusion"]
        p(name + " 混淆矩阵，行是真实类别，列是匹配后的类别：")
        table(["真实类别", "左手", "右手", "双脚", "舌头", "未分配"],
              [[LABEL_NAMES[i + 1]] + matrix[i] for i in range(4)], [4, 2.6, 2.6, 2.6, 2.6, 2.6])
        errors = sorted([(matrix[i][j], i + 1, j + 1)
                         for i in range(4) for j in range(4) if i != j], reverse=True)
        descriptions = [f"{LABEL_NAMES[a]}分为{LABEL_NAMES[b]} {count} 次" for count, a, b in errors[:3] if count]
        p("主要错分为：" + ("；".join(descriptions) if descriptions else "没有已分配样本的类别错分") + "。")
        if method == "dbscan":
            p("各类被标为噪声的次数为：" + "，".join(
                f"{LABEL_NAMES[i]} {result['class_noise'][str(i)]} 次" for i in range(1, 5)) + "。")
    p("K-means 把全部样本分到四个簇，优点是简单，结果比较容易解释。但它倾向于球形或凸形划分，"
      "对于非球形分布可能把同一类拆开，也可能把不同类合到一起。本次左手和右手的错分较多，"
      "说明四个自然聚类簇与四种运动想象没有很好地对应。")
    p("DBSCAN 可以形成非球形簇，也能把稀疏样本标为噪声。缺点是对 eps 和 minPts 比较敏感，"
      "密度不均时，较小的 eps 会丢弃稀疏区域，较大的 eps 又容易把不同区域连在一起。"
      f"本次只得到 {ev['dbscan']['n_clusters']} 个非噪声簇，噪声占 {1 - ev['dbscan']['coverage']:.2%}，"
      "说明在本次特征和参数范围内，按密度分组没有较好地还原四类。")
    assignments = data["assignments"]
    for offset in range(0, len(assignments), 46):
        heading("簇标签" if offset == 0 else "簇标签续表", newpage=True)
        p("真类：1左手，2右手，3双脚，4舌头。K 为 K-means 簇号，D 为 DBSCAN 簇号，-1 为噪声。"
          "trial 为原试验编号，缺号为作业一剔除的试验。")
        batch = assignments[offset:offset + 46]
        rows = []
        for i in range(23):
            values = []
            for index in [i, i + 23]:
                if index < len(batch):
                    row = batch[index]
                    values.extend([row['trial_id'], row['label'], row['kmeans_cluster'], row['dbscan_cluster']])
                else:
                    values.extend(["", "", "", ""])
            rows.append(values)
        table(["trial", "真类", "K", "D", "trial", "真类", "K", "D"], rows,
              [2.4, 2.1, 2, 2, 2.4, 2.1, 2, 2], size=10.5)
    for root in [doc.styles.element, doc.element]:
        for border in list(root.iter(qn("w:pBdr"))):
            border.getparent().remove(border)
    output = folder / data["report_name"]
    doc.save(output)
    return output


def save_results(x, source, columns, metadata, clustering, folder, seed):
    evaluations, assignments = {}, metadata.copy()
    # 评价发生在全部参数与簇标签已固定以后。
    for method in ["kmeans", "dbscan"]:
        result, mapped = evaluate(x, metadata.label.to_numpy(dtype=int), clustering[method])
        evaluations[method] = result
        assignments[f"{method}_cluster"] = clustering[method]
        assignments[f"{method}_matched_class"] = mapped
    assignments.to_csv(folder / "cluster_labels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(clustering["candidates"]).to_csv(folder / "dbscan_search.csv", index=False, encoding="utf-8-sig")
    metrics = pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, dict) and k != "confusion"}
                            | {"method": method} for method, result in evaluations.items()])
    metrics.to_csv(folder / "metrics.csv", index=False, encoding="utf-8-sig")
    chosen = {k: v for k, v in clustering["chosen"].items() if k != "selected"}
    payload = {"input_name": source.name, "subject": source.stem.split("_")[0],
               "n_samples": len(metadata), "n_features": len(columns),
               "features": columns, "seed": seed, "chosen": chosen,
               "min_samples_grid": sorted({r["min_samples"] for r in clustering["candidates"]}),
               "n_candidates": len(clustering["candidates"]), "selection_reason": clustering["selection_reason"],
               "evaluations": evaluations, "assignments": assignments.to_dict(orient="records"),
               "report_name": source.stem.replace("_task2", "") + "_task3.docx"}
    path = folder / "task3_results.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return evaluations, path


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=base / "output" / "task2" / "A01T_task2.xlsx")
    parser.add_argument("--output-dir", type=Path, help="新的输出目录，默认 output/task3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, help="手动指定正的 DBSCAN eps；省略时自动无监督搜索")
    parser.add_argument("--min-samples", type=int, default=5, help="手动 eps 对应的 minPts，包含自身")
    parser.add_argument("--report-python", type=Path, help="可选的文档阶段 Python 解释器，需安装 python-docx")
    parser.add_argument("--render-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.render_report:
        print(build_report(args.render_report))
        return
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed 必须在 0 到 2**32-1 之间。")
    load_science()
    if args.eps is not None and (not np.isfinite(args.eps) or args.eps <= 0):
        parser.error("eps 必须为有限正数。")
    source = args.input.expanduser().resolve()
    if not source.is_file():
        parser.error(f"任务二输出不存在：{source}，请先运行 task2.py。")
    x, columns, metadata = read_pca(source)
    if not 2 <= args.min_samples <= len(x):
        parser.error("min-samples 必须在 2 到样本数之间。")
    print(f"读取任务二 PCA：{len(x)} trial × {len(columns)} 维。", flush=True)
    clustering = cluster_features(x, args.seed, args.eps, args.min_samples)
    folder = (args.output_dir or base / "output" / "task3").expanduser().resolve()
    if args.output_dir and folder.exists():
        raise FileExistsError(f"输出目录已存在，请指定新目录：{folder}")
    default, version = folder, 2
    while folder.exists():
        folder = default.with_name(f"{default.name}_{version}")
        version += 1
    folder.mkdir(parents=True)
    evaluations, payload_path = save_results(x, source, columns, metadata, clustering, folder, args.seed)
    draw_figures(x, clustering, metadata, evaluations, folder)
    if args.report_python:
        subprocess.run([str(args.report_python.expanduser().resolve()), str(Path(__file__).resolve()),
                        "--render-report", str(payload_path)], check=True)
    else:
        build_report(payload_path)
    for method, result in evaluations.items():
        print(f"{method}: {result['n_clusters']} 簇，{result['n_noise']} 噪声，"
              f"覆盖 {result['coverage']:.2%}，全体匹配准确率 {result['accuracy_all']:.2%}")
    print(f"DBSCAN：eps={clustering['chosen']['eps']:.6f}，minPts={clustering['chosen']['min_samples']}。")
    print(f"完成，结果目录：{folder}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileExistsError, subprocess.CalledProcessError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)

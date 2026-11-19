"""随堂作业二：承接 main.py 的 Excel，做特征选择、PCA 和对比分析。

直接运行：python task2.py
指定输入：python task2.py --input output/A01T_features.xlsx --top-n 30 --variance 0.90
依赖安装：python -m pip install numpy pandas scipy scikit-learn openpyxl matplotlib pillow
默认输出至脚本旁 output/task2/；重复运行自动建立 task2_2/ 等新目录。

实现 Filter（四分类 ANOVA F 排序）和 Wrapper（决策树 RFE）两种特征选择。
PCA 在标准化后的原始特征上独立进行，取累计方差贡献率 >= 阈值的最小维数。
筛选后的矩阵保留原始单位；PCA 得分来自标准化空间。所有表保留 trial 标识和标签。
本作业是全数据的描述性分析；不将训练内的类簇分离程度解释为泛化分类性能。

官方方法说明：
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # 保存图像，在 PyCharm/无图形桌面环境都能运行。
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import RFE, f_classif
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from openpyxl.styles import Alignment, Font, PatternFill
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请在当前 Python 环境运行：\n"
        "python -m pip install numpy pandas scipy scikit-learn openpyxl matplotlib pillow"
    ) from exc

LABELS = {1: "left_hand", 2: "right_hand", 3: "feet", 4: "tongue"}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
METHOD_NAMES = {"original": "Original", "filter": "ANOVA Filter", "rfe": "Tree RFE", "pca": "PCA"}


def read_task1(path):
    """只从任务一的定义表选择特征，避免把 trial 编号、标签混入模型。"""
    with pd.ExcelFile(path) as book:
        if not {"features", "feature_definitions"}.issubset(book.sheet_names):
            raise ValueError("请输入任务一生成的 Excel，需含 features、feature_definitions 表。")
        table = pd.read_excel(book, sheet_name="features")
        definitions = pd.read_excel(book, sheet_name="feature_definitions")
    if "feature" not in definitions or "label" not in table or "trial_id" not in table:
        raise ValueError("缺少 feature 定义、label 或 trial_id。")
    features = definitions["feature"].tolist()
    forbidden = {"trial_id", "run_id", "trial_in_run", "run_onset_sample", "onset_sample",
                 "end_sample", "label", "class_name", "official_reject"}
    if (not features or len(set(features)) != len(features)
            or forbidden.intersection(features) or any(name not in table for name in features)):
        raise ValueError("特征定义为空、重复、包含元数据，或与 features 表不一致。")
    if table.trial_id.isna().any() or not table.trial_id.is_unique:
        raise ValueError("trial_id 必须非空且唯一。")
    y = pd.to_numeric(table["label"], errors="raise").to_numpy()
    if not np.isin(y, list(LABELS)).all():
        raise ValueError("类别标签必须是 1、2、3、4。")
    classes, counts = np.unique(y, return_counts=True)
    if set(classes) != set(LABELS) or counts.min() < 2:
        raise ValueError("要求保留四个类别，且每类至少两个 trial。")
    metadata = table.drop(columns=features).copy()
    metadata["class_name"] = metadata["label"].map(LABELS)
    x = table[features].apply(pd.to_numeric, errors="raise")
    return metadata, x, definitions


def prepare_features(x):
    """处理非有限值和常量列，再按列做 z-score；记录可复现的参数。"""
    clean = x.replace([np.inf, -np.inf], np.nan)
    missing = clean.isna().sum()
    medians = clean.median()
    if medians.isna().any():
        raise ValueError(f"存在整列缺失特征：{list(medians[medians.isna()].index)}")
    clean = clean.fillna(medians)
    # 不以绝对方差阈值删除：µV、µV²、CSP 的尺度不同，绝对阈值不合适。
    constant = clean.nunique(dropna=False) <= 1
    usable = clean.loc[:, ~constant]
    if usable.shape[1] < 2:
        raise ValueError("去除常量后少于两个特征，无法完成筛选和二维可视化。")
    scaler = StandardScaler()
    z = scaler.fit_transform(usable)
    audit = pd.DataFrame({"feature": clean.columns, "missing_filled": missing.to_numpy(),
                          "fill_median": medians.to_numpy(), "constant_removed": constant.to_numpy(),
                          "mean": clean.mean().to_numpy(), "std_ddof0": clean.std(ddof=0).to_numpy()})
    return clean, usable, z, audit, scaler


def analyze(x, y, top_n=30, variance=0.90, seed=42):
    clean, usable, z, audit, scaler = prepare_features(x)
    if not 2 <= top_n <= usable.shape[1]:
        raise ValueError(f"top-n 应在 2 到 {usable.shape[1]} 之间。")
    if not 0 < variance <= 1:
        raise ValueError("variance 须大于 0 且不超过 1。")
    with warnings.catch_warnings():
        # 对于类内方差为零的非恒定特征，ANOVA 可给出正无穷 F，仍排最前。
        warnings.simplefilter("ignore", RuntimeWarning)
        f_scores, p_values = f_classif(z, y)
    if np.isnan(f_scores).any() or np.isnan(p_values).any():
        raise ValueError("ANOVA 得到未定义结果，请检查特征的类内/类间方差。")
    order = np.argsort(-f_scores, kind="stable")
    filter_idx = order[:top_n]
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)

    # 每轮删除一个特征，重要性为当前决策树的 impurity importance。
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5,
                                  class_weight="balanced", random_state=seed)
    rfe = RFE(tree, n_features_to_select=top_n, step=1)
    rfe.fit(z, y)
    rfe_idx = np.flatnonzero(rfe.support_)
    selected_importance = rfe.estimator_.feature_importances_
    # 输出时按最终树重要性排序，筛选集合仍由 RFE 确定。
    rfe_idx = rfe_idx[np.argsort(-selected_importance, kind="stable")]
    importance = np.zeros(len(order))
    importance[np.flatnonzero(rfe.support_)] = selected_importance

    pca = PCA(svd_solver="full")
    all_scores = pca.fit_transform(z)
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    # clamp 避免 variance=1 时浮点累计值略小于 1 导致下标超界。
    n_pc = min(int(np.searchsorted(cumulative, variance, side="left") + 1), len(ratios))
    pc_names = [f"PC{i + 1:03d}" for i in range(n_pc)]
    pca_scores = all_scores[:, :n_pc]
    views = {"original": z, "filter": z[:, filter_idx], "rfe": z[:, rfe_idx], "pca": pca_scores}
    rows = []
    for method, values in views.items():
        rows.append({"method": METHOD_NAMES[method], "n_trials": len(y),
                     "n_features": values.shape[1],
                     "reduction_ratio": 1 - values.shape[1] / x.shape[1],
                     "silhouette": silhouette_score(values, y, metric="euclidean")})
    comparison = pd.DataFrame(rows)
    # 原始特征计数单独保留；常量删除属于预处理，不属于 Filter/RFE 选择数量。
    comparison.loc[0, "n_features"] = x.shape[1]
    comparison.loc[0, "reduction_ratio"] = 0.0

    selection = pd.DataFrame({"feature": usable.columns, "anova_F": f_scores,
                              "anova_p": p_values, "anova_rank": ranks,
                              "filter_selected": np.isin(np.arange(len(order)), filter_idx),
                              "rfe_rank": rfe.ranking_, "rfe_selected": rfe.support_,
                              "rfe_final_importance": importance})
    selection = selection.sort_values("anova_rank", kind="stable").reset_index(drop=True)
    spectrum = pd.DataFrame({"component": [f"PC{i + 1:03d}" for i in range(len(ratios))],
                             "variance": pca.explained_variance_, "explained_ratio": ratios,
                             "cumulative_ratio": cumulative,
                             "retained": np.arange(len(ratios)) < n_pc})
    # 每个子图的坐标明确记录，避免把不同投影的散点距离当成同一尺度比较。
    projections = {"original": z[:, :2], "filter": z[:, filter_idx[:2]],
                   "rfe": z[:, rfe_idx[:2]], "pca": all_scores[:, :2]}
    axes = {"original": [str(c) for c in usable.columns[:2]],
            "filter": [str(c) for c in usable.columns[filter_idx[:2]]],
            "rfe": [str(c) for c in usable.columns[rfe_idx[:2]]],
            "pca": [f"PC1 ({ratios[0]:.1%})", f"PC2 ({ratios[1]:.1%})"]}
    return {"clean": clean, "usable": usable, "z": z, "audit": audit,
            "filter_features": clean.loc[:, usable.columns[filter_idx]],
            "rfe_features": clean.loc[:, usable.columns[rfe_idx]],
            "pca_features": pd.DataFrame(pca_scores, columns=pc_names),
            "comparison": comparison, "selection": selection, "spectrum": spectrum,
            "projections": projections, "axes": axes, "n_pc": n_pc,
            "pca": pca, "scaler": scaler, "rfe": rfe}


def make_figures(result, y, folder, threshold):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for ax, (method, coords) in zip(axs.flat, result["projections"].items()):
        for label, color in zip(LABELS, COLORS):
            points = coords[y == label]
            ax.scatter(points[:, 0], points[:, 1], s=24, alpha=.7, color=color,
                       label=LABELS[label], edgecolors="none")
        dims = int(result["comparison"].loc[result["comparison"].method == METHOD_NAMES[method], "n_features"].iloc[0])
        ax.set_title(f"{METHOD_NAMES[method]} ({dims} dimensions)", loc="left", fontweight="bold")
        ax.set_xlabel(result["axes"][method][0])
        ax.set_ylabel(result["axes"][method][1])
        ax.grid(alpha=.15)
    axs[0, 0].legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle("Motor imagery classes: feature selection and PCA", fontsize=15)
    scatter_path = folder / "feature_comparison.png"
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    spec = result["spectrum"]
    numbers = np.arange(1, len(spec) + 1)
    axs[0].bar(numbers, spec.explained_ratio * 100, width=.8, color="#4878A8")
    axs[0].set(xlabel="Principal component", ylabel="Explained variance (%)", title="Variance per component")
    axs[1].plot(numbers, spec.cumulative_ratio * 100, color="#4878A8", linewidth=2)
    axs[1].axhline(threshold * 100, linestyle="--", color="#D55E00", label=f"Target {threshold:.0%}")
    k = result["n_pc"]
    actual = float(spec.cumulative_ratio.iloc[k - 1])
    axs[1].axvline(k, linestyle=":", color="#555555")
    axs[1].scatter([k], [actual * 100], color="#D55E00", zorder=3)
    axs[1].annotate(f"{k} PCs: {actual:.2%}", xy=(k, actual * 100), xytext=(12, -32),
                    textcoords="offset points", arrowprops={"arrowstyle": "-", "color": "#555555"})
    axs[1].set(xlabel="Number of components", ylabel="Cumulative variance (%)", ylim=(0, 103),
               title="Cumulative variance and retained dimension")
    axs[1].legend(frameon=False, loc="lower right")
    variance_path = folder / "pca_variance.png"
    fig.savefig(variance_path, dpi=180)
    plt.close(fig)
    return scatter_path, variance_path


def method_notes(source, result, top_n, threshold, seed):
    k = result["n_pc"]
    actual = float(result["spectrum"].cumulative_ratio.iloc[k - 1])
    comp = result["comparison"].set_index("method")
    original_sil = float(comp.loc["Original", "silhouette"])
    best = comp.silhouette.idxmax()
    return [
        ("输入", source.name + "；features 为 trial 表，feature_definitions 指定特征列"),
        ("目标", "承接任务一，比较原始、ANOVA Filter、Tree RFE、PCA 四种表示。未额外执行可选的 LDA。"),
        ("标签", "1=左手，2=右手，3=双脚，4=舌头。trial_id、run_id、label 等元数据不参与筛选或 PCA。"),
        ("数据检查", f"{len(result['clean'])} 个 trial，原有 {result['clean'].shape[1]} 个特征。"
         f"非有限值用各列中位数填补 {int(result['audit'].missing_filled.sum())} 个，"
         f"去除 {int(result['audit'].constant_removed.sum())} 个常量特征。整列缺失或非法标签会报错。"),
        ("标准化", "对非常量特征使用 StandardScaler：z=(x-列均值)/总体标准差，ddof=0。"
         "不同量纲因此可比较，避免 µV² 特征支配 PCA。筛选矩阵仍输出原始单位，PCA 表输出得分。"),
        ("Filter", f"四分类单因素 ANOVA，按 F 从大到小保留 Top {top_n}；相同分数按原始列顺序。"
         "F 衡量类间差异相对类内波动，适合离散四分类标签，不把 1–4 当连续有序量。"
         "p 值仅作描述，未作多重比较校正，也不作为本次筛选阈值。"),
        ("Wrapper", f"决策树 RFE 保留 {top_n} 个特征，每轮重训并删除重要性最低的一个特征。"
         f"max_depth=5，min_samples_leaf=5，class_weight=balanced，random_state={seed}。"
         "固定 Top N 以便与 Filter 对照；未把该数量当成调参得出的最优维度。"),
        ("RFE 排名", "rank=1 为最终选中；最终树重要性只对选中的特征有意义。"
         "被删除特征的 rfe_final_importance 记为 0，删除顺序见 rfe_rank。"),
        ("PCA", f"在标准化后的全部非常量原始特征上独立做 full SVD，不先做 Filter 或 RFE。"
         f"保留达到累计方差贡献率 ≥{threshold:.0%} 的最少 {k} 个主成分，实际累计贡献率 {actual:.4%}。"
         "PCA 最大化总体方差，不保证类别分离一定增强。"),
        ("二维图", "feature_comparison.png：原始特征取标准化后的前两列；Filter 取 ANOVA 前两列；"
         "RFE 取最终树重要性最高的两列；PCA 展示 PC1、PC2。各图坐标不同，"
         "只能直观观察重叠，不能仅凭图形宣称可分性改善；二维坐标保存在 projections_2d。"),
        ("二维 PCA", f"PC1+PC2 仅解释 {result['spectrum'].explained_ratio.iloc[:2].sum():.2%} 方差；"
         f"Excel 的 pca_features 保留完整的 {k} 维，并非只保留图中两维。"),
        ("紧凑性比较", "comparison 的 silhouette 为给定真实类别的欧氏距离轮廓系数，范围 [-1,1]，"
         "越高通常表示类内更紧凑、类间更分离。原始/筛选表示使用 z-score 空间；"
         "PCA 使用其得分且不再次逐主成分标准化。常量列不参与距离。"),
        ("分析结果", f"原始轮廓系数 {original_sil:.5f}；"
         + "；".join(f"{name} {float(comp.loc[name, 'silhouette']):.5f}"
                     f"（相对原始 {float(comp.loc[name, 'silhouette']) - original_sil:+.5f}）"
                     for name in ["ANOVA Filter", "Tree RFE", "PCA"])
         + f"。四组中 {best} 的轮廓系数最高。正向变化是当前数据的描述性结果。"),
        ("评估范围", "本次按作业要求在任务一的整张特征表上做描述性筛选和降维。"
         "任务一 CSP 已使用全体标签，因此本表不能直接用于无偏分类性能评估；"
         "以后若做分类验证，须把 CSP、填补、标准化、筛选、PCA 都放进训练折内重新拟合。"),
        ("保留维度的理由", f"Filter/RFE 统一保留 {top_n} 个以控制特征预算并比较选择结果；"
         f"PCA 按 ≥{threshold:.0%} 方差阈值确定最小维度 {k}，不根据二维图主观选择。"),
        ("来源", "任务一 Excel（见输入）；ANOVA: https://scikit-learn.org/stable/modules/generated/"
         "sklearn.feature_selection.f_classif.html ；RFE: https://scikit-learn.org/stable/modules/generated/"
         "sklearn.feature_selection.RFE.html ；PCA: https://scikit-learn.org/stable/modules/generated/"
         "sklearn.decomposition.PCA.html"),
    ]


def export_result(metadata, definitions, result, source, folder, args):
    """Python 程序自行生成作业 Excel；每个 trial 的元数据与标签始终同行。"""
    notes = method_notes(source, result, args.top_n, args.variance, args.seed)
    selection = result["selection"].merge(definitions, on="feature", how="left", validate="one_to_one")
    # Excel 不支持数值无穷；仅在 F 为无穷时明确写文本，不伪装成有限分数。
    selection["anova_F"] = selection.anova_F.map(lambda v: "inf" if np.isposinf(v) else v)
    projections = metadata.copy()
    for method, coords in result["projections"].items():
        projections[f"{method}_x"] = coords[:, 0]
        projections[f"{method}_y"] = coords[:, 1]
    frames = {"comparison": result["comparison"],
              "filter_features": pd.concat([metadata, result["filter_features"]], axis=1),
              "rfe_features": pd.concat([metadata, result["rfe_features"]], axis=1),
              "pca_features": pd.concat([metadata, result["pca_features"]], axis=1),
              "selection_details": selection,
              "pca_variance": result["spectrum"],
              "preprocessing": result["audit"],
              "projections_2d": projections,
              "method_notes": pd.DataFrame(notes, columns=["item", "description"])}
    output = folder / f"{source.stem.replace('_features', '')}_task2.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.sheet_view.showGridLines = False
            sheet.freeze_panes = "A2" if name not in ("comparison", "method_notes") else None
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.fill = PatternFill("solid", fgColor="243E5A")
                cell.font = Font(name="Arial", size=10, color="FFFFFF", bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center")
            sheet.row_dimensions[1].height = 28
            for column in sheet.columns:
                letter = column[0].column_letter
                header = str(column[0].value)
                width = max(14, len(header) + 3)
                if header in ("feature", "definition", "description"):
                    width = {"feature": 30, "definition": 65, "description": 95}[header]
                elif name == "comparison" and header == "method":
                    width = 23
                sheet.column_dimensions[letter].width = min(width, 95)
                for cell in column[1:]:
                    cell.font = Font(name="Arial", size=10, color="202020")
                    wrap = header in ("description", "definition") or (name == "method_notes" and header == "item")
                    cell.alignment = Alignment(vertical="center", wrap_text=wrap)
                    if isinstance(cell.value, float):
                        if "ratio" in header:
                            cell.number_format = "0.00%"
                        elif header in ("anova_p", "anova_F") or (0 < abs(cell.value) < 1e-4):
                            cell.number_format = "0.0000E+00"
                        else:
                            cell.number_format = "0.00000"
            if name == "method_notes":
                sheet.column_dimensions["A"].width = 24
                for row in range(2, sheet.max_row + 1):
                    length = len(str(sheet.cell(row, 2).value))
                    sheet.row_dimensions[row].height = max(32, 16 * (int(np.ceil(length / 42)) + 1))
            if name == "selection_details":
                for row in range(2, sheet.max_row + 1):
                    sheet.row_dimensions[row].height = 45
        writer.sheets["comparison"].sheet_properties.tabColor = "243E5A"
    # 回读实际保存文件，检查标签和矩阵维度；不只检查内存结果。
    with pd.ExcelFile(output) as book:
        for method in ("filter", "rfe", "pca"):
            saved = pd.read_excel(book, f"{method}_features")
            expected = result[f"{method}_features"]
            if not np.array_equal(saved["label"].to_numpy(), metadata.label.to_numpy()):
                raise RuntimeError(f"{method} 导出标签未对齐。")
            if not np.array_equal(saved.trial_id.to_numpy(), metadata.trial_id.to_numpy()):
                raise RuntimeError(f"{method} 导出 trial 未对齐。")
            if not np.allclose(saved[expected.columns].to_numpy(), expected.to_numpy(), rtol=1e-10, atol=1e-12):
                raise RuntimeError(f"{method} 导出矩阵未通过数值核对。")
    report = folder / "task2_report.txt"
    report.write_text("随堂作业二：特征选择与降维\n\n" +
                      "\n\n".join(f"{item}\n{description}" for item, description in notes), encoding="utf-8")
    # JSON 保存配置便于复现；不序列化不安全的 pickle 模型。
    (folder / "parameters.json").write_text(json.dumps(
        {"input": str(source), "top_n": args.top_n, "pca_variance_target": args.variance,
         "random_seed": args.seed, "pca_components": result["n_pc"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return output, report


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=base / "output" / "A01T_features.xlsx")
    parser.add_argument("--output-dir", type=Path, help="新的输出目录，默认为 output/task2，已存在时自动编号")
    parser.add_argument("--top-n", type=int, default=30, help="Filter 和 RFE 各保留的特征数")
    parser.add_argument("--variance", type=float, default=0.90, help="PCA 累计方差阈值，默认 0.90")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    if not source.is_file():
        parser.error(f"任务一 Excel 不存在：{source}。请先运行 main.py 或用 --input 指定。")
    if args.seed < 0 or args.seed > 2**32 - 1:
        parser.error("seed 应为 0 到 2**32-1 的整数。")
    metadata, x, definitions = read_task1(source)
    print(f"读取任务一：{len(x)} trial × {x.shape[1]} 特征。", flush=True)
    result = analyze(x, metadata.label.to_numpy(), args.top_n, args.variance, args.seed)
    folder = (args.output_dir or base / "output" / "task2").expanduser().resolve()
    if args.output_dir and folder.exists():
        raise FileExistsError(f"输出目录已存在，请指定新目录：{folder}")
    default_folder = folder
    version = 2
    while folder.exists():
        folder = default_folder.with_name(f"{default_folder.name}_{version}")
        version += 1
    folder.mkdir(parents=True)
    make_figures(result, metadata.label.to_numpy(), folder, args.variance)
    output, report = export_result(metadata, definitions, result, source, folder, args)
    print(result["comparison"].to_string(index=False))
    print(f"PCA 保留 {result['n_pc']} 维，累计贡献率 "
          f"{result['spectrum'].cumulative_ratio.iloc[result['n_pc'] - 1]:.4%}。")
    print(f"结果：{output}\n说明：{report}\n图像：{folder / 'feature_comparison.png'}\n"
          f"方差贡献图：{folder / 'pca_variance.png'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileExistsError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)

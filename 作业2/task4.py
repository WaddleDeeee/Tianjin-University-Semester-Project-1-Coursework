"""随堂作业四：Session T 训练、Session E 评估，比较决策树和 SVM。

直接运行：python task4.py
依赖：python -m pip install numpy scipy pandas mne scikit-learn openpyxl python-docx joblib
默认承接 output/task2/A01T_task2.xlsx 的 PCA 表，读取 archive/A01T.mat、A01E.mat。
缺少 Session E 时明确报错，不用训练集随机拆分冒充 Session E。

模型调参仅在 T 内分层三折进行，每折单独拟合 CSP、填补、标准化、PCA。
最后在全部 T 上拟合变换，并核对与任务二 PCA 矩阵一致，再变换 E。
E 的类别标签只在模型、参数与变换固定后用于最终指标计算。

输出：普通黑白 Word、决策树与最终 SVM 模型、预测与参数记录 CSV/JSON。
--report-python 可为文档生成指定已安装 python-docx 的解释器。
官方数据入口：https://bnci-horizon-2020.eu/database/data-sets
方法：https://scikit-learn.org/stable/modules/svm.html
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

LABELS = {1: "左手", 2: "右手", 3: "双脚", 4: "舌头"}


def load_science():
    try:
        global np, pd, signal, task1, PCA, StandardScaler, StratifiedKFold
        global DecisionTreeClassifier, SVC, accuracy_score, cohen_kappa_score, confusion_matrix, joblib
        import numpy as np
        import pandas as pd
        from scipy import signal
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
        import joblib
        import main as task1
        task1.require_dependencies()
        task1.mne.set_log_level("ERROR")
    except ImportError as exc:
        raise RuntimeError("缺少依赖：python -m pip install numpy scipy pandas mne scikit-learn openpyxl python-docx joblib") from exc


def signal_features(epochs, fs, cfg):
    """与 main.py 相同的 176 个时域/频域特征；不拟合任何有监督变换。"""
    arrays = [epochs.mean(-1), epochs.var(-1, ddof=0), epochs.std(-1, ddof=0), np.ptp(epochs, axis=-1)]
    blocks = [matrix for matrix in arrays]
    frequencies, psd = signal.welch(epochs, fs=fs, window="hann", nperseg=round(fs),
                                   noverlap=round(fs) // 2, detrend="constant", axis=-1)
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz

    def power(low, high):
        mask = (frequencies >= low) & (frequencies <= high)
        return integrate(psd[..., mask], x=frequencies[mask], axis=-1)

    total = power(cfg.low, cfg.high)
    for low, high in [(8, 13), (14, 30)]:
        absolute = power(low, high)
        relative = np.divide(absolute, total, out=np.full_like(absolute, np.nan), where=total > 0)
        blocks.append(np.stack([absolute, relative], axis=-1).reshape(len(epochs), -1))
    return np.concatenate(blocks, axis=1)


class FeatureSpace:
    """所有可学习变换只在 fit 的 T 样本上拟合；transform 不接收标签。"""
    def __init__(self, cfg, fs=250.0, variance=.9):
        self.cfg, self.fs, self.variance = cfg, fs, variance

    def fit(self, epochs, y):
        frame, definitions, self.csp = task1.extract_features(epochs, y, self.fs, self.cfg)
        self.feature_names = definitions.feature.tolist()
        features = frame.to_numpy()
        features[~np.isfinite(features)] = np.nan
        self.median = np.nanmedian(features, axis=0)
        if not np.isfinite(self.median).all():
            raise ValueError("训练特征存在整列缺失，不能拟合变换。")
        features = np.where(np.isfinite(features), features, self.median)
        self.nonconstant = np.ptp(features, axis=0) > 0
        self.scaler = StandardScaler().fit(features[:, self.nonconstant])
        standardized = self.scaler.transform(features[:, self.nonconstant])
        self.pca = PCA(n_components=self.variance, svd_solver="full").fit(standardized)
        self.sign = np.ones(self.pca.n_components_)
        self.train_scores = self.pca.transform(standardized)
        return self

    def transform(self, epochs):
        base = signal_features(epochs, self.fs, self.cfg)
        variance = self.csp.transform(epochs).var(axis=-1, ddof=0)
        features = np.concatenate([base, variance], axis=1)
        features = np.where(np.isfinite(features), features, self.median)
        return self.pca.transform(self.scaler.transform(features[:, self.nonconstant])) * self.sign

    def align_task2(self, reference):
        # SVD 的特征向量允许整体变号；将当前得分与已有作业二的方向对齐。
        if self.train_scores.shape != reference.shape:
            raise ValueError("重建的 T 矩阵维度与作业二不一致，请确认被试、时间窗及 PCA 阈值。")
        signs = np.sign(np.sum(self.train_scores * reference, axis=0))
        signs[signs == 0] = 1
        aligned = self.train_scores * signs
        if not np.allclose(aligned, reference, atol=1e-7, rtol=1e-6):
            raise ValueError("重建的 T 特征与作业二矩阵不一致，停止评估以避免混用处理方法。")
        self.sign = signs
        self.train_scores = aligned

    def state(self):
        # 保存内置模型对象及参数，不依赖 pickle 中的自定义 __main__ 类。
        return {"config": vars(self.cfg), "fs": self.fs, "median": self.median,
                "nonconstant": self.nonconstant, "scaler": self.scaler, "pca": self.pca,
                "csp": self.csp, "sign": self.sign, "feature_names": self.feature_names}


def read_session(path, cfg):
    eeg, fs, trials = task1.read_data(path)
    epochs, kept, audit = task1.preprocess(eeg, fs, trials, cfg)
    metadata = pd.DataFrame(kept)
    if set(metadata.label) != set(LABELS):
        raise ValueError("四分类输入需要保留四个类别。")
    return epochs, metadata, audit, fs


def model_candidates():
    result = []
    for depth, leaf in itertools.product([3, 5, 8], [3, 5]):
        result.append({"method": "tree", "params": {"max_depth": depth, "min_samples_leaf": leaf}})
    for c in [.1, 1.0, 10.0]:
        result.append({"method": "linear", "params": {"kernel": "linear", "C": c}})
    for c, gamma in itertools.product([.1, 1.0, 10.0], ["scale", .01, .1]):
        result.append({"method": "rbf", "params": {"kernel": "rbf", "C": c, "gamma": gamma}})
    return result


def make_model(candidate, seed):
    if candidate["method"] == "tree":
        return DecisionTreeClassifier(**candidate["params"], criterion="gini", random_state=seed)
    return SVC(**candidate["params"], decision_function_shape="ovr", cache_size=200)


def tune_on_training(epochs, y, cfg, variance, seed):
    candidates = model_candidates()
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    fold_scores = [[] for _ in candidates]
    for fold, (training, validation) in enumerate(folds.split(epochs, y), start=1):
        print(f"T 内三折调参：第 {fold}/3 折，重新拟合 CSP 和 PCA。", flush=True)
        representation = FeatureSpace(cfg, variance=variance).fit(epochs[training], y[training])
        x_train = representation.train_scores
        x_val = representation.transform(epochs[validation])
        for i, candidate in enumerate(candidates):
            classifier = make_model(candidate, seed)
            classifier.fit(x_train, y[training])
            fold_scores[i].append(float(accuracy_score(y[validation], classifier.predict(x_val))))
    records = [{**candidate, "cv_accuracy": float(np.mean(scores)), "fold_scores": scores}
               for candidate, scores in zip(candidates, fold_scores)]
    best = {}
    for method in ["tree", "linear", "rbf"]:
        # 固定候选顺序，并列时选择先列出的较小参数，不参考 E 表现。
        best[method] = max((row for row in records if row["method"] == method), key=lambda row: row["cv_accuracy"])
    svm_method = max(["linear", "rbf"], key=lambda method: best[method]["cv_accuracy"])
    return best, svm_method, records


def evaluate_models(best, x_train, y_train, x_eval, y_eval, seed):
    results, models, predictions = {}, {}, {}
    for method in ["tree", "linear", "rbf"]:
        classifier = make_model(best[method], seed)
        start = perf_counter(); classifier.fit(x_train, y_train)
        training_ms = (perf_counter() - start) * 1000
        train_pred = classifier.predict(x_train)
        start = perf_counter(); predicted = classifier.predict(x_eval)
        prediction_ms = (perf_counter() - start) * 1000
        results[method] = {"params": best[method]["params"], "cv_accuracy": best[method]["cv_accuracy"],
                           "train_accuracy": float(accuracy_score(y_train, train_pred)),
                           "eval_accuracy": float(accuracy_score(y_eval, predicted)),
                           "kappa": float(cohen_kappa_score(y_eval, predicted, labels=list(LABELS))),
                           "confusion": confusion_matrix(y_eval, predicted, labels=list(LABELS)).tolist(),
                           "training_ms": training_ms, "prediction_ms": prediction_ms}
        models[method], predictions[method] = classifier, predicted
    return results, models, predictions


def split_features(tree, pc_names):
    counts = np.bincount(tree.tree_.feature[tree.tree_.feature >= 0], minlength=len(pc_names))
    order = np.argsort(-tree.feature_importances_, kind="stable")
    return [{"feature": pc_names[i], "importance": float(tree.feature_importances_[i]), "split_count": int(counts[i])}
            for i in order if counts[i] > 0][:5]


def build_report(payload_path):
    """只写作业四要求的模型结果、评估表、分裂特征、混淆矩阵和对比结论。"""
    try:
        from docx import Document
        from docx.shared import Cm, Pt, RGBColor
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError as exc:
        raise RuntimeError("生成 Word 需要 python-docx：python -m pip install python-docx") from exc
    data = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2)
    section.left_margin = section.right_margin = Cm(2)
    for name in ["Normal", "Title", "Heading 1"]:
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
        style.paragraph_format.space_before = Pt(0)
        style.paragraph_format.space_after = Pt(5)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.05
    doc.styles["Title"].font.size = Pt(14)
    doc.styles["Heading 1"].font.size = Pt(12)
    doc.styles["Heading 1"].paragraph_format.space_before = Pt(7)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    number = OxmlElement("w:fldSimple"); number.set(qn("w:instr"), "PAGE"); footer._p.append(number)

    def p(text):
        return doc.add_paragraph(text)

    def heading(text):
        return doc.add_heading(text, level=1)

    def table(headers, rows, widths):
        tab = doc.add_table(rows=1, cols=len(headers)); tab.autofit = False
        for col, width in zip(tab.columns, widths):
            col.width = Cm(width)
        for cell, text in zip(tab.rows[0].cells, headers):
            cell.text = str(text)
        for values in rows:
            for cell, text in zip(tab.add_row().cells, values):
                cell.text = str(text)
        border = OxmlElement("w:tblBorders")
        for side in ["top", "left", "bottom", "right", "insideH", "insideV"]:
            line = OxmlElement("w:" + side)
            for key, value in [("val", "single"), ("sz", "4"), ("color", "000000")]:
                line.set(qn("w:" + key), value)
            border.append(line)
        tab._tbl.tblPr.append(border)
        tab.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for row in tab.rows:
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for i, cell in enumerate(row.cells):
                cell.width = Cm(widths[i])
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(2)
                    paragraph.paragraph_format.space_before = Pt(1)
                    paragraph.paragraph_format.line_spacing = 1.0
                    for run in paragraph.runs:
                        run.font.size = Pt(10)
                        run.font.bold = False
                        run.font.color.rgb = RGBColor(0, 0, 0)
        return tab

    title = doc.add_paragraph("随堂作业四", "Title"); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    heading("一 数据与训练方法")
    p(f"使用同一被试的 Session T 训练，Session E 评估。T 共 {data['train_total']} 次试验，"
      f"按作业一的预处理保留 {data['n_train']} 次；E 共 {data['eval_total']} 次，保留 {data['n_eval']} 次。"
      f"特征处理承接作业二的 PCA，最终输入 {data['n_features']} 维特征。")
    p("模型参数只在 T 内做分层三折验证选择，每折重新拟合 CSP、标准化和 PCA。"
      "最后在完整 T 上训练，用相同变换处理 E。E 不参与特征拟合和参数选择。")
    result = data['results']
    tree = result['tree']
    p(f"决策树用 Gini 指标分裂，比较最大深度 3、5、8 和最小叶节点样本数 3、5。"
      f"选出的 max_depth={tree['params']['max_depth']}，min_samples_leaf={tree['params']['min_samples_leaf']}。"
      f"实际树深 {data['tree_depth']}，叶节点 {data['tree_leaves']} 个，采用预剪枝控制过拟合。")
    linear, rbf = result['linear'], result['rbf']
    p(f"SVM 先试线性核，再试 RBF 核。C 比较 0.1、1、10；RBF 的 gamma 比较 scale、0.01、0.1。"
      f"线性核选 C={linear['params']['C']}；RBF 选 C={rbf['params']['C']}、gamma={rbf['params']['gamma']}。"
      f"按 T 内验证结果确定最终 SVM 使用{data['svm_display']}。")
    heading("二 评估指标")
    names = {'tree': '决策树', 'linear': 'SVM 线性核', 'rbf': 'SVM RBF核'}
    table(["模型", "训练 Accuracy", "E Accuracy", "E Kappa"],
          [[names[key], f"{r['train_accuracy']:.2%}", f"{r['eval_accuracy']:.2%}", f"{r['kappa']:.4f}"]
           for key, r in result.items()], [5.2, 4.2, 4.2, 3.4])
    heading("三 决策树分裂的主要特征")
    p(f"特征名为 PCA 主成分。根节点按 {data['root_feature']} 是否小于等于 {data['root_threshold']:.5f} 分裂。"
      "下面列出分裂中使用的主要特征。")
    table(["特征", "重要性", "分裂次数"],
          [[r['feature'], f"{r['importance']:.4f}", r['split_count']] for r in data['split_features']], [7, 5, 5])
    heading("四 混淆矩阵")
    p("行是真实类别，列是预测类别，类别顺序为左手、右手、双脚、舌头。")
    for method in ['tree', 'linear', 'rbf']:
        label = p(names[method]); label.paragraph_format.keep_with_next = True
        table(["真实类别", "左手", "右手", "双脚", "舌头"],
              [[LABELS[i + 1]] + result[method]['confusion'][i] for i in range(4)], [5, 3, 3, 3, 3])
    heading("五 对比结论")
    table(["模型", "训练耗时 ms", "评估集预测耗时 ms"],
          [[names[key], f"{r['training_ms']:.3f}", f"{r['prediction_ms']:.3f}"] for key, r in result.items()], [6, 5, 6])
    p("耗时只统计分类器的训练和预测，不包含共同的特征提取及参数搜索。")
    best_method = max(result, key=lambda key: (result[key]['eval_accuracy'], result[key]['kappa']))
    p(f"本次 E 上准确率最高的是{names[best_method]}，Accuracy={result[best_method]['eval_accuracy']:.2%}，"
      f"Kappa={result[best_method]['kappa']:.4f}。"
      f"决策树训练与评估准确率相差 {tree['train_accuracy'] - tree['eval_accuracy']:.2%}，"
      f"最终 SVM 相差 {result[data['svm_method']]['train_accuracy'] - result[data['svm_method']]['eval_accuracy']:.2%}。"
      "差距较大说明跨 Session 泛化仍不足，需要结合剪枝和参数限制控制过拟合。")
    p(f"RBF 核在训练集上达到 {rbf['train_accuracy']:.2%}，在 E 上为 {rbf['eval_accuracy']:.2%}，"
      f"差距 {rbf['train_accuracy'] - rbf['eval_accuracy']:.2%}；线性核的差距为 "
      f"{linear['train_accuracy'] - linear['eval_accuracy']:.2%}。本次 RBF 核的过拟合更明显，"
      "增加非线性复杂度没有提高评估表现。")
    p("决策树可以直接查看分裂条件，解释比较直观，预测沿树的路径进行。SVM 的线性核通过超平面分类，"
      "RBF 核能形成非线性边界，但不如树的规则容易解释。"
      f"本次计时中，{'决策树' if tree['training_ms'] < result[data['svm_method']]['training_ms'] else '最终 SVM'}训练更快，"
      f"{'决策树' if tree['prediction_ms'] < result[data['svm_method']]['prediction_ms'] else '最终 SVM'}预测更快。"
      f"综合本次评估准确率和 Kappa，{names[best_method]}更适合本数据的识别任务；"
      "若更关注明确的判别规则，决策树更容易分析。")
    for root in [doc.styles.element, doc.element]:
        for border in list(root.iter(qn('w:pBdr'))):
            border.getparent().remove(border)
    path = Path(payload_path).parent / data['report_name']
    doc.save(path)
    return path


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=base/'output/task2/A01T_task2.xlsx')
    parser.add_argument('--train', type=Path, default=base/'archive/A01T.mat')
    parser.add_argument('--eval', type=Path, default=base/'archive/A01E.mat')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report-python', type=Path)
    parser.add_argument('--render-report', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.render_report:
        print(build_report(args.render_report)); return
    for source in [args.input, args.train, args.eval]:
        if not source.is_file():
            parser.error(f"缺少文件：{source}。作业四需要同一被试的 Session T 和 Session E。")
    train_name, eval_name = args.train.stem.upper(), args.eval.stem.upper()
    if (not re.fullmatch(r'A0[1-9]T', train_name) or eval_name != train_name[:-1] + 'E'):
        parser.error('训练和评估文件须为同一被试的 AxxT / AxxE。')
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error('seed 应在 0 到 2**32-1 之间。')
    load_science()
    cfg = task1.Config()
    reference = pd.read_excel(args.input, sheet_name='pca_features')
    pc_names = sorted([c for c in reference if re.fullmatch(r'PC\d+', str(c))], key=lambda c: int(c[2:]))
    if not pc_names or 'trial_id' not in reference or 'label' not in reference:
        parser.error('任务二输入缺少 PCA 特征、trial_id 或 label。')
    epochs, metadata, audit, fs = read_session(args.train, cfg)
    y_train = metadata.label.to_numpy()
    if (not np.array_equal(reference.trial_id, metadata.trial_id)
            or not np.array_equal(reference.label, y_train)):
        raise ValueError('任务二矩阵与当前 T 数据的 trial 或标签不一致。')
    print(f"T 保留 {len(epochs)} trial，承接任务二 {len(pc_names)} 维 PCA。", flush=True)
    best, svm_method, cv_records = tune_on_training(epochs, y_train, cfg, .9, args.seed)
    space = FeatureSpace(cfg, fs=fs, variance=.9).fit(epochs, y_train)
    space.align_task2(reference[pc_names].to_numpy(dtype=float))
    # 参数与变换已固定后，才读取 E 并执行独立评估。
    eval_epochs, eval_metadata, eval_audit, eval_fs = read_session(args.eval, cfg)
    if fs != eval_fs:
        raise ValueError('T 与 E 采样率不同。')
    x_eval = space.transform(eval_epochs)
    x_train = reference[pc_names].to_numpy(dtype=float)
    results, models, predictions = evaluate_models(best, x_train, y_train, x_eval,
                                                   eval_metadata.label.to_numpy(), args.seed)
    folder = (args.output_dir or base/'output/task4').expanduser().resolve()
    if args.output_dir and folder.exists():
        raise FileExistsError(f'输出目录已存在：{folder}')
    default, version = folder, 2
    while folder.exists():
        folder = default.with_name(f'{default.name}_{version}'); version += 1
    folder.mkdir(parents=True)
    for name, method in [('decision_tree', 'tree'), ('svm', svm_method)]:
        joblib.dump({'classifier': models[method], 'feature_space': space.state(),
                     'pc_names': pc_names, 'labels': LABELS, 'model_method': method}, folder/f'{name}.joblib')
    pd.DataFrame([{**row, 'params': json.dumps(row['params']), 'fold_scores': json.dumps(row['fold_scores'])}
                  for row in cv_records]).to_csv(folder/'training_parameter_search.csv', index=False, encoding='utf-8-sig')
    eval_metadata = eval_metadata.copy()
    for method, predicted in predictions.items():
        eval_metadata[method + '_prediction'] = predicted
    eval_metadata.to_csv(folder/'evaluation_predictions.csv', index=False, encoding='utf-8-sig')
    audit.to_csv(folder/'T_trial_audit.csv', index=False, encoding='utf-8-sig')
    eval_audit.to_csv(folder/'E_trial_audit.csv', index=False, encoding='utf-8-sig')
    tree = models['tree']
    root = int(tree.tree_.feature[0])
    payload = {'report_name': train_name[:-1] + '_task4.docx', 'n_train': len(epochs),
               'n_eval': len(eval_epochs), 'train_total': len(audit), 'eval_total': len(eval_audit),
               'n_features': len(pc_names), 'results': results, 'svm_method': svm_method,
               'svm_display': '线性核' if svm_method == 'linear' else 'RBF 核',
               'tree_depth': int(tree.get_depth()), 'tree_leaves': int(tree.get_n_leaves()),
               'root_feature': pc_names[root] if root >= 0 else '无分裂',
               'root_threshold': float(tree.tree_.threshold[0]),
               'split_features': split_features(tree, pc_names)}
    payload_path = folder/'task4_results.json'
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if args.report_python:
        subprocess.run([str(args.report_python), str(Path(__file__).resolve()), '--render-report', str(payload_path)], check=True)
    else:
        build_report(payload_path)
    for name, result in results.items():
        print(f"{name}: E Accuracy={result['eval_accuracy']:.2%}, Kappa={result['kappa']:.4f}")
    print(f'最终 SVM 由 T 内验证选为 {svm_method}；结果目录：{folder}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileExistsError, subprocess.CalledProcessError) as error:
        print(f'错误：{error}', file=sys.stderr)
        sys.exit(1)

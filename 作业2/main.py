"""BCI Competition IV 2a：单个被试 Session T 脑电特征提取。

安装依赖：python -m pip install numpy scipy pandas mne scikit-learn openpyxl
运行：python main.py（默认选择 archive/A01T.mat）
指定被试：python main.py --input archive/A02T.mat
自检：python main.py --self-test （合成数据，不是实际作业结果）

时间均相对于事件 768（trial 开始）：基线 [0, 2) s，特征窗 [3, 6) s。
官方范式中提示发生在 trial 开始后 2 s，不能把 3–6 s 再加到提示时间上。
支持 BNCI 格式 MAT（data 中含 X/trial/y/fs/artifacts）以及官方 GDF。
参考：https://www.bbci.de/competition/IV/desc_2a.pdf
CSP：https://mne.tools/stable/generated/mne.decoding.CSP.html
MAT 单位/索引：https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bnci/base.py
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

LABELS = {1: "left_hand", 2: "right_hand", 3: "feet", 4: "tongue"}
CHANNELS = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1",
            "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4",
            "P1", "Pz", "P2", "POz"]


@dataclass(frozen=True)
class Config:
    start: float = 3.0
    stop: float = 6.0
    low: float = 1.0
    high: float = 40.0
    max_ptp_uv: float = 200.0  # 可调整的经验峰峰值阈值，不是数据集官方阈值
    csp_components: int = 6


def require_dependencies():
    try:
        global pd, signal, CSP, mne
        import pandas as pd
        from scipy import signal
        import mne
        from mne.decoding import CSP
        import sklearn  # noqa: F401：MNE CSP 的协方差正则化需要
        import openpyxl  # noqa: F401：导出 Excel 所需
    except ImportError as exc:
        raise RuntimeError(
            "缺少依赖，请在当前 Python 环境运行：\n"
            "python -m pip install numpy scipy pandas mne scikit-learn openpyxl"
        ) from exc


def read_gdf(path: Path):
    """返回连续 EEG（µV）、采样率和 trial 记录；排除末尾三个 EOG 通道。"""
    if path.suffix.lower() != ".gdf":
        raise ValueError("目前支持官方 .gdf；其他格式需先确认字段、事件和单位。")
    raw = mne.io.read_raw_gdf(str(path), preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    if not np.isclose(fs, 250) or len(raw.ch_names) != 25:
        raise ValueError(f"预期 IV 2a 的 250 Hz、25 通道（22 EEG + 3 EOG），"
                         f"实际 {fs} Hz、{len(raw.ch_names)} 通道。请确认数据格式。")
    # MNE 将 GDF 物理电压转换为 V；乘 1e6 后统一以 µV 计算统计量。
    eeg = raw.get_data(picks=list(range(22))) * 1e6
    event_codes = {str(code): code for code in (768, 769, 770, 771, 772, 783, 1023)}
    events, _ = mne.events_from_annotations(raw, event_id=event_codes, verbose="ERROR")
    events = events.copy()
    events[:, 0] -= raw.first_samp
    starts = events[events[:, 2] == 768, 0]
    if not len(starts):
        raise ValueError("找不到 trial 开始事件 768。")
    trials = []
    for i, onset in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else raw.n_times
        within = events[(events[:, 0] >= onset) & (events[:, 0] < end)]
        cues = within[np.isin(within[:, 2], [769, 770, 771, 772])]
        if len(cues) != 1:
            raise ValueError(f"trial {i + 1} 缺少唯一四分类标签，请提供 Session T 训练数据。")
        cue_offset = (int(cues[0, 0]) - int(onset)) / fs
        if not np.isclose(cue_offset, 2, atol=1 / fs):
            raise ValueError(f"trial {i + 1} 提示时间为 {cue_offset}s，需确认时间参考。")
        trials.append({"trial_id": i + 1, "onset_sample": int(onset),
                       "label": int(cues[0, 2] - 768),
                       "official_reject": bool(np.any(within[:, 2] == 1023))})
    return eeg, fs, trials


def read_mat(path: Path):
    """读取 BNCI 版 IV 2a MAT。X 已为 µV；trial 为 MATLAB 的 1 起始索引。

    无标签的眼动校准 run 跳过。各运动想象 run 拼接仅为统一索引，
    每个 trial 仍独立滤波，并检查边界，不跨 run 取样。
    """
    from scipy.io import loadmat

    data = loadmat(path, simplify_cells=True)
    if "data" not in data:
        raise ValueError("MAT 缺少 data 字段，不是预期的 BNCI 格式。")
    runs = data["data"]
    if isinstance(runs, dict):
        runs = [runs]
    blocks, trials = [], []
    offset = 0
    for run_id, run in enumerate(runs, start=1):
        required = {"X", "trial", "y", "fs", "artifacts"}
        if not isinstance(run, dict) or not required.issubset(run):
            raise ValueError(f"run {run_id} 缺少 BNCI 数据字段。")
        starts = np.atleast_1d(run["trial"]).ravel()
        if not len(starts):
            continue
        x = np.asarray(run["X"], dtype=float)
        labels = np.atleast_1d(run["y"]).ravel()
        artifacts = np.atleast_1d(run["artifacts"]).ravel()
        fs = float(run["fs"])
        if fs != 250 or x.ndim != 2 or x.shape[1] != 25:
            raise ValueError(f"run {run_id} 不是预期的 250 Hz、25 通道数据。")
        if len(starts) != len(labels) or len(starts) != len(artifacts):
            raise ValueError(f"run {run_id} 的 trial、y、artifacts 长度不一致。")
        if not np.isin(labels, list(LABELS)).all() or not np.isin(artifacts, [0, 1]).all():
            raise ValueError(f"run {run_id} 的类别或伪迹标签不合法。")
        if not np.isfinite(starts).all() or not np.equal(starts, np.floor(starts)).all():
            raise ValueError(f"run {run_id} 的 trial 起点必须为整数采样索引。")
        starts = starts.astype(int) - 1
        if np.any(starts < 0) or np.any(starts >= len(x)) or np.any(np.diff(starts) <= 0):
            raise ValueError(f"run {run_id} 的 trial 起点越界或未严格递增。")
        blocks.append(x[:, :22].T)
        for j, onset in enumerate(starts):
            boundary = int(starts[j + 1]) if j + 1 < len(starts) else len(x)
            trials.append({"trial_id": len(trials) + 1, "run_id": run_id,
                           "trial_in_run": j + 1,
                           "run_onset_sample": int(onset),
                           "onset_sample": offset + int(onset),
                           "end_sample": offset + boundary,
                           "label": int(labels[j]), "official_reject": bool(artifacts[j])})
        offset += len(x)
    if not trials:
        raise ValueError("MAT 中没有带标签的运动想象 trial。")
    print(f"读取 {path.name}：{len(blocks)} 个运动想象 run，{len(trials)} 个 trial。")
    return np.concatenate(blocks, axis=1), 250.0, trials


def read_data(path: Path):
    if path.suffix.lower() == ".mat":
        return read_mat(path)
    if path.suffix.lower() == ".gdf":
        return read_gdf(path)
    raise ValueError("支持 .mat（BNCI 格式）和 .gdf 文件。")


def preprocess(eeg, fs, trials, cfg):
    """先在完整 0–7 秒 trial 上滤波，再校正基线、截窗；保留丢弃记录。"""
    sos = signal.butter(4, [cfg.low, cfg.high], btype="bandpass", fs=fs, output="sos")
    epochs, kept, audit = [], [], []
    n_full = round(7 * fs)
    for trial in trials:
        reason = ""
        start = trial["onset_sample"]
        segment = eeg[:, start:start + n_full]
        if trial["official_reject"]:
            reason = "official_artifact"
        elif (start < 0 or segment.shape != (22, n_full)
              or start + n_full > trial.get("end_sample", eeg.shape[1])):
            reason = "incomplete_trial"
        elif not np.isfinite(segment).all():
            reason = "nonfinite_signal"
        if not reason:
            filtered = signal.sosfiltfilt(sos, segment, axis=-1)
            filtered -= filtered[:, :round(2 * fs)].mean(axis=-1, keepdims=True)
            epoch = filtered[:, round(cfg.start * fs):round(cfg.stop * fs)]
            if np.any(np.ptp(epoch, axis=-1) > cfg.max_ptp_uv):
                reason = "peak_to_peak_above_threshold"
            elif np.any(np.std(epoch, axis=-1) < 1e-6):
                reason = "flat_channel"
            else:
                epochs.append(epoch)
                kept.append(trial)
        audit.append({**trial, "status": "dropped" if reason else "kept", "reason": reason})
    if not epochs:
        raise ValueError("预处理后无有效 trial，请检查数据、单位、时间窗和伪迹阈值。")
    return np.stack(epochs), kept, pd.DataFrame(audit)


def extract_features(epochs, labels, fs, cfg):
    values, definitions = {}, []

    def add(name, array, family, definition, unit):
        values[name] = array
        definitions.append({"feature": name, "family": family,
                            "definition": definition, "unit": unit})

    temporal = {
        "mean": (epochs.mean(axis=-1), "窗口样本均值", "µV"),
        "variance": (epochs.var(axis=-1, ddof=0), "总体方差，ddof=0", "µV²"),
        "std": (epochs.std(axis=-1, ddof=0), "总体标准差，ddof=0", "µV"),
        "ptp": (np.ptp(epochs, axis=-1), "最大值减最小值", "µV"),
    }
    for stat, (matrix, definition, unit) in temporal.items():
        for channel, name in enumerate(CHANNELS):
            add(f"{name}_{stat}", matrix[:, channel], "time", definition, unit)

    freq, psd = signal.welch(epochs, fs=fs, window="hann", nperseg=round(fs),
                             noverlap=round(fs) // 2, detrend="constant", axis=-1)

    def band_power(low, high):
        mask = (freq >= low) & (freq <= high)
        # 兼容 NumPy 1.x 和 2.x，避免对零宽/单点频带积分。
        if mask.sum() < 2:
            raise ValueError("频率分辨率不足，无法计算频带功率。")
        integrate = getattr(np, "trapezoid", None)
        if integrate is None:
            integrate = np.trapz
        return integrate(psd[..., mask], x=freq[mask], axis=-1)

    total = band_power(cfg.low, cfg.high)
    for band, low, high in [("mu", 8, 13), ("beta", 14, 30)]:
        power = band_power(low, high)
        relative = np.divide(power, total, out=np.full_like(power, np.nan), where=total > 0)
        for channel, name in enumerate(CHANNELS):
            add(f"{name}_{band}_power", power[:, channel], "frequency",
                f"Welch PSD 在 {low}–{high} Hz 的梯形积分", "µV²")
            add(f"{name}_{band}_relative", relative[:, channel], "frequency",
                f"{band} 功率 / {cfg.low}–{cfg.high} Hz 总功率", "ratio")

    classes, counts = np.unique(labels, return_counts=True)
    if set(classes) != set(LABELS) or np.any(counts < 2):
        raise ValueError("四分类 CSP 需要保留全部四类，且每类至少两个有效 trial。")
    # CSP 是有监督变换。此处为作业特征表在全部保留 trial 上拟合；
    # 后续评估分类器时必须在每个训练折内重新拟合 CSP，避免标签泄漏。
    csp = CSP(n_components=cfg.csp_components, reg="ledoit_wolf",
              transform_into="csp_space", log=None, norm_trace=False)
    projected = csp.fit(epochs, labels).transform(epochs)
    variance = projected.var(axis=-1, ddof=0)
    for component in range(cfg.csp_components):
        add(f"CSP{component + 1:02d}_variance", variance[:, component], "spatial",
            "四分类 CSP 投影后的窗口总体方差；输入为 µV；滤波器尺度由 MNE 决定", "CSP scale")
    return pd.DataFrame(values), pd.DataFrame(definitions), csp


def check_features(features):
    """只填补非有限特征；有限极端值保留并以 IQR 统计，避免无依据裁剪。"""
    features = features.replace([np.inf, -np.inf], np.nan)
    missing = features.isna().sum()
    medians = features.median()
    if medians.isna().any():
        raise ValueError(f"特征整列缺失：{medians[medians.isna()].index.tolist()}")
    features = features.fillna(medians)
    q1, q3 = features.quantile(.25), features.quantile(.75)
    iqr = q3 - q1
    outliers = ((features < q1 - 1.5 * iqr) | (features > q3 + 1.5 * iqr)).sum()
    stats = features.describe().T
    stats["missing_filled"] = missing
    stats["iqr_outlier_count"] = outliers
    stats.index.name = "feature"
    return features, stats.reset_index()


def process(eeg, fs, trials, cfg):
    epochs, kept, audit = preprocess(eeg, fs, trials, cfg)
    columns = [name for name in ["trial_id", "run_id", "trial_in_run", "run_onset_sample",
                                "onset_sample", "label"] if name in kept[0]]
    metadata = pd.DataFrame(kept)[columns]
    metadata["class_name"] = metadata["label"].map(LABELS)
    features, definitions, csp = extract_features(epochs, metadata["label"].to_numpy(), fs, cfg)
    features, stats = check_features(features)
    table = pd.concat([metadata, features], axis=1)
    return table, definitions, stats, audit, csp


def export_result(result, source, output, cfg):
    table, definitions, stats, audit, csp = result
    counts = definitions.groupby("family").size().to_dict()
    notes = [
        ("input", str(source)),
        ("sampling", "250 Hz；前 22 通道 EEG；排除 3 通道 EOG；MAT 的 X 已为 µV，GDF 经 MNE 从 V 转为 µV"),
        ("time_reference", "trial 开始为 0 s（GDF 事件 768 / MAT trial）；提示在 2 s；不以提示为零点"),
        ("MAT_indexing", "MAT trial 从 1 起始，读取时减 1；跳过无 trial 的校准 run；"
         "run_id 为原始 data 中的位置；onset_sample 为拼接运动想象 run 后的零起始索引；"
         "run_onset_sample 为原始 run 内的零起始索引"),
        ("preprocessing", f"0–7 s trial 上四阶 Butterworth {cfg.low}–{cfg.high} Hz SOS 双向滤波；"
         f"减去 [0,2) s 基线均值；取 [{cfg.start},{cfg.stop}) s"),
        ("artifact", f"丢弃官方伪迹（GDF 1023 / MAT artifacts=1）、非有限数据、不完整 trial、平直通道及窗口峰峰值 > "
         f"{cfg.max_ptp_uv} µV 的 trial；阈值是可配置经验值；不做 ICA"),
        ("frequency", "Welch：Hann 窗 1 s、50% 重叠、去常数趋势；PSD 梯形积分"),
        ("feature_counts", str(counts)),
        ("matrix_shape", f"{len(table)} trial × {len(definitions)} 特征；另有 {len(table.columns) - len(definitions)} 列 trial/标签元数据"),
        ("retention", f"输入 {len(audit)}；保留 {len(table)}；丢弃 {len(audit) - len(table)}"),
        ("class_counts", str(table["class_name"].value_counts().to_dict())),
        ("label_mapping", "1=左手；2=右手；3=双脚；4=舌头"),
        ("feature_definitions", "时域每通道均值、总体方差、总体标准差、峰峰值，共 22×4=88；"
         "频域每通道 μ/β 绝对功率及相对功率，共 22×4=88；"
         f"空间为 {cfg.csp_components} 个 CSP 投影方差，总计 {len(definitions)} 个特征"),
        ("sources", "本地输入文件见 input；数据定义 https://www.bbci.de/competition/IV/desc_2a.pdf ；"
         "MAT 数据格式 https://bnci-horizon-2020.eu/database/data-sets ；"
         "CSP https://mne.tools/stable/generated/mne.decoding.CSP.html"),
        ("missing_values", f"非有限特征用该列中位数填补；共 {int(stats.missing_filled.sum())} 个；整列缺失则报错"),
        ("outliers", "有限极端值不裁剪；statistics 中报告 1.5×IQR 异常数量，trial_audit 记录丢弃原因"),
        ("CSP", "四分类 CSP，Ledoit-Wolf 协方差正则化；导出投影方差；spatial_filters 保存对应滤波器"),
        ("classification", "当前 CSP/填补参数来自全部保留训练数据，仅供本次描述性特征提取；"
         "后续交叉验证须在每个训练折内重新拟合 CSP 和填补参数，不能直接用此表评估 CSP 分类性能"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    # 避免误覆盖此前的结果；需重新运行时使用不同 --output 名称。
    if output.exists() or output.with_suffix(".txt").exists():
        raise FileExistsError(f"输出已存在，请更换 --output：{output}")
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in [("features", table), ("feature_definitions", definitions),
                            ("statistics", stats), ("trial_audit", audit),
                            ("notes", pd.DataFrame(notes, columns=["item", "description"])),
                            ("spatial_filters", pd.DataFrame(
                                csp.filters_[:cfg.csp_components], columns=CHANNELS))]:
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                letter = column[0].column_letter
                width = max(len(str(cell.value or "")) for cell in column[:50]) + 2
                sheet.column_dimensions[letter].width = min(max(width, 12), 65)
    output.with_suffix(".txt").write_text(
        "脑电特征提取说明\n\n" + "\n\n".join(f"{k}: {v}" for k, v in notes), encoding="utf-8")
    print(f"完成：{output}\n说明：{output.with_suffix('.txt')}\n"
          f"保留 {len(table)}/{len(audit)} 个 trial，提取 {len(definitions)} 个特征。")


def self_test(cfg):
    """用固定随机种子的合成四分类信号验证流程，不输出伪造的作业结果。"""
    fs = 250
    rng = np.random.default_rng(2026)
    trials, blocks = [], []
    t = np.arange(7 * fs) / fs
    for i in range(24):
        label = i % 4 + 1
        block = rng.normal(0, 3, (22, len(t)))
        block[label - 1] += 8 * np.sin(2 * np.pi * (8 + 2 * label) * t)
        blocks.append(block)
        trials.append({"trial_id": i + 1, "onset_sample": i * len(t),
                       "label": label, "official_reject": i == 0})
    blocks[1][0, 800] = np.nan
    blocks[2][0, 800:900] += 1000
    result = process(np.concatenate(blocks, axis=1), fs, trials, cfg)
    table, definitions, stats, audit, _ = result
    assert len(table) == 21 and len(definitions) == 176 + cfg.csp_components
    assert np.isfinite(table[definitions.feature].to_numpy()).all()
    assert set(audit[audit.status == "dropped"].reason) == {
        "official_artifact", "nonfinite_signal", "peak_to_peak_above_threshold"}
    assert (table.filter(regex="_relative$").to_numpy() >= 0).all()
    assert (table.filter(regex="_relative$").to_numpy() <= 1 + 1e-10).all()
    assert len(stats) == len(definitions)
    broken = pd.DataFrame({"test": [1.0, np.nan, np.inf, 3.0]})
    fixed, report = check_features(broken)
    assert fixed.test.tolist() == [1.0, 2.0, 2.0, 3.0]
    assert report.missing_filled.iloc[0] == 2
    print(f"合成数据自检通过：{len(table)} trial，{len(definitions)} 特征；"
          "伪迹剔除、功率比例、缺失值处理通过。")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="一个被试的 Session T MAT 或 GDF 文件")
    parser.add_argument("--output", type=Path, help="输出 .xlsx 路径；默认脚本旁 output/<文件名>_features.xlsx")
    parser.add_argument("--start", type=float, default=3.0, help="相对 trial 开始的窗口起点，秒")
    parser.add_argument("--stop", type=float, default=6.0, help="窗口终点，不含该采样点，秒")
    parser.add_argument("--max-ptp", type=float, default=200.0, help="峰峰值剔除阈值，µV")
    parser.add_argument("--csp-components", type=int, default=6)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    cfg = Config(start=args.start, stop=args.stop, max_ptp_uv=args.max_ptp,
                 csp_components=args.csp_components)
    if not (2 <= cfg.start < cfg.stop <= 6):
        parser.error("窗口须满足 2 <= start < stop <= 6（trial 开始为零点）。")
    if round((cfg.stop - cfg.start) * 250) < 250:
        parser.error("特征窗口至少 1 秒，以支持 1 秒 Welch 窗。")
    if not np.isfinite(cfg.max_ptp_uv) or cfg.max_ptp_uv <= 0 or not 1 <= cfg.csp_components <= 22:
        parser.error("max-ptp 须为正数；csp-components 须为 1–22。")
    base = Path(__file__).resolve().parent
    if not args.self_test and args.input is None:
        candidates = sorted({p.resolve() for folder in (base / "archive", base / "data", base)
                             if folder.exists() for p in folder.iterdir()
                             if p.is_file() and p.name.lower().endswith(("t.mat", "t.gdf"))})
        if not candidates:
            parser.error("未找到 Session T 数据，请用 --input 指定 AxxT.mat 或 AxxT.gdf。")
        args.input = min(candidates, key=lambda p: (p.name.lower(), str(p)))
        print(f"自动选择：{args.input}")
    require_dependencies()
    if args.self_test:
        self_test(cfg)
        return
    source = args.input.expanduser().resolve()
    if not source.is_file():
        parser.error(f"数据文件不存在：{source}")
    output = args.output or base / "output" / f"{source.stem}_features.xlsx"
    if args.output is None:
        # PyCharm 直接重复运行时生成新结果，保留已经检查过的版本。
        version = 2
        while output.exists() or output.with_suffix(".txt").exists():
            output = base / "output" / f"{source.stem}_features_{version}.xlsx"
            version += 1
    if output.suffix.lower() != ".xlsx":
        parser.error("--output 必须以 .xlsx 结尾。")
    eeg, fs, trials = read_data(source)
    export_result(process(eeg, fs, trials, cfg), source, output.expanduser().resolve(), cfg)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileExistsError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)

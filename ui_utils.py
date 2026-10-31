# -*- coding: utf-8 -*-
"""
ui_utils.py —— Streamlit 展示层的公共组件（绘图 / 侧边栏控件 / 缓存 / 导出）

把每个页面都要用到的matplotlib 绘图、侧边栏参数收集、数据集缓存、
CSV 导出集中在这里，避免 5 个页面重复代码（对应 5-4 指导 4.3"美化与工程化"）。
"""
from __future__ import annotations

import io
import os
import traceback
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import data_lab as dl
import pulse_core as pc

# --------------------------------------------------------------------------
# 全局样式：中文字体 + 浅色主题
# --------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["grid.color"] = "#dfe3e8"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.6

PALETTE = ["#c0392b", "#2c6fbb", "#1e8449", "#b9770e", "#7d3c98",
           "#138d75", "#d35400", "#566573"]

PARAM_LABELS = {
    "fc": "截止频率 fc (Hz)", "order": "滤波器阶数",
    "level": "小波分解层数", "thr": "阈值系数",
    "wname": "小波基", "mode": "阈值方式",
    "max_imf": "最大 IMF 数", "drop": "丢弃高频 IMF 数",
    "window": "滑动窗口长度", "poly": "多项式阶数",
    "lo": "下截止频率 (Hz)", "hi": "上截止频率 (Hz)",
    "kernel": "中值核大小", "f0": "陷波频率 (Hz)", "Q": "品质因数 Q",
}


def page_config(title: str = "脉搏波仿真平台"):
    """设置页面参数。用 try 包裹，避免在多页/测试环境下重复调用报错。"""
    try:
        st.set_page_config(page_title=title, page_icon="🫀", layout="wide",
                           initial_sidebar_state="expanded")
    except Exception:
        pass


def unique_str(s: "pd.Series") -> List[str]:
    """安全地取唯一字符串取值。

    pandas 3.x 的 `astype(str)` 会保留缺失值，直接 `.unique()` 可能混进
    float('nan')，再 sorted() 就会抛 "float 与 str 无法比较"。统一走这里。
    """
    return sorted({str(v) for v in s.dropna().tolist()})


def guard(func):
    """页面级异常兜底：出错时给出可读提示与降级建议，而不是白屏。"""
    def wrapper(*a, **kw):
        try:
            return func(*a, **kw)
        except Exception as e:
            # 设 PPG_STRICT=1 时直接抛出，便于自动化测试与排错
            # （否则 guard 会把异常吞掉，测试会误判为"通过"）
            if os.environ.get("PPG_STRICT"):
                raise
            st.error(f"页面运行出错：{type(e).__name__}: {e}")
            with st.expander("查看详细堆栈（排错用）"):
                st.code(traceback.format_exc())
            st.info("建议：检查数据文件路径 / 降低抽样条数 / 缩短仿真时长后重试。")
    return wrapper


# --------------------------------------------------------------------------
# 数据集缓存（@st.cache_data —— 5-4 指导 5.2）
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="正在载入并治理数据集…")
def load_dataset_cached(path: str, strategy: str = "分组中位数填补"):
    raw = dl.load_raw(path)
    clean, log, cost = dl.clean_dataframe(raw, strategy=strategy)
    return raw, clean, log, cost


@st.cache_data(show_spinner="正在生成质量报告…")
def quality_cached(_raw: pd.DataFrame) -> pd.DataFrame:
    return dl.quality_report(_raw)


@st.cache_data(show_spinner="正在解析上传的 CSV…")
def read_upload_cached(data: bytes) -> pd.DataFrame:
    """解析上传的 CSV（按字节缓存，19 MB 文件也不会在每次交互时重读）。

    内部走 dl.read_upload_csv：自动剔除 raw_p11 第 2 行的字段说明行，
    否则所有数值列会被这行文本污染，页面 ② 会报「信号长度不足」。
    """
    import io
    return dl.read_upload_csv(io.BytesIO(data))


# --------------------------------------------------------------------------
# 侧边栏：去噪方法参数控件（由 DENOISE_METHODS 自动派生）
# --------------------------------------------------------------------------
def method_params_widgets(method: str, key_prefix: str = "") -> Dict:
    spec = pc.DENOISE_METHODS[method]
    params: Dict = {}
    if not spec["params"]:
        st.caption("该方法无需调参（作为对比基线）")
        return params
    items = list(spec["params"].items())
    cols = st.columns(2)
    for i, (name, (lo, hi, default, step)) in enumerate(items):
        with cols[i % 2]:
            params[name] = st.slider(
                PARAM_LABELS.get(name, name), float(lo), float(hi),
                float(default), float(step),
                key=f"{key_prefix}{method}_{name}")
    for name, options in (spec.get("extra") or {}).items():
        params[name] = st.selectbox(
            PARAM_LABELS.get(name, name), options,
            key=f"{key_prefix}{method}_{name}")
    return params


def data_source_sidebar() -> Dict:
    """侧边栏上半部分：数据源 + 数据集路径。返回 dict(path, uploaded, source)。"""
    st.sidebar.header("📂 数据源")
    source = st.sidebar.radio(
        "分析对象", ["内置仿真数据集", "上传 CSV", "平台自生成（有真值）"],
        key="src_radio")
    path = None
    uploaded = None
    if source == "内置仿真数据集":
        default_path = dl.resolve_data_path()
        path = st.sidebar.text_input("数据集路径", value=default_path or "",
                                     key="data_path")
        if path and not os.path.exists(path):
            st.sidebar.warning("路径不存在，请检查或在下方上传")
            path = None
    elif source == "上传 CSV":
        uploaded = st.sidebar.file_uploader("选择 CSV 文件", type=["csv"],
                                            key="csv_upload")
    return {"source": source, "path": path, "uploaded": uploaded}


def generation_sidebar() -> Dict:
    """侧边栏：信号生成参数（对应 5-4 指导 1.2 第 1 条）。"""
    st.sidebar.header("🎛 信号生成参数")
    fs = st.sidebar.slider("采样率 fs (Hz)", 100, 1000, 250, 25, key="g_fs")
    hr = st.sidebar.slider("心率 HR (bpm)", 40, 180, 75, 1, key="g_hr")
    sec = st.sidebar.slider("仿真时长 (s)", 3, 30, 10, 1, key="g_sec")
    with st.sidebar.expander("三波形态（幅值 / 相对相位 / 相对脉宽）", expanded=False):
        st.caption("相对相位 φ∈(0,1) 与相对脉宽 w 会随心动周期自适应缩放，"
                   "因此任意心率下三波结构都保持正确。")
        a1 = st.slider("主波幅值 A1", 0.2, 1.5, 1.00, 0.05, key="g_a1")
        a2 = st.slider("潮波幅值 A2", 0.0, 1.0, 0.38, 0.02, key="g_a2")
        a3 = st.slider("重搏波幅值 A3", 0.0, 1.0, 0.24, 0.02, key="g_a3")
        p1 = st.slider("主波相对相位 φ1", 0.05, 0.45, 0.20, 0.01, key="g_p1")
        p2 = st.slider("潮波相对相位 φ2", 0.30, 0.60, 0.42, 0.01, key="g_p2")
        p3 = st.slider("重搏波相对相位 φ3", 0.50, 0.90, 0.62, 0.01, key="g_p3")
        w1 = st.slider("主波相对脉宽 w1", 0.02, 0.12, 0.055, 0.005, key="g_w1")
        w2 = st.slider("潮波相对脉宽 w2", 0.02, 0.12, 0.055, 0.005, key="g_w2")
        w3 = st.slider("重搏波相对脉宽 w3", 0.02, 0.15, 0.070, 0.005, key="g_w3")
    return dict(fs=fs, hr=hr, sec=sec,
                amps=(a1, a2, a3), phis=(p1, p2, p3), ws=(w1, w2, w3))


def noise_sidebar() -> Dict:
    st.sidebar.header("🔊 噪声设置")
    ntype = st.sidebar.selectbox("噪声类型", list(pc.NOISE_TYPES), index=0,
                                 key="n_type")
    snr_in = st.sidebar.slider("加噪 SNR (dB)", -5.0, 30.0, 12.0, 1.0,
                               key="n_snr")
    drift = mains = motion = 0.0
    if ntype in ("基线漂移", "运动伪影", "复合噪声"):
        drift = st.sidebar.slider("基线漂移幅度", 0.0, 0.8, 0.15, 0.05,
                                  key="n_drift")
    if ntype in ("工频干扰", "复合噪声"):
        mains = st.sidebar.slider("工频干扰幅度", 0.0, 0.3, 0.05, 0.01,
                                  key="n_mains")
    if ntype in ("运动伪影", "复合噪声"):
        motion = st.sidebar.slider("运动伪影幅度", 0.0, 0.6, 0.20, 0.05,
                                   key="n_motion")
    seed = st.sidebar.number_input("随机种子（固定 → 结果可复现）", 0, 9999,
                                   42, 1, key="n_seed")
    return dict(noise_type=ntype, snr_in=snr_in, drift_amp=drift,
                mains_amp=mains, motion_amp=motion, seed=int(seed))


def feature_sidebar() -> Dict:
    st.sidebar.header("🔎 特征检测设置")
    tol = st.sidebar.slider("命中容差 tol (ms)", 10.0, 200.0, 50.0, 10.0,
                            key="f_tol")
    prom = st.sidebar.slider("主波突出度系数", 0.05, 0.60, 0.25, 0.05,
                             key="f_prom")
    return dict(tol_ms=tol, prom_sys=prom)


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def _newfig(w=10, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    return fig, ax


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.getvalue()


def plot_overview(t, clean, noisy, denoised, title="去噪前后对比"):
    """干净 / 加噪 / 去噪 三叠波形图（5-4 指导 4.1 要求"去噪前后叠加显示"）。"""
    fig, ax = _newfig(11, 3.6)
    ax.plot(t, clean, lw=1.4, color="#2c6fbb", label="干净真值", zorder=3)
    if noisy is not None:
        ax.plot(t, noisy, lw=0.7, color="#95a5a6", alpha=0.75, label="加噪", zorder=1)
    if denoised is not None:
        ax.plot(t, denoised, lw=1.2, color="#c0392b", label="去噪结果", zorder=2)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅值 (a.u.)")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(True)
    return fig


def plot_zoom(t, clean, noisy, denoised, t0=0.0, t1=3.0):
    """局部放大：便于肉眼判断三波是否保住。"""
    fig, ax = _newfig(11, 3.0)
    m = (t >= t0) & (t <= t1)
    ax.plot(t[m], clean[m], lw=1.6, color="#2c6fbb", label="干净真值")
    if noisy is not None:
        ax.plot(t[m], noisy[m], lw=0.8, color="#95a5a6", alpha=0.8, label="加噪")
    if denoised is not None:
        ax.plot(t[m], denoised[m], lw=1.3, color="#c0392b", label="去噪结果")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅值 (a.u.)")
    ax.set_title(f"局部放大 {t0:.1f} – {t1:.1f} s")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True)
    return fig


def plot_single_cycle(t_ms, y, feat_row=None, title="单周期形态标注"):
    """单周期形态图标注主波 / 潮波 / 重搏波 / 重搏切迹（5-4 指导 3.3）。"""
    fig, ax = _newfig(9, 4.2)
    ax.plot(t_ms, y, lw=1.8, color="#2c3e50", zorder=1)
    ax.fill_between(t_ms, np.min(y), y, color="#2c6fbb", alpha=0.10)
    marks = [("主波", "main", "#c0392b", "^"),
             ("潮波", "tidal", "#b9770e", "s"),
             ("重搏波", "dicrotic", "#1e8449", "o"),
             ("重搏切迹", "notch", "#7d3c98", "v")]
    used = set()
    for label, key, color, marker in marks:
        v = feat_row.get(key + "_ms", np.nan) if feat_row is not None else np.nan
        if v is None or not np.isfinite(v):
            continue
        i = int(np.argmin(np.abs(t_ms - v)))
        ax.scatter([t_ms[i]], [y[i]], s=90, c=color, marker=marker,
                   zorder=5, edgecolors="white", linewidths=1.2)
        ax.annotate(f"{label}\n{v:.0f} ms", (t_ms[i], y[i]),
                    textcoords="offset points", xytext=(0, 16),
                    ha="center", fontsize=9, color=color, fontweight="bold")
        used.add(label)
    ax.set_xlabel("周期内时间 (ms)")
    ax.set_ylabel("幅值 (a.u.)")
    ax.set_title(title + ("（已标注：" + "、".join(sorted(used)) + "）"
                          if used else "（未检出特征点）"))
    ax.grid(True)
    return fig


def plot_spectrum(t, clean, noisy, denoised, fs, fmax=None):
    """频谱对比：验证低通 / 陷波的截止频率是否真的生效。"""
    fig, ax = _newfig(11, 3.0)
    fmax = fmax or min(60.0, fs / 2)
    for sig, name, color, lw in ((clean, "干净", "#2c6fbb", 1.5),
                                 (noisy, "加噪", "#95a5a6", 0.9),
                                 (denoised, "去噪", "#c0392b", 1.2)):
        if sig is None:
            continue
        f, a = pc.spectrum(sig, fs)
        if len(f) == 0:
            continue
        m = f <= fmax
        ax.semilogy(f[m], a[m] + 1e-9, lw=lw, color=color, label=name)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅度 (对数)")
    ax.set_title("幅度谱对比")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, which="both", alpha=0.4)
    return fig


def plot_residual(t, clean, denoised):
    """残差图：去噪结果 − 干净真值，越接近 0 说明保真度越高。"""
    fig, ax = _newfig(11, 2.6)
    r = np.asarray(denoised, float)[:len(clean)] - np.asarray(clean, float)
    ax.plot(t, r, lw=0.8, color="#7d3c98")
    ax.axhline(0, color="#2c3e50", lw=0.8)
    ax.fill_between(t, 0, r, color="#7d3c98", alpha=0.18)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("残差 (a.u.)")
    ax.set_title(f"残差（去噪 − 真值），RMSE = {pc.rmse(clean, denoised):.4f}")
    ax.grid(True)
    return fig


def plot_group_bar(df: pd.DataFrame, x: str, y: str, title: str,
                   horizontal=False, topn: int = 20):
    fig, ax = _newfig(9, 4.0)
    d = df.head(topn)
    if horizontal:
        ax.barh(d[x].astype(str), d[y], color="#2c6fbb")
        ax.invert_yaxis()
        ax.set_xlabel(y)
    else:
        ax.bar(d[x].astype(str), d[y], color="#2c6fbb")
        ax.set_ylabel(y)
        ax.tick_params(axis="x", rotation=35)
    ax.set_title(title)
    ax.grid(axis="y" if not horizontal else "x")
    return fig


def plot_box_by_group(df: pd.DataFrame, group: str, value: str, title: str,
                      max_groups: int = 10):
    fig, ax = _newfig(10, 4.2)
    sub = df[[group, value]].dropna()
    groups = sub[group].astype(str).value_counts().head(max_groups).index.tolist()
    data, labels = [], []
    for g in groups:
        v = pd.to_numeric(sub.loc[sub[group].astype(str) == g, value],
                          errors="coerce").dropna()
        if len(v) >= 3:
            data.append(v.values)
            labels.append(g)
    if not data:
        ax.text(0.5, 0.5, "有效样本不足", ha="center", va="center",
                transform=ax.transAxes)
        return fig
    try:  # matplotlib ≥3.9 用 tick_labels，旧版本用 labels
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        showmeans=True)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(PALETTE[i % len(PALETTE)])
        b.set_alpha(0.55)
    ax.set_ylabel(value)
    ax.set_title(title)
    ax.grid(axis="y")
    ax.tick_params(axis="x", rotation=25)
    return fig


def plot_lines(df: pd.DataFrame, x: str, y: str, hue: str, title: str,
               ylabel: str = ""):
    fig, ax = _newfig(10, 4.2)
    for i, (name, g) in enumerate(df.groupby(hue, sort=False)):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker="o", ms=4, lw=1.6,
                color=PALETTE[i % len(PALETTE)], label=str(name))
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True)
    return fig


def plot_heatmap(pivot: pd.DataFrame, title: str, cmap="RdYlGn", fmt="{:.2f}"):
    fig, ax = _newfig(max(6, 1.1 * len(pivot.columns) + 3),
                      max(3.2, 0.55 * len(pivot.index) + 2))
    v = pivot.to_numpy(dtype=float)
    im = ax.imshow(v, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_title(title)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            val = v[i, j]
            if np.isfinite(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def plot_wave_library(waves: List[np.ndarray], labels: List[str],
                      title="波形库抽样（归一化叠加）"):
    fig, ax = _newfig(11, 3.8)
    n_max = max((len(w) for w in waves), default=1)
    for i, w in enumerate(waves):
        if len(w) < 4:
            continue
        x = np.linspace(0, 100, len(w))
        y = (w - np.min(w)) / max(np.max(w) - np.min(w), 1e-9)
        ax.plot(x, y, lw=1.2, color=PALETTE[i % len(PALETTE)],
                label=labels[i] if i < len(labels) else None, alpha=0.85)
    ax.set_xlabel("周期内位置 (%)")
    ax.set_ylabel("归一化幅值")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True)
    return fig


def plot_scatter(mor: pd.DataFrame, xcol: str, ycol: str, title: str):
    fig, ax = _newfig(6.5, 5.0)
    d = mor[[xcol, ycol]].apply(pd.to_numeric, errors="coerce").dropna()
    if d.empty:
        ax.text(0.5, 0.5, "无可比样本", ha="center", va="center",
                transform=ax.transAxes)
        return fig
    ax.scatter(d[xcol], d[ycol], s=14, alpha=0.55, color="#2c6fbb",
               edgecolors="none")
    lo = float(min(d[xcol].min(), d[ycol].min()))
    hi = float(max(d[xcol].max(), d[ycol].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True)
    return fig


def plot_radar(summary: pd.DataFrame, value_col: str, title: str):
    """去噪算法 ΔSNR 雷达图（对应指导书任务 6 的可视化要求）。"""
    fig = plt.figure(figsize=(6.2, 5.4), dpi=110)
    ax = fig.add_subplot(111, polar=True)
    d = summary.dropna(subset=[value_col])
    if len(d) < 3:
        ax.text(0.5, 0.5, "方法数不足", ha="center", va="center",
                transform=ax.transAxes)
        return fig
    labels = d["方法"].astype(str).tolist()
    vals = d[value_col].to_numpy(dtype=float)
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals = list(vals) + [vals[0]]
    ang += ang[:1]
    ax.plot(ang, vals, lw=1.8, color="#c0392b")
    ax.fill(ang, vals, color="#c0392b", alpha=0.22)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, pad=18)
    return fig


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def download_df(df: pd.DataFrame, filename: str, label: str = "下载 CSV"):
    st.download_button(label, data=to_csv_bytes(df),
                       file_name=filename, mime="text/csv",
                       key=f"dl_{filename}")


def metric_row(items: List[tuple], ncol: int = 4):
    cols = st.columns(ncol)
    for i, (label, value, help_text) in enumerate(items):
        with cols[i % ncol]:
            st.metric(label, value, help=help_text or None)

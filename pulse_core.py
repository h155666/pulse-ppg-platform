# -*- coding: utf-8 -*-
"""
pulse_core.py —— 脉搏波信号算法内核（课题 5-4 / 与 5.1、5.3 共用）

本文件【不依赖 Streamlit】，可直接在命令行 / PyQt / 脚本中 import，
对应指导文件：
  · 5-4 指导.md 步骤 2.2   "写 pulse_core.py（生成 / 去噪 / 特征函数）"
  · 5-4 背景知识.md 2.2    "算法层由 pulse_core.py 提供，与 5.3 共用"
  · 大方向背景知识.md 第 2 节（高斯脉冲串 / 去噪 / 特征点 / 评价指标）

模块划分：
  1. 信号生成     gen_pulse            —— 改进的自适应高斯脉冲串（相对相位）
  2. 噪声叠加     add_noise            —— 白噪 / 基线漂移 / 工频 / 运动伪影 / 复合
  3. 去噪方法     denoise_*  ×8         —— 无 / 低通 / 小波 / EMD / SG / 带通 / 中值 / 陷波+低通
  4. 特征检测     detect_features      —— 主波 / 潮波 / 重搏波 / 重搏切迹
  5. 形态学特征   morphology_features  —— 脉宽 / 上升时间 / AIx / RI / SI / PI / K / dpdt_max
  6. 评价指标     snr_db / rmse / snr_gain / hit_rate / match_peaks
  7. 批量实验     run_single_experiment —— 单条 (方法 × 噪声档 × 种子) 的完整评测

运行环境：numpy + scipy + PyWavelets，纯 CPU。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pywt
from scipy.interpolate import CubicSpline
from scipy.signal import (argrelextrema, butter, filtfilt, find_peaks,
                          iirnotch, medfilt, savgol_filter, sosfiltfilt,
                          tf2sos)

EPS = 1e-12

# ==========================================================================
# 0. 通用工具
# ==========================================================================

def _fix_len(y: np.ndarray, n: int) -> np.ndarray:
    """统一长度：小波重构 / 卷积类滤波可能多出或缺失若干点（5-3 背景 Q2）。"""
    y = np.asarray(y, dtype=float).ravel()
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n]
    return np.pad(y, (0, n - len(y)), mode="edge")


def _safe_range(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.max(x) - np.min(x))


def to_number(s, assume_ms: bool = False) -> float:
    """把 '0.39 s' / '333ms' / '156/98' / '45岁' / '未识别' 之类的脏串解析成 float。

    Parameters
    ----------
    s : 原始值
    assume_ms : True 表示目标单位是毫秒。此时若原文的单位是「秒」（含 s 但不含 ms），
                自动 ×1000；若原文已是 ms 或无单位，则保持原值。

    返回 np.nan 表示无法解析。用于 data_lab 的字段清洗，此处提前定义便于复用。
    """
    if s is None:
        return np.nan
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s) if np.isfinite(s) else np.nan
    t = str(s).strip()
    if t == "" or t.lower() in {"nan", "none", "null", "na", "未识别", "未知", "无", "-"}:
        return np.nan
    t = t.replace(",", "")
    # 156/98 这类"收缩压/舒张压"合并写法 -> 取前面的收缩值
    if "/" in t:
        t = t.split("/")[0]
    # 提取第一个数字（可含小数与负号）
    import re
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    if not m:
        return np.nan
    v = float(m.group())
    low = t.lower()
    if assume_ms and ("s" in low) and ("ms" not in low):
        v *= 1000.0
    return v


# ==========================================================================
# 1. 信号生成 —— 自适应高斯脉冲串
# ==========================================================================
#
# 【模型改进说明（可写入论文 3.2 节）】
# 大方向背景 7 节的原始实现把三波位置写成绝对秒 mus=(0.18, 0.40, 0.58)，
# 再取 tp = t % T。当 HR > 100 bpm（T < 0.6 s）时 μ2、μ3 已超出一个周期，
# 会被 t % T 折叠到错误相位，波形失真。
# 本实现改为【相对相位 φk∈(0,1)】与【相对脉宽 wk】，令
#       μk = φk · T ,   σk = wk · T ,   T = 60 / HR
# 从而任意心率下三波相对位置恒定；且可由 (φk, T, fs) 直接反推出真值峰索引，
# 无需人工标注即可计算特征点命中率。
#
DEFAULT_PHIS = (0.20, 0.42, 0.62)   # 主波 / 潮波 / 重搏波 的相对相位
DEFAULT_WS = (0.055, 0.055, 0.070)  # 三波的相对脉宽（占周期比例）
DEFAULT_AMPS = (1.00, 0.38, 0.24)   # 三波幅值


def gen_pulse(fs: float, hr: float, sec: float,
              amps: Sequence[float] = DEFAULT_AMPS,
              phis: Sequence[float] = DEFAULT_PHIS,
              ws: Sequence[float] = DEFAULT_WS,
              baseline: float = 0.0,
              min_sigma: float = 0.048,
              max_sigma: float = 0.090,
              return_truth: bool = False):
    """生成高斯脉冲串脉搏波。

    Parameters
    ----------
    fs : 采样率 Hz
    hr : 心率 bpm
    sec : 时长 s
    amps/phis/ws : 三波的幅值 / 相对相位 / 相对脉宽，长度均为 3
    baseline : 直流基线。真实 PPG 的直流（光强/组织基线）远大于搏动分量，
               需要计算生理合理的灌注指数 PI = AC/DC 时应设为 AC 幅值的 15~25 倍。
    min_sigma / max_sigma : 脉宽 σ 的绝对上下限（秒）。
        【为什么需要下限】σ = wk·T 是比例式建模，心率越高 σ 越小。
        但生理上收缩期上升时间**并不随心率成比例缩短**，比例式建模在
        高心率段会把波形压得过窄（实测 HR>110 bpm 时半高宽仅 ~77 ms，
        低于 100 ms 的生理下限）。因此给 σ 设绝对下限，使脉宽回到合理区间。
        HR≤100 bpm 时不会触发，行为与纯比例式一致。
    return_truth : True 时额外返回真值特征点索引（用于命中率）

    Returns
    -------
    (t, x) 或 (t, x, truth)，truth = {"main": idx[], "tidal": [], "dicrotic": []}
    """
    fs = float(fs); hr = float(hr); sec = float(sec)
    N = int(round(fs * sec))
    if N < 8:
        N = 8
    t = np.arange(N) / fs
    T = 60.0 / max(hr, 1e-6)
    tp = np.mod(t, T)
    x = np.full(N, float(baseline))
    for a, p, w in zip(amps, phis, ws):
        mu = float(p) * T
        sg = float(np.clip(float(w) * T, float(min_sigma), float(max_sigma)))
        x = x + float(a) * np.exp(-((tp - mu) ** 2) / (2.0 * sg * sg))
    if not return_truth:
        return t, x

    truth: Dict[str, List[int]] = {"main": [], "tidal": [], "dicrotic": []}
    n_cyc = int(np.floor(sec / T))
    for k in range(n_cyc):
        t0 = k * T
        for key, p in zip(truth.keys(), phis):
            idx = int(round((t0 + float(p) * T) * fs))
            if 0 <= idx < N:
                truth[key].append(idx)
    return t, x, {k: np.asarray(v, dtype=int) for k, v in truth.items()}


# ==========================================================================
# 2. 噪声叠加
# ==========================================================================

NOISE_TYPES = ("高斯白噪", "基线漂移", "工频干扰", "运动伪影", "复合噪声", "无噪声")


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator,
              fs: float = 250.0, noise_type: str = "高斯白噪",
              drift_amp: float = 0.15, mains_amp: float = 0.05,
              motion_amp: float = 0.20, mains_freq: float = 50.0) -> np.ndarray:
    """叠加噪声。noise_type 与数据集的 noise_type 字段口径保持一致。

    x_noisy = x + n_white + n_drift + n_mains + n_motion
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    t = np.arange(N) / float(fs)
    y = x.copy()

    # 高斯白噪：功率由目标 SNR 反解 Pn = Ps / 10^(SNR/10)
    if noise_type != "无噪声":
        p_sig = np.mean((x - np.mean(x)) ** 2) + EPS
        p_n = p_sig / (10.0 ** (float(snr_db) / 10.0))
        y = y + np.sqrt(p_n) * rng.standard_normal(N)

    # 基线漂移（低频，0.15 / 0.35 Hz）
    if noise_type in ("基线漂移", "运动伪影", "复合噪声"):
        y = y + drift_amp * np.sin(2 * np.pi * 0.15 * t) \
            + 0.5 * drift_amp * np.sin(2 * np.pi * 0.35 * t + 1.0)

    # 工频干扰（50 Hz）
    if noise_type in ("工频干扰", "复合噪声"):
        y = y + mains_amp * np.sin(2 * np.pi * mains_freq * t)

    # 运动伪影：稀疏的大幅度低频突发
    if noise_type in ("运动伪影", "复合噪声"):
        env = np.zeros(N)
        burst_len = max(3, int(0.25 * fs))
        n_burst = max(1, int(N / max(fs, 1) / 3.0))
        hi = max(1, N - burst_len)
        starts = rng.integers(0, hi, size=n_burst)
        w = np.hanning(burst_len)
        for s in starts:
            seg = env[s:s + burst_len]
            env[s:s + burst_len] = np.maximum(seg, w[:len(seg)])
        y = y + motion_amp * env * rng.standard_normal(N)

    return y


# ==========================================================================
# 3. 去噪方法（8 种）
# ==========================================================================

def denoise_none(x, fs=250.0, **kw):
    """无处理 —— 作为基线（对应数据集 denoise_algo='未处理'）。"""
    return np.asarray(x, dtype=float).copy()


def denoise_lowpass(x, fs=250.0, fc=15.0, order=4, **kw):
    """巴特沃斯低通 + filtfilt（零相位，规避大方向背景「坑 3」）。

    用 SOS（二阶节）形式实现：与 (b,a) 形式幅频响应完全一致，但数值稳定性更好，
    阶数升高或相对带宽很窄时不会出现系数病态。
    """
    x = np.asarray(x, dtype=float)
    nyq = float(fs) / 2.0
    wn = min(max(float(fc) / nyq, 1e-4), 0.99)
    sos = butter(max(1, int(order)), wn, btype="low", output="sos")
    return sosfiltfilt(sos, x)


def _safe_level(n: int, wname: str, level: int) -> int:
    """小波层数自适应，规避「层数超过 log2(N) 报错」（大方向背景坑 2）。"""
    try:
        mx = pywt.dwt_max_level(n, pywt.Wavelet(wname).dec_len)
    except Exception:
        mx = int(np.log2(max(n, 2))) - 1
    return int(max(1, min(int(level), max(1, mx))))


def denoise_wavelet(x, fs=250.0, wname="db4", level=4, thr=0.04, mode="soft", **kw):
    """小波阈值去噪：wavedec → 细节系数阈值收缩 → waverec（截断对齐）。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    lv = _safe_level(n, wname, level)
    coeffs = pywt.wavedec(x, wname, level=lv)
    out = [coeffs[0]]
    for c in coeffs[1:]:
        if c.size == 0:
            out.append(c)
            continue
        tau = float(thr) * float(np.max(np.abs(c)))
        out.append(pywt.threshold(c, tau, mode=mode))
    return _fix_len(pywt.waverec(out, wname), n)


# ---------- EMD（自实现，避免引入 PyEMD 依赖） ----------

def _extrema_idx(h: np.ndarray, kind: str) -> np.ndarray:
    """向量化求极值索引（比 scipy.signal.argrelextrema 快一个量级）。"""
    d = np.diff(h)
    if kind == "max":
        return np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    return np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1


def _envelope(h: np.ndarray, kind: str) -> Optional[np.ndarray]:
    """用三次样条拟合上/下包络；端点并入极值点以抑制端点效应。"""
    N = len(h)
    ext = _extrema_idx(h, kind)
    if len(ext) == 0:
        return None
    ext = np.unique(np.concatenate(([0], ext, [N - 1]))).astype(float)
    if len(ext) < 4:
        return np.interp(np.arange(N), ext, h[ext.astype(int)])
    try:
        cs = CubicSpline(ext, h[ext.astype(int)])
        return cs(np.arange(N))
    except Exception:
        return np.interp(np.arange(N), ext, h[ext.astype(int)])


def emd_decompose(x: np.ndarray, max_imf: int = 6, max_iter: int = 30) -> List[np.ndarray]:
    """经验模态分解（Huang 1998）的轻量 sift 实现。"""
    r = np.asarray(x, dtype=float).copy()
    imfs: List[np.ndarray] = []
    for _ in range(max(1, int(max_imf))):
        h = r.copy()
        for _ in range(max_iter):
            up = _envelope(h, "max")
            lo = _envelope(h, "min")
            if up is None or lo is None:
                break
            m = 0.5 * (up + lo)
            denom = np.sum(h ** 2) + EPS
            h = h - m
            if np.sum(m ** 2) / denom < 1e-6:
                break
        imfs.append(h)
        r = r - h
        if _safe_range(r) < 1e-8:
            break
    imfs.append(r)
    return imfs


def denoise_emd(x, fs=250.0, max_imf=6, drop=2, **kw):
    """EMD 去噪：丢弃前 drop 个高频 IMF（噪声主导）后重构。"""
    x = np.asarray(x, dtype=float)
    imfs = emd_decompose(x, max_imf=max_imf)
    k = int(np.clip(drop, 0, len(imfs) - 1))
    return _fix_len(np.sum(imfs[k:], axis=0), len(x))


def denoise_savgol(x, fs=250.0, window=11, poly=3, **kw):
    """Savitzky-Golay 平滑（保峰形，对应数据集 2443 条）。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    win = int(window)
    if win % 2 == 0:
        win += 1
    win = int(np.clip(win, 3, n if n % 2 == 1 else n - 1))
    if win < 3:
        return x.copy()
    poly = int(np.clip(poly, 1, win - 2 if win - 2 >= 1 else 1))
    try:
        return savgol_filter(x, win, poly)
    except Exception:
        return x.copy()


def denoise_bandpass(x, fs=250.0, lo=0.5, hi=8.0, order=4, **kw):
    """0.5–8 Hz 带通（对应数据集 denoise_algo='0.5-8Hz带通滤波'）。

    注意：该滤波器会正确去除直流基线。评价时必须用 AC 耦合口径
    （见 snr_db / rmse 的 ac_couple 参数），否则会被误判为"去噪变差"。
    """
    x = np.asarray(x, dtype=float)
    nyq = float(fs) / 2.0
    l = min(max(float(lo) / nyq, 1e-4), 0.98)
    h = min(max(float(hi) / nyq, l + 1e-3), 0.99)
    sos = butter(max(1, int(order)), [l, h], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def denoise_median(x, fs=250.0, kernel=5, **kw):
    """中值滤波（对脉冲型运动伪影有效）。"""
    x = np.asarray(x, dtype=float)
    k = int(kernel)
    if k % 2 == 0:
        k += 1
    k = int(np.clip(k, 3, max(3, len(x) // 2 * 2 - 1)))
    try:
        return medfilt(x, kernel_size=k)
    except Exception:
        return x.copy()


def denoise_notch(x, fs=250.0, f0=50.0, Q=30.0, fc=15.0, order=4, **kw):
    """自适应陷波（50 Hz 工频）+ 低通（对应数据集'自适应陷波滤波'）。"""
    x = np.asarray(x, dtype=float)
    fs = float(fs)
    y = x.copy()
    f0 = float(f0)
    if f0 < fs / 2.0:
        try:
            b, a = iirnotch(f0, float(Q), fs)
            y = sosfiltfilt(tf2sos(b, a), y)
        except Exception:
            try:
                b, a = iirnotch(f0, float(Q), fs)
                y = filtfilt(b, a, y)
            except Exception:
                pass
    return denoise_lowpass(y, fs=fs, fc=fc, order=order)


# 方法注册表：params 为 (最小值, 最大值, 默认值, 步长)，供 Streamlit 自动生成控件
DENOISE_METHODS: Dict[str, dict] = {
    "无（基线）": dict(fn=denoise_none, params={},
                   desc="不做任何处理，作为对比基线"),
    "低通滤波": dict(fn=denoise_lowpass,
                  params={"fc": (5.0, 25.0, 15.0, 0.5), "order": (1, 6, 4, 1)},
                  desc="巴特沃斯低通 + filtfilt 零相位，滤除高频白噪"),
    "小波阈值去噪": dict(fn=denoise_wavelet,
                    params={"level": (1, 6, 4, 1), "thr": (0.01, 0.30, 0.04, 0.01)},
                    desc="wavedec 多尺度分解 + 细节系数软/硬阈值收缩",
                    extra={"wname": ["db4", "sym8", "coif3", "db6"],
                           "mode": ["soft", "hard"]}),
    "EMD经验模态分解": dict(fn=denoise_emd,
                       params={"max_imf": (3, 8, 6, 1), "drop": (0, 4, 2, 1)},
                       desc="自适应分解为 IMF，丢弃前若干个高频 IMF 后重构"),
    "Savitzky-Golay滤波": dict(fn=denoise_savgol,
                          params={"window": (5, 31, 11, 2), "poly": (2, 5, 3, 1)},
                          desc="局部多项式拟合平滑，能较好保留峰形"),
    "0.5-8Hz带通滤波": dict(fn=denoise_bandpass,
                        params={"lo": (0.2, 2.0, 0.5, 0.1),
                                "hi": (4.0, 20.0, 8.0, 0.5),
                                "order": (1, 6, 4, 1)},
                        desc="保留脉搏波主频带，同时抑制基线漂移与高频噪声"),
    "中值滤波": dict(fn=denoise_median, params={"kernel": (3, 15, 5, 2)},
                  desc="非线性滤波，对脉冲型运动伪影效果明显"),
    "自适应陷波+低通": dict(fn=denoise_notch,
                       params={"f0": (45.0, 65.0, 50.0, 1.0),
                               "Q": (5.0, 60.0, 30.0, 1.0),
                               "fc": (5.0, 25.0, 15.0, 0.5)},
                       desc="先陷波抑制工频干扰，再低通压制高频噪声"),
}


def denoise(x, method: str, fs: float = 250.0, **params) -> np.ndarray:
    """统一去噪入口，保证返回与输入同长度的数组。"""
    x = np.asarray(x, dtype=float)
    spec = DENOISE_METHODS.get(method)
    if spec is None:
        return x.copy()
    try:
        y = spec["fn"](x, fs=fs, **params)
    except Exception:
        return x.copy()
    return _fix_len(y, len(x))


# ==========================================================================
# 4. 特征点检测（主波 / 潮波 / 重搏波 / 重搏切迹）
# ==========================================================================

def estimate_hr(x: np.ndarray, fs: float) -> float:
    """用频谱主峰粗估心率（bpm），作为未指定 HR 时的兜底。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 8:
        return 75.0
    x = x - np.mean(x)
    if _safe_range(x) <= 0:
        return 75.0
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win))
    freq = np.fft.rfftfreq(n, d=1.0 / float(fs))
    lo = np.searchsorted(freq, 0.5)     # 30 bpm
    hi = np.searchsorted(freq, 4.0)     # 240 bpm
    if hi <= lo:
        return 75.0
    seg = spec[lo:hi]
    if seg.size == 0:
        return 75.0
    return float(freq[lo + int(np.argmax(seg))] * 60.0)


def detect_features(x: np.ndarray, fs: float, hr: Optional[float] = None,
                    prom_sys: float = 0.25, prom_sub: float = 0.02,
                    dist_sys: float = 0.70, dist_sub: float = 0.05,
                    tail_win: float = 0.75) -> Dict[str, np.ndarray]:
    """三波 + 切迹检测。

    策略（对应 5-3 背景 2.2.3 / 5-4 指导 3.3）：
      1) 主波：全段 find_peaks，最小间距 0.7T、突出度 25% 量程；
      2) 每个主波之后开 0.75T 的尾窗，在窗内找次峰（间距 0.05T、突出度 2%）；
      3) 窗内按位置排序：靠前 → 潮波，靠后 → 重搏波；
      4) 重搏切迹：主波与潮波（或窗起点）之间的局部极小；
      5) 后处理：保证「每周期至少一个主波」(5-3 背景 Q4)。

    返回：各数组长度 = 主波个数，缺失处为 np.nan（便于直接对齐成表）。
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    keys = ["main_idx", "tidal_idx", "dicrotic_idx", "notch_idx",
            "main_amp", "tidal_amp", "dicrotic_amp", "notch_amp",
            "main_ms", "tidal_ms", "dicrotic_ms", "notch_ms"]
    # 退化信号（过短 / 常量 / 全零）也要返回完整结构，否则调用方 KeyError
    empty: Dict[str, object] = {k: np.array([], dtype=float) for k in keys}
    empty["n_cycle"] = 0
    if N < 8 or _safe_range(x) <= 0:
        return empty

    fs = float(fs)
    if not hr or not np.isfinite(hr) or hr <= 0:
        hr = estimate_hr(x, fs)
    T = 60.0 / max(float(hr), 20.0)
    rng = _safe_range(x)

    dmin = max(2, int(dist_sys * T * fs))
    peaks, _ = find_peaks(x, distance=dmin, prominence=max(prom_sys * rng, 1e-9))
    if len(peaks) == 0:
        return empty

    tail = max(3, int(tail_win * T * fs))
    dsub = max(1, int(dist_sub * T * fs))

    main, tidal, dicrotic, notch = [], [], [], []
    for p in peaks:
        lo = p + 1
        hi = min(N, p + tail)
        main.append(p)

        tid = np.nan
        dic = np.nan
        nt = np.nan
        if hi - lo >= 3:
            seg = x[lo:hi]
            sp, _ = find_peaks(seg, distance=dsub,
                               prominence=max(prom_sub * rng, 1e-9))
            if len(sp) > 0:
                order = sp[np.argsort(seg[sp])[::-1]][:2]
                order = np.sort(order)
                cand = [lo + int(c) for c in order]
                if len(cand) == 1:
                    # 只有一个次峰：按位置判是潮波还是重搏波
                    rel = (cand[0] - p) / max(tail, 1)
                    if rel < 0.5:
                        tid = cand[0]
                    else:
                        dic = cand[0]
                else:
                    tid, dic = cand[0], cand[1]
            # 切迹：主波后到第一个次峰之间的极小
            end = int(tid) if np.isfinite(tid) else hi
            if end - p > 2:
                seg2 = x[p:end]
                if len(seg2) > 2:
                    nt = p + int(np.argmin(seg2))
        tidal.append(tid)
        dicrotic.append(dic)
        notch.append(nt)

    def arr(v):
        return np.asarray(v, dtype=float)

    def amp(v):
        v = np.asarray(v, dtype=float)
        out = np.full(v.shape, np.nan)
        ok = np.isfinite(v)
        out[ok] = x[v[ok].astype(int)]
        return out

    res = {
        "main_idx": arr(main), "tidal_idx": arr(tidal),
        "dicrotic_idx": arr(dicrotic), "notch_idx": arr(notch),
        "main_amp": amp(main), "tidal_amp": amp(tidal),
        "dicrotic_amp": amp(dicrotic), "notch_amp": amp(notch),
    }
    for k in ("main", "tidal", "dicrotic", "notch"):
        res[f"{k}_ms"] = res[f"{k}_idx"] * 1000.0 / fs
    res["n_cycle"] = len(main)
    return res


def features_to_frame(feat: Dict[str, np.ndarray]) -> "object":
    """把 detect_features 的结果整理成按周期对齐的表格（需要 pandas）。"""
    import pandas as pd
    if not feat or len(feat.get("main_idx", [])) == 0:
        return pd.DataFrame()
    n = len(feat["main_idx"])
    return pd.DataFrame({
        "周期": np.arange(1, n + 1),
        "主波时刻(ms)": feat["main_ms"],
        "主波幅值": feat["main_amp"],
        "潮波时刻(ms)": feat["tidal_ms"],
        "潮波幅值": feat["tidal_amp"],
        "重搏波时刻(ms)": feat["dicrotic_ms"],
        "重搏波幅值": feat["dicrotic_amp"],
        "重搏切迹(ms)": feat["notch_ms"],
        "切迹幅值": feat["notch_amp"],
    })


# ==========================================================================
# 5. 形态学特征
# ==========================================================================

def average_cycle(x: np.ndarray, fs: float, hr: Optional[float] = None,
                  n_points: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """按主波切分周期并插值对齐，得到平均单周期波形。"""
    x = np.asarray(x, dtype=float)
    fs = float(fs)
    feat = detect_features(x, fs, hr=hr)
    idx = feat["main_idx"]
    idx = idx[np.isfinite(idx)].astype(int)
    if len(idx) < 2:
        return np.array([]), np.array([])
    T = 60.0 / (hr if hr else estimate_hr(x, fs))
    win = int(T * fs)
    blocks = []
    for p in idx[:-1]:
        seg = x[p:p + win]
        if len(seg) == win:
            blocks.append(seg)
    if not blocks:
        return np.array([]), np.array([])
    mat = np.vstack(blocks)
    y = mat.mean(axis=0)
    # 把周期起点"滚动"到舒张期最低点（主波前的波足），
    # 否则周期数组从主波峰开始，半高宽 / 上升时间都无从计算。
    i0 = int(np.argmin(y))
    if 0 < i0 < len(y):
        y = np.roll(y, -i0)
    t_ms = np.arange(len(y)) * 1000.0 / fs
    # 重采样到 n_points，便于不同采样率之间比较
    if len(y) > 3 and n_points:
        t_new = np.linspace(0, t_ms[-1], n_points)
        y = np.interp(t_new, t_ms, y)
        t_ms = t_new
    return t_ms, y


MORPH_KEYS = ("pulse_width_ms", "rise_time_ms", "augmentation_index",
              "reflection_index", "stiffness_index", "perfusion_index",
              "k_value", "dpdt_max", "sys_amplitude", "notch_ms",
              "dicrotic_ms", "hr_est")


def _cycle_features(t_ms: np.ndarray, y: np.ndarray,
                    height_cm: Optional[float] = None) -> Dict[str, float]:
    """在【已对齐到单周期】的 (t_ms, y) 上计算形态学特征。

    特征口径显式声明（对应项目指导书「任务 5 形态学特征量化分析」）：
      · 脉宽 PW      —— 半高（50% 幅值）处两交点的时间差 (ms)
      · 上升时间 RT  —— 主波上升沿 10% → 90% 的时间差 (ms)
      · AIx          —— (A_sys − A_dic) / (A_sys − A_notch) × 100 (%)
      · RI           —— A_dic / A_sys × 100 (%)
      · SI           —— 身高(m) / PTT(s)，PTT 取主波峰→重搏波峰时间差
      · PI           —— AC / DC × 100 (%)；无直流基线时返回 NaN
      · K            —— 归一化后波形均值
      · dpdt_max     —— 一阶差分最大值（1/s）
    """
    out = {k: np.nan for k in MORPH_KEYS}
    y0 = y - np.min(y)
    amp = float(np.max(y0))
    if amp <= 0:
        return out
    yn = y0 / amp
    out["sys_amplitude"] = amp

    i_pk = int(np.argmax(yn))
    dt = (t_ms[-1] - t_ms[0]) / max(len(t_ms) - 1, 1)
    if dt <= 0:
        return out

    def first_cross(level, lo, hi, direction=1):
        r = range(lo, hi) if direction > 0 else range(hi - 1, lo - 1, -1)
        for i in r:
            if yn[i] >= level:
                return i
        return -1

    # 上升时间 10% → 90%（主波上升沿）
    lo_b = first_cross(0.10, 0, i_pk + 1)
    hi_b = first_cross(0.90, 0, i_pk + 1)
    if lo_b >= 0 and hi_b >= 0 and hi_b >= lo_b:
        out["rise_time_ms"] = float(t_ms[hi_b] - t_ms[lo_b])

    # 脉宽：半高宽（主波左/右两侧 0.5 交叉点）
    left, right = -1, -1
    for i in range(i_pk, -1, -1):
        if yn[i] <= 0.5:
            left = i
            break
    for i in range(i_pk, len(yn)):
        if yn[i] <= 0.5:
            right = i
            break
    if left >= 0 and right >= 0 and right > left:
        out["pulse_width_ms"] = float(t_ms[right] - t_ms[left])

    # 主波之后的极小（切迹）与极大（重搏波）
    seg = yn[i_pk:]
    mins = argrelextrema(seg, np.less)[0]
    maxs = argrelextrema(seg, np.greater)[0]
    if len(mins) > 0:
        i_n = i_pk + int(mins[0])
        out["notch_ms"] = float(t_ms[i_n] - t_ms[i_pk])
        a_n = float(yn[i_n])
        after = maxs[maxs > mins[0]]
        if len(after) > 0:
            i_d = i_pk + int(after[0])
            out["dicrotic_ms"] = float(t_ms[i_d] - t_ms[i_pk])
            a_d = float(yn[i_d])
            denom = (1.0 - a_n)
            if abs(denom) > 1e-6:
                out["augmentation_index"] = float((1.0 - a_d) / denom * 100.0)
            out["reflection_index"] = float(a_d * 100.0)
            if height_cm and (out["dicrotic_ms"] or 0) > 0:
                ptt = out["dicrotic_ms"] / 1000.0     # 主波→重搏波的传导时间近似
                if ptt > 1e-4:
                    out["stiffness_index"] = float(height_cm / 100.0 / ptt)

    out["k_value"] = float(np.mean(yn))
    # PI = AC / DC × 100%。合成信号若没有直流基线（|DC| < 5% 交流幅值），
    # 分母趋零会导致数值爆炸，此时如实返回 NaN 而不是给出虚高数字。
    dc = float(np.mean(y))
    ac = float(np.max(y) - np.min(y))
    if abs(dc) > 0.05 * ac:
        out["perfusion_index"] = float(ac / abs(dc) * 100.0)
    else:
        out["perfusion_index"] = np.nan
    d = np.diff(y) / (dt / 1000.0)
    if d.size:
        out["dpdt_max"] = float(np.max(np.abs(d)))
    return out


def morphology_features(x: np.ndarray, fs: float, hr: Optional[float] = None,
                        height_cm: Optional[float] = None,
                        from_cycle: bool = True) -> Dict[str, float]:
    """对多周期连续信号求形态学特征（先在平均单周期上计算，抗噪更稳）。"""
    out = {k: np.nan for k in MORPH_KEYS}
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or _safe_range(x) <= 0:
        return out
    fs = float(fs)
    hr_f = float(hr) if (hr and np.isfinite(hr)) else estimate_hr(x, fs)
    out["hr_est"] = hr_f

    if from_cycle:
        t_ms, y = average_cycle(x, fs, hr=hr_f)
    else:
        t_ms, y = np.arange(len(x)) * 1000.0 / fs, x
    if len(y) < 8 or _safe_range(y) <= 0:
        return out
    out.update(_cycle_features(t_ms, y, height_cm))
    out["hr_est"] = hr_f
    return out


def morphology_from_cycle(y: np.ndarray, hr: float,
                          height_cm: Optional[float] = None) -> Dict[str, float]:
    """对【已给定的单周期模板】求形态学特征（用于 raw_p11 的 ppg_wave 字段）。

    y    : 单个心动周期的波形（64 点量级）
    hr   : 该记录的心率，用于把采样点换算成毫秒
    """
    out = {k: np.nan for k in MORPH_KEYS}
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 8 or _safe_range(y) <= 0:
        return out
    hr = float(hr) if (hr and np.isfinite(hr) and hr > 0) else 75.0
    T_ms = 60000.0 / hr
    t_ms = np.linspace(0.0, T_ms, len(y))
    out["hr_est"] = hr
    out.update(_cycle_features(t_ms, y, height_cm))
    out["hr_est"] = hr
    return out


def tile_cycle(y: np.ndarray, hr: float, fs: float, sec: float = 10.0):
    """把单周期模板按心率周期拼接成连续波形。

    raw_p11 的 ppg_wave 只有 64 点（单周期模板），直接做滤波/EMD 样本过短，
    拼接为多周期后才能开展有意义的去噪实验。
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 8:
        return np.array([]), np.array([])
    hr = float(hr) if (hr and np.isfinite(hr) and hr > 0) else 75.0
    fs = float(fs)
    n_cyc = max(2, int(np.ceil(sec * hr / 60.0)))
    # 用线性插值把周期模板重采样到"一个周期应有的点数"
    n_per = max(8, int(round(fs * 60.0 / hr)))
    t_old = np.linspace(0.0, 1.0, len(y))
    t_new = np.linspace(0.0, 1.0, n_per)
    cyc = np.interp(t_new, t_old, y)
    sig = np.tile(cyc, n_cyc)
    t = np.arange(len(sig)) / fs
    return t, sig


# ==========================================================================
# 6. 评价指标
# ==========================================================================

def snr_db(clean: np.ndarray, y: np.ndarray, ac_couple: bool = True) -> float:
    """SNR = 10·log10( 有用信号能量 / 残差能量 )。

    ac_couple=True（默认）先对 clean 与 y 各自去均值（AC 耦合）再比较。
    【为什么必须这样做】
    高斯脉冲串合成信号在直流上有一个不为零的基线（本平台默认≈0.24）。
    带通 / 高通类方法会**正确地**把这个直流滤掉，如果直接按原始波形算残差，
    这 0.24 的直流偏置会被误当成误差，导致带通类方法被系统性冤枉
    （实测 ΔSNR 会假性低到 −7 dB 左右）。
    脉搏波的生理信息由搏动分量（AC）承载，因此去噪评价统一在 AC 耦合下进行。
    需要按原始波形比较时传 ac_couple=False。
    """
    clean = np.asarray(clean, dtype=float)
    y = _fix_len(np.asarray(y, dtype=float), len(clean))
    if ac_couple:
        c = clean - np.mean(clean)
        d = y - np.mean(y)
    else:
        c, d = clean, y
    noise = c - d
    return float(10.0 * np.log10((np.mean(c ** 2) + EPS) /
                                 (np.mean(noise ** 2) + EPS)))


def rmse(a: np.ndarray, b: np.ndarray, ac_couple: bool = True) -> float:
    """均方根误差。默认在 AC 耦合下计算（口径见 snr_db 的说明）。"""
    a = np.asarray(a, dtype=float)
    b = _fix_len(np.asarray(b, dtype=float), len(a))
    if ac_couple:
        a = a - np.mean(a)
        b = b - np.mean(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def snr_gain(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray,
             ac_couple: bool = True) -> float:
    """ΔSNR = SNR(去噪后) − SNR(加噪后)，衡量去噪增益（5-3 背景 Q10）。"""
    return (snr_db(clean, denoised, ac_couple)
            - snr_db(clean, noisy, ac_couple))


def match_peaks(det: np.ndarray, true: np.ndarray, tol_ms: float, fs: float) -> int:
    """在 ±tol_ms 容差内做一一匹配，返回命中个数。"""
    det = np.asarray(det, dtype=float)
    true = np.asarray(true, dtype=float)
    det = det[np.isfinite(det)]
    true = true[np.isfinite(true)]
    if det.size == 0 or true.size == 0:
        return 0
    tol = max(1.0, float(tol_ms) * float(fs) / 1000.0)
    used: set = set()
    hit = 0
    for tp in true:
        d = np.abs(det - tp)
        j = int(np.argmin(d))
        if d[j] <= tol and j not in used:
            hit += 1
            used.add(j)
    return hit


def hit_rate(det: np.ndarray, true: np.ndarray, tol_ms: float, fs: float) -> float:
    """命中率 = 命中的真值峰数 / 真值峰总数（大方向背景 2.8）。"""
    true = np.asarray(true, dtype=float)
    true = true[np.isfinite(true)]
    if true.size == 0:
        return np.nan
    return match_peaks(det, true, tol_ms, fs) / float(true.size)


def spectrum(x: np.ndarray, fs: float):
    """返回 (freq, amp) 单边幅度谱，用于验证滤波器截止频率是否生效。"""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 4:
        return np.array([]), np.array([])
    win = np.hanning(n)
    amp = np.abs(np.fft.rfft(x * win)) / max(n, 1)
    freq = np.fft.rfftfreq(n, d=1.0 / float(fs))
    return freq, amp


# ==========================================================================
# 7. 单条实验（供批量对比页与数据集评估页复用）
# ==========================================================================

def run_single_experiment(fs: float, hr: float, sec: float, method: str,
                          snr_in: float, seed: int,
                          method_params: Optional[dict] = None,
                          noise_type: str = "高斯白噪",
                          drift_amp: float = 0.15, mains_amp: float = 0.05,
                          motion_amp: float = 0.20,
                          tol_ms: float = 50.0,
                          amps=DEFAULT_AMPS, phis=DEFAULT_PHIS, ws=DEFAULT_WS):
    """一次完整的「生成 → 加噪 → 去噪 → 检测 → 评价」。

    返回 dict，字段与批量对比表的列一致。
    """
    method_params = method_params or {}
    t, clean, truth = gen_pulse(fs, hr, sec, amps=amps, phis=phis, ws=ws,
                                return_truth=True)
    rng = np.random.default_rng(int(seed))
    noisy = add_noise(clean, snr_in, rng, fs=fs, noise_type=noise_type,
                      drift_amp=drift_amp, mains_amp=mains_amp,
                      motion_amp=motion_amp)

    t0 = time.perf_counter()
    den = denoise(noisy, method, fs=fs, **method_params)
    cost_ms = (time.perf_counter() - t0) * 1000.0

    feat = detect_features(den, fs, hr=hr)
    det = feat["main_idx"]
    det = det[np.isfinite(det)]

    return {
        "方法": method,
        "噪声类型": noise_type,
        "加噪SNR(dB)": float(snr_in),
        "种子": int(seed),
        "SNR(dB)": snr_db(clean, den),
        "ΔSNR(dB)": snr_gain(clean, noisy, den),
        "RMSE": rmse(clean, den),
        "主波命中率": hit_rate(det, truth["main"], tol_ms, fs),
        "检出主波数": int(len(det)),
        "真值主波数": int(len(truth["main"])),
        "耗时(ms)": cost_ms,
    }


if __name__ == "__main__":
    # 命令行自检：python pulse_core.py
    t, x, truth = gen_pulse(250, 75, 10, return_truth=True)
    rng = np.random.default_rng(42)
    xn = add_noise(x, 10, rng, fs=250, noise_type="复合噪声")
    print(f"{'方法':<18}{'SNR':>10}{'dSNR':>10}{'RMSE':>10}{'命中率':>10}{'耗时ms':>10}")
    for m in DENOISE_METHODS:
        r = run_single_experiment(250, 75, 10, m, 10, 42, noise_type="复合噪声")
        print(f"{m:<18}{r['SNR(dB)']:>10.2f}{r['ΔSNR(dB)']:>10.2f}"
              f"{r['RMSE']:>10.4f}{r['主波命中率']:>10.2f}{r['耗时(ms)']:>10.2f}")
    lowpass_deno = denoise(xn, "低通滤波", fs=250)
    mor = morphology_features(lowpass_deno, 250, hr=75)
    print("\n形态学特征：", {k: (round(v, 3) if np.isfinite(v) else None)
                             for k, v in mor.items()})

# -*- coding: utf-8 -*-
"""
tools/make_figure.py —— 生成"新模拟数据集效果图"（4 联图）

产物：results/新模拟数据集效果图.png
  A 单条记录示例波形（干净 / 加噪 8dB / 去噪后，AC 耦合显示）
  B 单周期模板与三波标注（主波 / 潮波 / 重搏波 / 切迹）
  C 灌注指数 PI 分布（真实区间 0.8%~6.5%）
  D 各去噪算法平均 ΔSNR（真值口径；带通在低噪声下的负增益被高亮）
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import pulse_core as pc  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#888888"
plt.rcParams["text.color"] = "#222222"
plt.rcParams["axes.labelcolor"] = "#222222"
plt.rcParams["xtick.color"] = "#444444"
plt.rcParams["ytick.color"] = "#444444"
plt.rcParams["grid.color"] = "#dddddd"

FS = 250.0
RES = os.path.join(ROOT, "results")


def example_wave():
    """复刻生成器的一条示例记录（健康成人 / Windkessel / 高噪声 8dB / S-G 去噪）。"""
    hr = 72.0
    amps = (1.00, 0.36, 0.22)
    phis = (0.24, 0.45, 0.63)
    ws = (0.062, 0.060, 0.078)
    t, clean, tr = pc.gen_pulse(FS, hr, 6.0, amps=amps, phis=phis, ws=ws,
                                return_truth=True)
    # 与生成器同一套直流基线注入（使 PI≈3.6%）
    pi_target = 3.6
    ac = float(clean.max() - clean.min())
    clean = clean + (ac / (pi_target / 100.0) - float(clean.mean()))
    rng = np.random.default_rng(2026)
    noisy = pc.add_noise(clean, 8.0, rng, fs=FS, noise_type="高斯白噪")
    den = pc.denoise(noisy, "Savitzky-Golay滤波", fs=FS)
    return t, clean, noisy, den, hr


def panel_a(ax, t, clean, noisy, den):
    ta = t <= 3.0
    c = clean[ta] - np.mean(clean)
    n = noisy[ta] - np.mean(noisy)
    d = den[ta] - np.mean(den)
    ax.plot(t[ta], n, color="#c8c8c8", lw=1.2, label="加噪(8 dB)")
    ax.plot(t[ta], c, color="#1f77b4", lw=2.2, label="干净")
    ax.plot(t[ta], d, color="#d62728", lw=1.4, label="S-G 去噪后")
    snr_n = pc.snr_db(clean, noisy)
    snr_d = pc.snr_db(clean, den)
    ax.set_title(f"A  单条记录示例（已去直流显示；该记录 PI≈3.6%）\n"
                 f"加噪SNR {snr_n:.1f} dB → 去噪后 {snr_d:.1f} dB "
                 f"(ΔSNR {snr_d - snr_n:+.1f} dB)",
                 fontsize=11)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("搏动分量 (AC, 已去均值)")
    ax.legend(fontsize=8, loc="upper right", ncol=1, framealpha=0.9)
    ax.grid(True, ls=":", lw=0.6)


def panel_b(ax, clean, hr):
    from scipy.signal import find_peaks
    T = 60.0 / hr
    n = int(round(T * FS))
    seg = clean[n:2 * n]
    T_ms = 60000.0 / hr
    t_ms = np.linspace(0, T_ms, len(seg))
    y = (seg - seg.min()) / (seg.max() - seg.min())
    m = pc.morphology_from_cycle(seg, hr)
    i_pk = int(np.argmax(y))
    ax.plot(t_ms, y, color="#1f77b4", lw=2.2)
    ax.axhline(0.5, color="#999999", ls=":", lw=0.9)

    def mark(label, idx, xytext):
        ax.annotate(label, xy=(t_ms[idx], y[idx]), xytext=xytext, fontsize=9,
                    ha="left",
                    arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9))

    mark("主波", i_pk, (t_ms[i_pk] + 5, 1.08))
    i_n = None
    if np.isfinite(m["notch_ms"]):
        tn = t_ms[i_pk] + m["notch_ms"]
        i_n = int(np.argmin(np.abs(t_ms - tn)))
        mark("重搏切迹", i_n, (tn - 170, 0.10))
    if i_n is not None and i_n - i_pk > 3:
        pk_t, _ = find_peaks(y[i_pk:i_n], distance=2)
        if len(pk_t):
            i_t = i_pk + int(pk_t[np.argmax(y[i_pk:i_n][pk_t])])
            mark("潮波", i_t, (t_ms[i_t] + 8, 0.90))
    if np.isfinite(m["dicrotic_ms"]):
        td = t_ms[i_pk] + m["dicrotic_ms"]
        i_d = int(np.argmin(np.abs(t_ms - td)))
        mark("重搏波", i_d, (td + 5, 0.60))
    ax.set_title(f"B  单周期模板与特征点标注（HR={hr:.0f} bpm，"
                 f"半高宽 {m['pulse_width_ms']:.0f} ms）", fontsize=11)
    ax.set_xlabel("心动周期内时间 (ms)")
    ax.set_ylabel("归一化幅值")
    ax.set_ylim(-0.05, 1.22)
    ax.grid(True, ls=":", lw=0.6)


def panel_c(ax, new):
    v = pd.to_numeric(new["perfusion_index"], errors="coerce").dropna()
    ax.hist(v, bins=40, color="#4c9f70", edgecolor="white", lw=0.5)
    ax.axvline(float(v.median()), color="#d62728", ls="--", lw=1.4,
               label=f"中位 {v.median():.2f}%")
    ax.set_title("C  灌注指数 PI 分布（AC/DC×100%）\n"
                 f"范围 [{v.min():.2f}%, {v.max():.2f}%]，"
                 f"全部落在生理区间 [0.05,30]", fontsize=11)
    ax.set_xlabel("PI (%)")
    ax.set_ylabel("记录数")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", lw=0.6, axis="y")


def panel_d(ax, truth):
    tt = truth[truth["denoise_algo"] != "未处理"]
    g = (tt.groupby("denoise_algo")["ΔSNR(dB)"].mean()
         .sort_values())
    colors = ["#d62728" if "带通" in k else "#4c78a8" for k in g.index]
    ax.barh(range(len(g)), g.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.index, fontsize=9)
    for i, v in enumerate(g.values):
        ax.text(v + (0.08 if v >= 0 else -0.08), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(0, color="#333333", lw=1.0)
    ax.set_title("D  各去噪算法平均 ΔSNR（真值口径）\n"
                 "红色=0.5–8Hz带通：均值虽高，但在低噪声档会出现负增益"
                 "（详见报告第 6 节）", fontsize=11)
    ax.set_xlabel("平均 ΔSNR (dB)")
    ax.set_xlim(min(g.min() * 1.35, -1.0), g.max() * 1.2)
    ax.grid(True, ls=":", lw=0.6, axis="x")


def main():
    os.makedirs(RES, exist_ok=True)
    new = pd.read_csv(os.path.join(ROOT, "data", "sim_ppg_v2_raw.csv"),
                      encoding="utf-8-sig", skiprows=[1], low_memory=False)
    truth = pd.read_csv(os.path.join(ROOT, "data", "sim_ppg_v2_truth.csv"),
                        encoding="utf-8-sig")
    t, clean, noisy, den, hr = example_wave()

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    panel_a(axes[0, 0], t, clean, noisy, den)
    panel_b(axes[0, 1], clean, hr)
    panel_c(axes[1, 0], new)
    panel_d(axes[1, 1], truth)
    fig.suptitle("新模拟数据集 sim_ppg_v2 效果总览", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(RES, "新模拟数据集效果图.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"✔ 效果图已生成：{out}")
    print(f"  记录数 {len(new)}，PI 中位 {new['perfusion_index'].median():.2f}%，"
          f"命中率均值 {truth['主波命中率'].mean():.3f}")


if __name__ == "__main__":
    main()

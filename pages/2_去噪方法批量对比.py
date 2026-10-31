# -*- coding: utf-8 -*-
"""
页面 ③ 去噪方法批量对比

对应 5-4 指导.md：
  步骤 3.2 设计对比实验页（st.session_state 缓存）
  步骤 4.2 实现批量对比页（方法 × 加噪档 指标表）
  步骤 5.2 性能优化（@st.cache_data）
  步骤 5.3 实验导出（st.download_button）
"""
import time

import numpy as np
import pandas as pd
import streamlit as st

import pulse_core as pc
import ui_utils as ui

ui.page_config("去噪方法批量对比")
st.title("③ 去噪方法批量对比")
st.caption("固定加噪档位 × 多种去噪方法 × 多个随机种子 → 指标表 / 折线 / 热力图。"
           "本页结果是论文第 4 章「去噪算法对比」的主要数据来源。")


@st.cache_data(show_spinner=False)
def run_batch(fs, hr, sec, amps, phis, ws, methods, snr_list, n_seed,
              noise_type, tol_ms, drift, mains, motion, base_seed):
    """批量实验（结果按参数缓存，重复调参不会重复计算 —— 指导 5.2）。"""
    rows = []
    n_total = len(methods) * len(snr_list) * n_seed
    prog = st.progress(0.0, text="批量实验进行中…")
    done = 0
    for m in methods:
        for s in snr_list:
            for k in range(n_seed):
                seed = base_seed + k
                rows.append(pc.run_single_experiment(
                    fs=fs, hr=hr, sec=sec, method=m, snr_in=s, seed=seed,
                    noise_type=noise_type, drift_amp=drift,
                    mains_amp=mains, motion_amp=motion, tol_ms=tol_ms,
                    amps=amps, phis=phis, ws=ws))
                done += 1
                if done % max(1, n_total // 50) == 0 or done == n_total:
                    prog.progress(done / n_total,
                                  text=f"批量实验 {done}/{n_total}")
    prog.empty()
    return pd.DataFrame(rows)


@ui.guard
def main():
    # ================= 侧边栏 =================
    st.sidebar.header("🎛 实验配置")
    fs = st.sidebar.slider("采样率 fs (Hz)", 100, 1000, 250, 25, key="b_fs")
    hr = st.sidebar.slider("心率 HR (bpm)", 40, 180, 75, 1, key="b_hr")
    sec = st.sidebar.slider("每条信号时长 (s)", 3, 20, 8, 1, key="b_sec")
    noise_type = st.sidebar.selectbox("噪声类型", list(pc.NOISE_TYPES),
                                      index=0, key="b_ntype")
    snr_list = st.sidebar.multiselect(
        "加噪档位 SNR (dB)", [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        default=[5.0, 10.0, 15.0, 20.0], key="b_snr")
    methods = st.sidebar.multiselect(
        "去噪方法", list(pc.DENOISE_METHODS.keys()),
        default=list(pc.DENOISE_METHODS.keys()), key="b_methods")
    n_seed = st.sidebar.slider("随机种子个数", 1, 10, 3, 1, key="b_nseed")
    base_seed = st.sidebar.number_input("起始种子", 0, 9999, 42, 1, key="b_seed")
    tol_ms = st.sidebar.slider("命中容差 tol (ms)", 10.0, 200.0, 50.0, 10.0,
                               key="b_tol")
    drift = st.sidebar.slider("基线漂移幅度", 0.0, 0.8, 0.15, 0.05, key="b_drift")
    mains = st.sidebar.slider("工频干扰幅度", 0.0, 0.3, 0.05, 0.01, key="b_mains")
    motion = st.sidebar.slider("运动伪影幅度", 0.0, 0.6, 0.20, 0.05, key="b_motion")

    if not snr_list or not methods:
        st.info("请至少选择一个加噪档位与一种去噪方法。"); st.stop()

    n_runs = len(methods) * len(snr_list) * n_seed
    st.sidebar.caption(f"本次将执行 {n_runs} 次实验")

    if st.sidebar.button("▶ 开始批量实验", type="primary", key="b_run"):
        st.session_state["batch_t0"] = time.perf_counter()
        st.session_state["batch_done"] = True

    if not st.session_state.get("batch_done"):
        st.info("点击左侧「开始批量实验」运行。结果会被缓存，"
                "仅改动参数才会重新计算（@st.cache_data + session_state）。")
        st.stop()

    t0 = time.perf_counter()
    df = run_batch(float(fs), float(hr), float(sec), pc.DEFAULT_AMPS,
                   pc.DEFAULT_PHIS, pc.DEFAULT_WS, tuple(methods),
                   tuple(sorted(snr_list)), int(n_seed), noise_type,
                   float(tol_ms), float(drift), float(mains), float(motion),
                   int(base_seed))
    wall = time.perf_counter() - t0

    st.success(f"完成 {len(df)} 次实验，总耗时 {wall:.2f} s"
               f"（若命中缓存则接近 0 s）")

    # ================= 明细 =================
    st.subheader("① 实验明细")
    with st.expander("展开完整明细（方法 × 加噪档 × 种子）", expanded=False):
        st.dataframe(df.round(4), width="stretch", hide_index=True)

    # ================= 透视表 =================
    st.subheader("② 透视表（按加噪档平均）")
    tab1, tab2, tab3 = st.tabs(["ΔSNR (dB)", "RMSE", "主波命中率"])
    piv = {}
    for name, col, fmt, better in (("ΔSNR(dB)", "ΔSNR(dB)", "{:+.2f}", "max"),
                                   ("RMSE", "RMSE", "{:.4f}", "min"),
                                   ("主波命中率", "主波命中率", "{:.3f}", "max")):
        piv[col] = df.pivot_table(index="方法", columns="加噪SNR(dB)",
                                  values=col, aggfunc="mean")
    with tab1:
        st.dataframe(piv["ΔSNR(dB)"].round(2).style.background_gradient(
            cmap="RdYlGn", axis=None), width="stretch")
        st.pyplot(ui.plot_heatmap(piv["ΔSNR(dB)"], "ΔSNR (dB) 热力图",
                                  cmap="RdYlGn", fmt="{:+.1f}"))
    with tab2:
        st.dataframe(piv["RMSE"].round(4).style.background_gradient(
            cmap="RdYlGn_r", axis=None), width="stretch")
        st.pyplot(ui.plot_heatmap(piv["RMSE"], "RMSE 热力图（越小越好）",
                                  cmap="RdYlGn_r", fmt="{:.3f}"))
    with tab3:
        st.dataframe(piv["主波命中率"].round(3).style.background_gradient(
            cmap="RdYlGn", axis=None), width="stretch")
        st.pyplot(ui.plot_heatmap(piv["主波命中率"], "主波命中率热力图",
                                  cmap="RdYlGn", fmt="{:.2f}"))

    # ================= 折线 =================
    st.subheader("③ 指标随加噪强度的变化")
    agg = df.groupby(["方法", "加噪SNR(dB)"], as_index=False).agg(
        {"ΔSNR(dB)": "mean", "RMSE": "mean", "主波命中率": "mean",
         "耗时(ms)": "mean"})
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(ui.plot_lines(agg, "加噪SNR(dB)", "ΔSNR(dB)", "方法",
                                "ΔSNR 随加噪强度变化", "ΔSNR (dB)"))
    with c2:
        st.pyplot(ui.plot_lines(agg, "加噪SNR(dB)", "主波命中率", "方法",
                                "主波命中率随加噪强度变化", "命中率"))

    # ================= 排名与耗时 =================
    st.subheader("④ 综合排名与性能")
    summary = df.groupby("方法", as_index=False).agg(
        ΔSNR均值=("ΔSNR(dB)", "mean"),
        RMSE均值=("RMSE", "mean"),
        命中率均值=("主波命中率", "mean"),
        平均耗时ms=("耗时(ms)", "mean"),
        实验次数=("ΔSNR(dB)", "count"),
    ).sort_values("ΔSNR均值", ascending=False)
    summary["排名"] = np.arange(1, len(summary) + 1)
    c3, c4 = st.columns([1, 1])
    with c3:
        st.dataframe(summary.round(4), width="stretch", hide_index=True)
    with c4:
        st.pyplot(ui.plot_radar(summary, "ΔSNR均值", "各方法 ΔSNR 均值雷达图"))

    st.pyplot(_cost_bar(summary))

    best = summary.iloc[0]
    st.markdown(
        f"**结论摘要**：在 `噪声类型 = {noise_type}`、加噪档 "
        f"`{sorted(snr_list)} dB`、{n_seed} 个随机种子的条件下，"
        f"**{best['方法']}** 的 ΔSNR 均值最高（**{best['ΔSNR均值']:+.2f} dB**），"
        f"RMSE 均值 {best['RMSE均值']:.4f}，主波命中率 "
        f"{best['命中率均值']*100:.1f}%。"
        f"其中 **无（基线）** 一行的 ΔSNR 恒为 0，可作为对照基准。")

    st.divider()
    st.subheader("⑤ 导出")
    ui.download_df(df, "denoise_batch_detail.csv", "⬇ 实验明细 CSV")
    ui.download_df(summary, "denoise_batch_summary.csv", "⬇ 方法排名汇总 CSV")
    st.caption("两份 CSV 可直接作为论文表 4-x 的数据源；"
               "固定起始种子 %d，结果可完全复现。" % int(base_seed))


def _cost_bar(summary: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=110)
    ax.bar(summary["方法"].astype(str), summary["平均耗时ms"],
           color="#566573")
    ax.set_ylabel("平均耗时 (ms)")
    ax.set_title("各方法单条信号处理耗时（性能佐证）")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y")
    return fig


if __name__ == "__main__":
    main()

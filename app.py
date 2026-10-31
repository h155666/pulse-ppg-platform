# -*- coding: utf-8 -*-
"""
app.py —— 人体脉搏波信号生成与特征分析虚拟仿真平台（课题 5-4）· 入口与总览页

运行：
    streamlit run app.py
    streamlit run app.py --server.port 8502     # 端口被占用时（5-4 背景 Q4）

页面导航（pages/ 目录自动挂载）：
    ① 平台总览（本文件）
    ② 信号生成与特征分析
    ③ 去噪方法批量对比
    ④ 仿真数据集治理与分析
    ⑤ 形态学特征与去噪评估
    ⑥ 关于与运行说明
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

import data_lab as dl
import pulse_core as pc
import ui_utils as ui

ui.page_config("脉搏波信号生成与特征分析平台")


@ui.guard
def main():
    st.title("🫀 人体脉搏波信号生成与特征分析虚拟仿真平台")
    st.caption("课题 5-4 · 基于 Streamlit + SciPy + Matplotlib 的网页端平台 ｜ "
               "数据源：raw_p11_ppg_sim_signal.csv")

    # ---------------- 侧边栏 ----------------
    st.sidebar.title("导航")
    st.sidebar.markdown(
        "① **平台总览**（当前）\n"
        "② 信号生成与特征分析\n"
        "③ 去噪方法批量对比\n"
        "④ 仿真数据集治理与分析\n"
        "⑤ 形态学特征与去噪评估\n"
        "⑥ 关于与运行说明")
    st.sidebar.divider()
    src = ui.data_source_sidebar()
    st.sidebar.divider()
    strategy = st.sidebar.selectbox(
        "异常值治理策略", dl.STRATEGIES,
        index=dl.STRATEGIES.index("分组中位数填补"), key="ov_strategy")

    # ---------------- 载入数据 ----------------
    df = None
    if src["source"] == "内置仿真数据集":
        if not src["path"]:
            st.warning("未找到内置数据集。请在左侧填写正确路径，或改用「上传 CSV」。")
            st.stop()
        raw, clean, log, cost = ui.load_dataset_cached(src["path"], strategy)
        df = clean
        raw_df = raw
    elif src["source"] == "上传 CSV":
        if src["uploaded"] is None:
            st.info("请在左侧上传 CSV 文件（两列 time,value 或本数据集格式）。")
            st.stop()
        raw = ui.read_upload_cached(src["uploaded"].getvalue())
        clean, log, cost = dl.clean_dataframe(raw, strategy=strategy)
        df, raw_df = clean, raw
    else:
        st.info("当前选择「平台自生成（有真值）」，请前往页面 ② 调节参数生成信号。")
        st.markdown("### 平台能力一览")
        st.markdown("""
| 能力 | 说明 |
|---|---|
| 信号生成 | 自适应高斯脉冲串，三波位置随心率自动缩放 |
| 噪声叠加 | 高斯白噪 / 基线漂移 / 工频干扰 / 运动伪影 / 复合噪声 |
| 去噪方法 | 无、低通、小波、EMD、Savitzky-Golay、带通、中值、陷波+低通（共 8 种） |
| 特征检测 | 主波、潮波、重搏波、重搏切迹 |
| 形态学 | 脉宽、上升时间、AIx、RI、SI、PI、K 值、dp/dt_max |
| 评价指标 | SNR、ΔSNR、RMSE、主波命中率、处理耗时 |
| 批量实验 | 方法 × 加噪档 × 随机种子，透视表 / 折线 / 热力图 / 导出 |
| 数据治理 | 字段口径归一、异常值识别与治理、波形解析与特征工程 |
""")
        st.stop()

    # ---------------- KPI ----------------
    st.subheader("① 数据集概览")
    size_mb = (os.path.getsize(src["path"]) / 1024 / 1024) \
        if src["source"] == "内置仿真数据集" and src["path"] else np.nan
    ui.metric_row([
        ("有效记录数", f"{len(df):,}", "已剔除第 2 行字段说明记录"),
        ("字段数", f"{df.shape[1]}", "原始 48 列 + 治理新增列"),
        ("受试者类型", f"{df['subject_type'].nunique() if 'subject_type' in df else '-'}",
         "健康成人 / 高血压 / 运动员 等"),
        ("仿真生成模型", f"{df['sim_model'].nunique() if 'sim_model' in df else '-'}",
         "Windkessel / 双高斯 / LSTM 等"),
        ("治理耗时 (s)", f"{cost:.2f}", "清洗 + 归一 + 异常治理总耗时"),
        ("数据体积 (MB)", f"{size_mb:.1f}" if size_mb == size_mb else "-",
         "单机 pandas 可完全承载，无需 Spark 集群"),
    ], ncol=6)

    st.divider()

    # ---------------- 分布图 ----------------
    c1, c2 = st.columns(2)
    with c1:
        if "subject_type" in df.columns:
            vc = df["subject_type"].astype(str).value_counts().reset_index()
            vc.columns = ["受试者类型", "记录数"]
            st.pyplot(ui.plot_group_bar(vc, "受试者类型", "记录数",
                                        "受试者类型分布", horizontal=True))
    with c2:
        if "sim_model" in df.columns:
            vc = df["sim_model"].astype(str).value_counts().reset_index()
            vc.columns = ["仿真模型", "记录数"]
            st.pyplot(ui.plot_group_bar(vc, "仿真模型", "记录数",
                                        "仿真生成模型分布", horizontal=True))

    c3, c4 = st.columns(2)
    with c3:
        if {"noise_type", "noise_level"}.issubset(df.columns):
            ct = pd.crosstab(df["noise_type"].astype(str),
                             df["noise_level"].astype(str))
            fig, ax = _stacked(ct, "噪声类型 × 噪声等级 分布")
            st.pyplot(fig)
    with c4:
        if {"heart_rate", "sbp_mmhg", "subject_type"}.issubset(df.columns):
            st.pyplot(_hr_sbp_scatter(df))

    st.divider()

    # ---------------- 数据质量概览 ----------------
    st.subheader("② 数据质量概览（缺失 / 越界 Top 15）")
    qr = ui.quality_cached(raw_df)
    bad = qr[(qr["缺失率(%)"] > 0) | (qr["越界率(%)"] > 0)].copy()
    bad = bad.sort_values(["缺失率(%)", "越界率(%)"], ascending=False).head(15)
    st.dataframe(bad, width="stretch", hide_index=True)
    st.caption("「越界」按生理/工程合理区间判定，详见 ④ 页面的规则表。")

    st.divider()

    # ---------------- 快速开始 ----------------
    st.subheader("③ 三步快速开始")
    st.markdown("""
1. **页面 ② 信号生成与特征分析** —— 在左侧拖动「采样率 / 心率 / 加噪 SNR / 去噪方法」，
   主区实时给出「干净—加噪—去噪」三叠波形、单周期三波标注图、频谱、残差与 6 个指标卡；
2. **页面 ③ 去噪方法批量对比** —— 勾选方法与加噪档位，一键跑出「方法 × 加噪档」指标表并导出 CSV，
   这是论文第 4 章的主要数据来源；
3. **页面 ④⑤ 数据集治理与评估** —— 查看 17835 条仿真记录的字段归一、异常治理效果，
   以及在真实仿真波形上复现的 8 种去噪算法排名。
""")
    with st.expander("本平台与 5.1 / 5.2 / 5.3 的技术边界（答辩要点）"):
        st.markdown("""
| 课题 | 技术栈 | 与本机共用什么 |
|---|---|---|
| 5.1 PyQt 桌面 | PyQt5 + matplotlib | 共用 `pulse_core.py` 算法内核 |
| 5.2 Vue 前端 | Vue3 + ECharts（纯 JS） | 算法需用 JavaScript 重写一遍 |
| 5.3 纯算法 | scipy + PyWavelets | 算法内核即本平台的 `pulse_core.py` |
| **5.4 本平台** | **Streamlit + scipy + matplotlib** | **直接 import，零前端代码** |

本平台在 5-4「技术栈固定」的硬约束下，用 pandas 单机列向量化等价实现了项目指导书中
Spark/Hive 负责的「清洗—标准化—特征工程」三段，处理耗时见上方 KPI，可作为工程佐证写入论文。
""")


def _stacked(ct: pd.DataFrame, title: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.0), dpi=110)
    bottom = np.zeros(len(ct))
    for i, col in enumerate(ct.columns):
        vals = ct[col].to_numpy(dtype=float)
        ax.bar(ct.index.astype(str), vals, bottom=bottom,
               label=str(col), color=ui.PALETTE[i % len(ui.PALETTE)])
        bottom += vals
    ax.set_ylabel("记录数")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y")
    return fig, ax


def _hr_sbp_scatter(df: pd.DataFrame, n=3000):
    import matplotlib.pyplot as plt
    sub = df[["heart_rate", "sbp_mmhg", "subject_type"]].dropna()
    sub = sub.sample(min(n, len(sub)), random_state=0) if len(sub) > n else sub
    x = pd.to_numeric(sub["heart_rate"], errors="coerce")
    y = pd.to_numeric(sub["sbp_mmhg"], errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 4.0), dpi=110)
    groups = sub["subject_type"].astype(str).unique().tolist()
    for i, g in enumerate(groups):
        m = sub["subject_type"].astype(str) == g
        ax.scatter(x[m], y[m], s=8, alpha=0.5,
                   color=ui.PALETTE[i % len(ui.PALETTE)], label=g,
                   edgecolors="none")
    ax.set_xlabel("心率 (bpm)")
    ax.set_ylabel("收缩压 (mmHg)")
    ax.set_title("心率 — 收缩压 分布（按受试者类型着色）")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True)
    return fig


if __name__ == "__main__":
    main()

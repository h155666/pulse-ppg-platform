# -*- coding: utf-8 -*-
"""
页面 ④ 仿真数据集治理与分析

对应《大数据开发项目指导书》任务 2（数据规范清洗）、任务 3（数仓 ODS→DWD→DWS）、
任务 4（波形特征工程），以及 5-4 指导 2.4（上传 CSV 分析真实数据）。
"""
import numpy as np
import pandas as pd
import streamlit as st

import data_lab as dl
import pulse_core as pc
import ui_utils as ui

ui.page_config("仿真数据集治理与分析")
st.title("④ 仿真数据集治理与分析")
st.caption("字段口径归一 → 异常值识别与治理 → 群体分布统计 → 波形库抽样。"
           "单机 pandas 完成，与指导书中 Spark 清洗口径等价。")

KEY_NUMERIC = ["point_count", "augmentation_index", "snr_db", "dbp_mmhg",
               "cardiac_output", "pulse_amplitude", "dpdt_max", "age",
               "height_cm", "weight_kg", "heart_rate", "stiffness_index"]


@st.cache_data(show_spinner="正在抽样解析波形…")
def sample_waves_cached(_df: pd.DataFrame, n_sample: int, seed: int,
                        subject_type: str):
    sub = dl.sample_records(_df, n_sample, seed,
                            None if subject_type == "全部" else subject_type)
    waves = dl.decode_waves(sub)
    labels = (sub["subject_type"].astype(str) + " / " +
              sub["sim_model"].astype(str)).tolist()
    return waves, labels


@ui.guard
def main():
    st.sidebar.header("⚙️ 治理配置")
    path = st.sidebar.text_input("数据集路径",
                                 value=dl.resolve_data_path() or "",
                                 key="p4_path")
    uploaded = st.sidebar.file_uploader("或上传 CSV", type=["csv"], key="p4_up")
    strategy = st.sidebar.selectbox("异常值治理策略", dl.STRATEGIES,
                                    index=dl.STRATEGIES.index("分组中位数填补"),
                                    key="p4_strategy")
    group_by = st.sidebar.selectbox("分组填补依据",
                                    ["subject_type", "sim_model", "gender"],
                                    key="p4_groupby")
    apply_range = st.sidebar.checkbox("启用生理/工程区间治理", value=True,
                                      key="p4_range")

    if uploaded is not None:
        raw = ui.read_upload_cached(uploaded.getvalue())
    elif path and __import__("os").path.exists(path):
        raw = dl.load_raw(path)
    else:
        st.warning("未找到数据集，请在左侧指定路径或上传 CSV。"); st.stop()

    t0 = __import__("time").perf_counter()
    clean, log, _ = dl.clean_dataframe(raw, strategy=strategy,
                                       group_by=group_by,
                                       apply_range=apply_range)
    cost = __import__("time").perf_counter() - t0

    ui.metric_row([
        ("原始记录", f"{len(raw):,}", "已剔除字段说明行"),
        ("治理后记录", f"{len(clean):,}", "按 signal_id 去重后"),
        ("字段数", f"{raw.shape[1]} → {clean.shape[1]}", "治理可能新增标记列"),
        ("治理日志", f"{len(log)} 条", "每一步都可追溯"),
        ("治理耗时 (s)", f"{cost:.2f}", "单机 pandas，无需集群"),
    ], ncol=5)

    tabs = st.tabs(["① 数据质量报告", "② 字段口径归一", "③ 异常值治理",
                    "④ 群体分布与波形库"])

    # ---------------- ① 质量报告 ----------------
    with tabs[0]:
        st.subheader("逐列质量报告")
        qr = ui.quality_cached(raw)
        bad = qr[(qr["缺失率(%)"] > 0) | (qr["越界率(%)"] > 0)]
        st.dataframe(qr, width="stretch", hide_index=True, height=420)
        c1, c2 = st.columns(2)
        with c1:
            top = qr.sort_values("缺失率(%)", ascending=False).head(12)
            st.pyplot(ui.plot_group_bar(top, "字段", "缺失率(%)",
                                        "缺失率 Top12 (%)"))
        with c2:
            oob = qr[qr["越界率(%)"] > 0].sort_values("越界率(%)",
                                                     ascending=False).head(12)
            if len(oob):
                st.pyplot(ui.plot_group_bar(oob, "字段", "越界率(%)",
                                            "越界率 Top12 (%)"))
            else:
                st.info("未发现越界字段")
        st.caption(f"共 {len(qr)} 个字段，其中 {len(bad)} 个存在缺失或越界。")

    # ---------------- ② 口径归一 ----------------
    with tabs[1]:
        st.subheader("字段口径归一")
        st.markdown("数据集里的分类字段存在大量混写（如性别有 `男/Male/f/1/先生` 等），"
                    "不做归一会导致分组统计完全失真。")
        cols = [c for c in dl.NORMALIZERS if c in raw.columns]
        for c in cols:
            before = (raw[c].astype(str).str.strip()
                      .replace({"": np.nan, "nan": np.nan}).dropna())
            after = clean[c].dropna()
            b1, b2 = st.columns([1, 2])
            with b1:
                st.markdown(f"**{c}**：{before.nunique()} 种写法 → "
                            f"{after.nunique()} 种口径")
                cmp_df = pd.DataFrame({
                    "原始写法": before.value_counts().index[:40],
                    "出现次数": before.value_counts().values[:40],
                })
                cmp_df["归一后"] = cmp_df["原始写法"].map(
                    lambda v: dl.NORMALIZERS[c](v))
                st.dataframe(cmp_df, width="stretch",
                             hide_index=True, height=200)
            with b2:
                vc = after.value_counts().reset_index()
                vc.columns = ["口径", "记录数"]
                st.pyplot(ui.plot_group_bar(vc, "口径", "记录数",
                                            f"{c} 归一后分布"))

    # ---------------- ③ 异常值治理 ----------------
    with tabs[2]:
        st.subheader("异常值识别与治理")
        st.markdown("**判定规则表**（生理 / 工程合理区间，来自指导书 3.4）")
        rules_df = pd.DataFrame([
            {"字段": k, "合理区间": f"[{v[0]}, {v[1]}]",
             "原始越界数": int(_count_outlier(raw, k, v)),
             "治理后越界数": int(_count_outlier(clean, k, v))}
            for k, v in dl.NUMERIC_RULES.items() if k in raw.columns
        ])
        st.dataframe(rules_df, width="stretch", hide_index=True)

        st.markdown("**治理前后统计对比**")
        stat_rows = []
        for c in KEY_NUMERIC:
            if c not in raw.columns:
                continue
            a = pd.to_numeric(raw[c], errors="coerce")
            b = pd.to_numeric(clean[c], errors="coerce")
            stat_rows.append({
                "字段": c,
                "前·有效数": int(a.notna().sum()),
                "前·最小值": _r(a.min()), "前·中位数": _r(a.median()),
                "前·最大值": _r(a.max()),
                "后·有效数": int(b.notna().sum()),
                "后·最小值": _r(b.min()), "后·中位数": _r(b.median()),
                "后·最大值": _r(b.max()),
            })
        st.dataframe(pd.DataFrame(stat_rows), width="stretch",
                     hide_index=True)

        st.markdown("**point_count 特殊说明**")
        st.caption("该列存在 178 条负值与 97 条 2^30 量级哨兵值（max = 1073741824）。"
                   "本平台对 ≥10^6 的哨兵值按 `采样率 × 仿真时长` 重建，"
                   "负值则按策略置缺失后填补。")

        st.markdown("**治理日志**")
        st.dataframe(log, width="stretch", hide_index=True, height=300)
        ui.download_df(log, "dwd_clean_log.csv", "⬇ 治理日志 CSV")

    # ---------------- ④ 群体分布 ----------------
    with tabs[3]:
        st.subheader("群体分布分析")
        value = st.selectbox("形态学指标", [k for k in dl.NUMERIC_RULES
                                            if k in clean.columns],
                             index=min(2, len(clean.columns) - 1),
                             key="p4_value")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(ui.plot_box_by_group(clean, "subject_type", value,
                                           f"受试者类型 → {value} 分布"))
        with c2:
            st.pyplot(ui.plot_box_by_group(clean, "sim_model", value,
                                           f"仿真模型 → {value} 分布"))

        if {"noise_type", "denoise_snr_gain"}.issubset(clean.columns):
            g = (clean.groupby("noise_type", as_index=False)["denoise_snr_gain"]
                 .mean().sort_values("denoise_snr_gain", ascending=False))
            st.pyplot(ui.plot_group_bar(g, "noise_type", "denoise_snr_gain",
                                        "不同噪声类型下的平均去噪增益 (dB)"))

        st.subheader("波形库抽样")
        n_s = st.slider("抽样条数", 4, 24, 12, 2, key="p4_ns")
        s_type = st.selectbox("受试者类型过滤",
                              ["全部"] + ui.unique_str(clean["subject_type"]),
                              key="p4_stype")
        waves, labels = sample_waves_cached(clean, n_s, 42, s_type)
        st.pyplot(ui.plot_wave_library(waves, labels,
                                       "ppg_wave 单周期模板（归一化叠加）"))
        lens = [len(w) for w in waves]
        st.caption(f"抽样 {len(waves)} 条，单条点数范围 {min(lens)} – {max(lens)}；"
                   "该字段为逗号分隔的单周期采样序列，需先解析再参与计算。")

        st.divider()
        ui.download_df(clean.head(5000), "dwd_pulse_clean_sample.csv",
                       "⬇ 治理后数据（前 5000 行）")


def _count_outlier(df: pd.DataFrame, col: str, rng: tuple) -> int:
    if col not in df.columns:
        return 0
    v = pd.to_numeric(df[col], errors="coerce")
    return int(((v < rng[0]) | (v > rng[1])).fillna(False).sum())


def _r(v):
    try:
        return round(float(v), 3)
    except Exception:
        return np.nan


if __name__ == "__main__":
    main()

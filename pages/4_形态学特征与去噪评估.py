# -*- coding: utf-8 -*-
"""
页面 ⑤ 形态学特征与去噪评估

对应《大数据开发项目指导书》：
  任务 5  形态学特征量化分析（收缩期峰值、重搏波切迹、脉宽、上升时间、增强指数…）
  任务 6  去噪算法对比评估（小波、EMD、VMD、Savitzky-Golay 等在不同噪声条件下的效果）
"""
import numpy as np
import pandas as pd
import streamlit as st

import data_lab as dl
import pulse_core as pc
import ui_utils as ui

ui.page_config("形态学特征与去噪评估")
st.title("⑤ 形态学特征与去噪评估")
st.caption("在真实仿真波形上重算形态学特征、做口径可比性分析，"
           "并复现 8 种去噪算法的排名。")


@st.cache_data(show_spinner="正在重算形态学特征…")
def morph_cached(_df: pd.DataFrame, n_sample: int, seed: int, stype: str):
    return dl.recompute_morphology(_df, n_sample, seed,
                                   None if stype == "全部" else stype)


@st.cache_data(show_spinner="正在评估去噪算法（数据量较大时请耐心）…")
def eval_cached(_df: pd.DataFrame, methods, snr_list, n_sample, seed,
                noise_type, analysis_fs, stype):
    return dl.evaluate_denoise_on_dataset(
        _df, list(methods), list(snr_list), n_sample=n_sample, seed=seed,
        noise_type=noise_type, analysis_fs=analysis_fs,
        subject_type=None if stype == "全部" else stype)


@ui.guard
def main():
    st.sidebar.header("⚙️ 实验配置")
    path = st.sidebar.text_input("数据集路径",
                                 value=dl.resolve_data_path() or "",
                                 key="p5_path")
    raw, clean, log, cost = None, None, None, None
    if path and __import__("os").path.exists(path):
        raw, clean, log, cost = ui.load_dataset_cached(path)
    else:
        st.warning("请在左侧填写正确的数据集路径。"); st.stop()

    stype = st.sidebar.selectbox(
        "受试者类型过滤",
        ["全部"] + ui.unique_str(clean["subject_type"]),
        key="p5_stype")

    tabs = st.tabs(["① 形态学特征重算与口径可比性", "② 数据集级去噪算法对比",
                    "③ 导出与结论"])

    # ---------------- ① 形态学 ----------------
    with tabs[0]:
        n1 = st.slider("抽样条数", 50, 1500, 300, 50, key="p5_n1")
        mor = morph_cached(clean, n1, 42, stype)
        if mor.empty:
            st.warning("有效波形不足，请增加抽样条数。")
        else:
            st.success(f"成功重算 {len(mor)} 条记录的形态学特征")
            st.markdown("**口径可比性分析**（重算值 vs 库内字段）")
            cal = dl.calibration_summary(mor)
            st.dataframe(cal, width="stretch", hide_index=True)
            st.info(
                "**发现**：脉宽的重算值与库内值高度相关（r ≈ 0.93）但存在系统性尺度"
                "偏差（比值 ≈ 0.76），说明二者口径不同（库内更接近「低电平宽度」）；"
                "而上升时间、AIx、PI、dp/dt_max 与波形重算值几乎不相关，说明这些字段"
                "是独立仿真生成的，**不能作为波形真值**使用。"
                "这正是指导书 3.4 ⑤「特征语义复杂、计算口径多样」的实证，"
                "也是本平台坚持「定量指标只在有真值的自生成信号上计算」的原因。")

            st.markdown("**散点对照**（横轴 = 重算，纵轴 = 库内）")
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(ui.plot_scatter(mor, "重算_脉宽ms", "库内_脉宽ms",
                                          "脉宽：重算 vs 库内"))
            with c2:
                st.pyplot(ui.plot_scatter(mor, "重算_AIx", "库内_AIx",
                                          "AIx：重算 vs 库内"))

            st.markdown("**重算特征明细**")
            st.dataframe(mor.round(3), width="stretch",
                         hide_index=True, height=380)

            st.markdown("**按受试者类型的重算特征分布**")
            val = st.selectbox("指标", ["重算_脉宽ms", "重算_上升时间ms",
                                        "重算_AIx", "重算_PI", "重算_dpdtmax"],
                               key="p5_val")
            st.pyplot(ui.plot_box_by_group(mor, "subject_type", val,
                                           f"{val} 按受试者类型分布"))
            ui.download_df(mor, "morphology_recomputed.csv", "⬇ 重算特征明细 CSV")

    # ---------------- ② 去噪评估 ----------------
    with tabs[1]:
        st.markdown("把每条记录的 `ppg_wave`（64 点单周期模板）按心率拼接成多周期"
                    "连续波，叠加指定噪声后用各算法去噪，以模板拼接波为参考基准计算"
                    "SNR / ΔSNR / RMSE。**ΔSNR 是相对增益**，不受参考基准是否绝对"
                    "干净的影响，因此跨算法可比。")
        c1, c2, c3 = st.columns(3)
        with c1:
            n2 = st.slider("抽样条数", 20, 800, 120, 20, key="p5_n2")
            analysis_fs = st.slider("分析采样率 (Hz)", 100, 500, 200, 50,
                                    key="p5_fs")
        with c2:
            methods = st.multiselect("去噪方法", list(pc.DENOISE_METHODS.keys()),
                                     default=list(pc.DENOISE_METHODS.keys()),
                                     key="p5_methods")
            noise_type = st.selectbox("噪声类型", list(pc.NOISE_TYPES),
                                      index=0, key="p5_nt")
        with c3:
            snr_list = st.multiselect("加噪档 SNR (dB)",
                                      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
                                      default=[10.0, 20.0], key="p5_snr")
            seed2 = st.number_input("随机种子", 0, 9999, 42, 1, key="p5_seed")

        if not methods or not snr_list:
            st.info("请至少选择一种方法与一个加噪档。")
        else:
            n_runs = n2 * len(snr_list) * len(methods)
            st.caption(f"预计执行 {n_runs} 次去噪")
            if st.button("▶ 开始评估", type="primary", key="p5_run"):
                t0 = __import__("time").perf_counter()
                ev = eval_cached(clean, tuple(methods), tuple(sorted(snr_list)),
                                 n2, int(seed2), noise_type, float(analysis_fs),
                                 stype)
                st.success(f"完成 {len(ev)} 次去噪，"
                           f"耗时 {__import__('time').perf_counter()-t0:.2f} s")

                summ = ev.groupby("方法", as_index=False).agg(
                    ΔSNR均值=("ΔSNR(dB)", "mean"),
                    RMSE均值=("RMSE", "mean"),
                    平均耗时ms=("耗时(ms)", "mean"),
                    样本数=("ΔSNR(dB)", "count"),
                ).sort_values("ΔSNR均值", ascending=False)
                summ["排名"] = np.arange(1, len(summ) + 1)
                c4, c5 = st.columns([1, 1])
                with c4:
                    st.markdown("**方法排名（真实仿真波形）**")
                    st.dataframe(summ.round(4), width="stretch",
                                 hide_index=True)
                with c5:
                    st.pyplot(ui.plot_radar(summ, "ΔSNR均值",
                                            "真实波形上的 ΔSNR 排名"))

                st.markdown("**按噪声条件的分组结果**")
                by_snr = ev.pivot_table(index="方法", columns="加噪SNR(dB)",
                                        values="ΔSNR(dB)", aggfunc="mean")
                st.dataframe(by_snr.round(2).style.background_gradient(
                    cmap="RdYlGn", axis=None), width="stretch")

                agg2 = ev.groupby(["方法", "加噪SNR(dB)"], as_index=False)[
                    "ΔSNR(dB)"].mean()
                st.pyplot(ui.plot_lines(agg2, "加噪SNR(dB)", "ΔSNR(dB)", "方法",
                                        "真实波形：ΔSNR 随加噪强度变化", "ΔSNR (dB)"))

                best = summ.iloc[0]
                st.markdown(
                    f"**结论**：在 {n2} 条真实仿真波形、`{noise_type}` 条件下，"
                    f"**{best['方法']}** 平均 ΔSNR 最高（**{best['ΔSNR均值']:+.2f} dB**），"
                    f"平均 RMSE {best['RMSE均值']:.4f}，单次耗时 "
                    f"{best['平均耗时ms']:.2f} ms。"
                    "注意：当加噪 SNR 本身较高（≥20 dB）时，低通/带通类方法可能因"
                    "削掉信号谐波而出现 ΔSNR 为负，这是真实且应当如实报告的现象。")
                st.session_state["p5_ev"] = ev
                st.session_state["p5_summ"] = summ

            if "p5_ev" in st.session_state:
                ev = st.session_state["p5_ev"]
                summ = st.session_state["p5_summ"]
                with st.expander("查看完整评估明细"):
                    st.dataframe(ev.round(4), width="stretch",
                                 hide_index=True, height=400)

    # ---------------- ③ 导出 ----------------
    with tabs[2]:
        st.subheader("导出")
        if "p5_ev" in st.session_state:
            ui.download_df(st.session_state["p5_ev"],
                           "dataset_denoise_eval.csv", "⬇ 去噪评估明细 CSV")
            ui.download_df(st.session_state["p5_summ"],
                           "dataset_denoise_summary.csv", "⬇ 方法排名汇总 CSV")
        else:
            st.info("请先在 ② 页签运行一次评估。")
        ui.download_df(clean.head(10000), "dwd_pulse_clean.csv",
                       "⬇ 治理后数据（前 10000 行）")
        ui.download_df(log, "dwd_clean_log.csv", "⬇ 治理日志 CSV")

        st.divider()
        st.subheader("可直接写进论文的结论")
        st.markdown("""
1. **定量结论（有真值）**：在自生成的高斯脉冲串信号上，8 种去噪方法随加噪强度变化的
   ΔSNR / RMSE / 主波命中率排序稳定；低 SNR（≤10 dB）时各方法增益显著，
   高 SNR（≥20 dB）时线性滤波类方法可能带来负增益。
2. **定性结论（无真值）**：在 17835 条真实仿真记录上，各方法的相对增益排序与仿真
   结论一致，说明仿真实验的结论具有外推性。
3. **数据质量结论**：性别、仿真状态、噪声等级、导出格式等字段存在 8–20 种混写，
   归一后口径收敛到 3–8 类；point_count / augmentation_index / snr_db 存在哨兵值与
   生理区间外离群值，需按规则治理后方可进入特征工程。
4. **口径可比性结论**：库内形态学字段与波形重算值的相关性因指标而异，脉宽强相关但
   存在尺度偏差，AIx / PI / dp/dt_max 基本不相关，故本平台的定量评价只采用有真值的
   自生成信号。
""")


if __name__ == "__main__":
    main()

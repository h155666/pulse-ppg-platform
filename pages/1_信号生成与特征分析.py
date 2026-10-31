# -*- coding: utf-8 -*-
"""
页面 ② 信号生成与特征分析

对应 5-4 指导.md：
  步骤 2.1 最小 demo          步骤 2.2 生成与去噪模块
  步骤 2.3 接参数与绘图        步骤 2.4 指标展示与上传
  步骤 3.3 特征标注方案        步骤 4.1 完善生成与去噪页
  步骤 5.3 实验导出
"""
import numpy as np
import pandas as pd
import streamlit as st

import data_lab as dl
import pulse_core as pc
import ui_utils as ui

ui.page_config("信号生成与特征分析")
st.title("② 信号生成与特征分析")
st.caption("拖动左侧参数 → 页面自动重跑 → 波形、三波标注与指标同步更新")


@ui.guard
def main():
    # ================= 侧边栏 =================
    src = ui.data_source_sidebar()
    st.sidebar.divider()
    gp = ui.generation_sidebar()
    st.sidebar.divider()
    nz = ui.noise_sidebar()
    st.sidebar.divider()

    st.sidebar.header("🧹 去噪设置")
    method = st.sidebar.selectbox("去噪方法", list(pc.DENOISE_METHODS.keys()),
                                  index=1, key="p2_method")
    st.sidebar.caption(pc.DENOISE_METHODS[method]["desc"])
    with st.sidebar.expander("方法参数", expanded=False):
        mp = ui.method_params_widgets(method, key_prefix="p2_")
    compare_all = st.sidebar.checkbox("同时对比全部 8 种方法", value=False,
                                      key="p2_cmp_all")
    st.sidebar.divider()
    fp = ui.feature_sidebar()

    # ================= 取信号 =================
    fs = float(gp["fs"])
    src_kind = src["source"]
    clean = None
    truth = None
    t = None
    note = ""

    if src_kind == "平台自生成（有真值）":
        t, clean, truth = pc.gen_pulse(
            fs, gp["hr"], gp["sec"], amps=gp["amps"], phis=gp["phis"],
            ws=gp["ws"], return_truth=True)
        note = "自适应高斯脉冲串（含真值特征点，可算 RMSE 与命中率）"
    elif src_kind == "上传 CSV" and src["uploaded"] is not None:
        up = ui.read_upload_cached(src["uploaded"].getvalue())
        if "ppg_wave" in up.columns:
            # 数据集格式（48 列，波形存在 ppg_wave 单元格里）：与"内置仿真
            # 数据集"同一条路径 —— 抽一条记录，把单周期模板拼接成连续波。
            rec = st.sidebar.number_input("抽取第几条记录", 1, max(len(up), 1),
                                          1, 1, key="p2_rec_up")
            row = up.iloc[int(rec) - 1]
            w = dl.parse_ppg_wave(row.get("ppg_wave"))
            hr = pc.to_number(row.get("heart_rate"))
            hr = hr if np.isfinite(hr) else 75.0
            fs = min(fs, 250.0)
            t, clean = pc.tile_cycle(w, hr, fs, sec=float(gp["sec"]))
            clean = clean - np.mean(clean)
            note = (f"上传数据集第 {rec} 条（{row.get('subject_type', '-')} / "
                    f"{row.get('sim_model', '-')}），单周期模板拼接，无真值")
        else:
            clean, t, fs = _from_frame(up, fs)
            note = "上传数据（无真值，仅输出去噪后形态与特征）"
    elif src_kind == "内置仿真数据集" and src["path"]:
        rec = st.sidebar.number_input("抽取第几条记录", 1, 17835, 1, 1,
                                      key="p2_rec")
        raw = dl.load_raw(src["path"])
        row = raw.iloc[int(rec) - 1]
        w = dl.parse_ppg_wave(row.get("ppg_wave"))
        hr = pc.to_number(row.get("heart_rate"))
        hr = hr if np.isfinite(hr) else 75.0
        t, clean = pc.tile_cycle(w, hr, min(fs, 250.0), sec=float(gp["sec"]))
        clean = clean - np.mean(clean)
        note = f"来自数据集第 {rec} 条（{row.get('subject_type','-')} / " \
               f"{row.get('sim_model','-')}），单周期模板拼接，无真值"
    else:
        st.info("请在左侧选择数据源。推荐先使用「平台自生成（有真值）」。")
        st.stop()

    if clean is None or len(clean) < 16:
        st.error("信号长度不足，无法分析。请加长仿真时长或换一条记录。")
        st.stop()

    # ================= 加噪 + 去噪 =================
    rng = np.random.default_rng(nz["seed"])
    noisy = pc.add_noise(clean, nz["snr_in"], rng, fs=fs,
                         noise_type=nz["noise_type"],
                         drift_amp=nz["drift_amp"],
                         mains_amp=nz["mains_amp"],
                         motion_amp=nz["motion_amp"])
    denoised = pc.denoise(noisy, method, fs=fs, **mp)

    # ================= 指标 =================
    has_truth = truth is not None
    snr_v = pc.snr_db(clean, denoised)
    d_snr = pc.snr_gain(clean, noisy, denoised)
    rmse_v = pc.rmse(clean, denoised)
    feat = pc.detect_features(denoised, fs, hr=gp["hr"],
                              prom_sys=fp["prom_sys"])
    if has_truth:
        hit = pc.hit_rate(feat["main_idx"], truth["main"], fp["tol_ms"], fs)
        n_true = len(truth["main"])
    else:
        hit, n_true = np.nan, 0
    hr_est = _hr_from_peaks(feat["main_idx"], fs)

    st.subheader("指标卡")
    ui.metric_row([
        ("SNR (dB)", f"{snr_v:.2f}", "10·log10(信号能量 / 残差能量)，越大越好"),
        ("ΔSNR (dB)", f"{d_snr:+.2f}", "去噪增益 = SNR(去噪后) − SNR(加噪后)"),
        ("RMSE", f"{rmse_v:.4f}", "与干净真值的均方根误差，越小越好"),
        ("主波命中率", f"{hit*100:.1f} %" if has_truth else "—（无真值）",
         f"容差 ±{fp['tol_ms']:.0f} ms"),
        ("检出主波数", f"{len(feat['main_idx'])}",
         f"真值 {n_true} 个" if has_truth else "按峰距与突出度自动检出"),
        ("心率估计 (bpm)", f"{hr_est:.1f}" if np.isfinite(hr_est) else "—",
         f"设定值 {gp['hr']} bpm"),
    ], ncol=6)
    st.caption(f"信号来源：{note}　｜　噪声类型：{nz['noise_type']}　｜　"
               f"去噪方法：{method}")

    st.divider()

    # ================= 波形区 =================
    st.subheader("波形")
    st.pyplot(ui.plot_overview(t[:len(clean)], clean, noisy, denoised,
                               f"去噪前后对比（{method}）"))
    zoom_max = float(t[-1]) if len(t) else 1.0
    z0, z1 = st.slider("局部放大区间 (s)", 0.0, max(zoom_max, 1.0),
                       (0.0, min(3.0, max(zoom_max, 1.0))), key="p2_zoom")
    st.pyplot(ui.plot_zoom(t[:len(clean)], clean, noisy, denoised, z0, z1))

    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(ui.plot_spectrum(t, clean, noisy, denoised, fs))
    with c2:
        st.pyplot(ui.plot_residual(t[:len(clean)], clean, denoised))

    st.divider()

    # ================= 特征区 =================
    st.subheader("单周期形态与特征点")
    fc1, fc2 = st.columns([1, 1])
    with fc1:
        t_ms, cyc = pc.average_cycle(denoised, fs, hr=gp["hr"])
        if len(cyc) >= 8:
            row0 = _first_cycle_row(feat, t_ms, cyc)
            st.pyplot(ui.plot_single_cycle(t_ms, cyc, row0,
                                           "单周期形态标注（去噪后平均周期）"))
        else:
            st.warning("有效周期不足 2 个，无法生成平均单周期图。")
    with fc2:
        mor = pc.morphology_features(denoised, fs, hr=gp["hr"],
                                     height_cm=170.0)
        mor_df = pd.DataFrame({
            "特征": ["脉宽 PW (ms)", "上升时间 RT (ms)", "主波幅值",
                     "增强指数 AIx (%)", "反射指数 RI (%)", "硬化指数 SI (m/s)",
                     "灌注指数 PI (%)", "K 值", "dp/dt_max (1/s)",
                     "重搏切迹时刻 (ms)", "重搏波时刻 (ms)"],
            "取值": [mor["pulse_width_ms"], mor["rise_time_ms"],
                     mor["sys_amplitude"], mor["augmentation_index"],
                     mor["reflection_index"], mor["stiffness_index"],
                     mor["perfusion_index"], mor["k_value"], mor["dpdt_max"],
                     mor["notch_ms"], mor["dicrotic_ms"]],
        })
        mor_df["取值"] = mor_df["取值"].astype(float).round(3)
        st.dataframe(mor_df, width="stretch", hide_index=True)
        st.caption("口径：PW = 半高宽；RT = 10%→90%；"
                   "AIx = (A_sys−A_dic)/(A_sys−A_notch)×100%；"
                   "SI = 身高/PTT（身高固定 1.70 m 演示）")

    st.subheader("特征点明细（按周期）")
    fdf = pc.features_to_frame(feat)
    if not fdf.empty:
        st.dataframe(fdf.round(2), width="stretch", hide_index=True)
    else:
        st.warning("未检出特征点，请降低「主波突出度系数」或改善去噪效果。")

    st.divider()

    # ================= 八法速览 =================
    if compare_all:
        st.subheader("八种去噪方法速览")
        rows = []
        for m in pc.DENOISE_METHODS:
            y = pc.denoise(noisy, m, fs=fs)
            f2 = pc.detect_features(y, fs, hr=gp["hr"],
                                    prom_sys=fp["prom_sys"])
            det2 = f2["main_idx"]
            det2 = det2[np.isfinite(det2)]
            rows.append({
                "方法": m,
                "SNR(dB)": round(pc.snr_db(clean, y), 2),
                "ΔSNR(dB)": round(pc.snr_gain(clean, noisy, y), 2),
                "RMSE": round(pc.rmse(clean, y), 4),
                "主波命中率": (round(pc.hit_rate(det2, truth["main"],
                                               fp["tol_ms"], fs), 3)
                           if has_truth else np.nan),
                "检出主波数": int(len(det2)),
            })
        cmp_df = pd.DataFrame(rows).sort_values("ΔSNR(dB)", ascending=False)
        st.dataframe(cmp_df, width="stretch", hide_index=True)

        fig, ax = _quick_bars(cmp_df)
        st.pyplot(fig)

    st.divider()

    # ================= 导出 =================
    st.subheader("导出")
    e1, e2, e3 = st.columns(3)
    out = pd.DataFrame({
        "时间(s)": t[:len(clean)], "干净真值": clean,
        "加噪": noisy, f"去噪_{method}": denoised,
    })
    with e1:
        ui.download_df(out, "pulse_waveform.csv", "⬇ 波形数据 CSV")
    with e2:
        if not fdf.empty:
            ui.download_df(fdf, "pulse_features.csv", "⬇ 特征点明细 CSV")
    with e3:
        ui.download_df(mor_df, "pulse_morphology.csv", "⬇ 形态学特征 CSV")
    st.caption("随机种子固定为 %d，导出结果与页面显示完全一致，可直接用于论文复现。" % nz["seed"])


# --------------------------------------------------------------------------
def _from_frame(raw: pd.DataFrame, fs_default: float):
    """从上传的 DataFrame 里取出一维信号。"""
    if raw.shape[1] >= 2 and np.issubdtype(raw.dtypes.iloc[0], np.number):
        tcol = raw.columns[0]
        t = pd.to_numeric(raw[tcol], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(raw[raw.columns[1]], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t) & np.isfinite(v)
        t, v = t[ok], v[ok]
        if len(t) > 2 and t[-1] > t[0]:
            fs = (len(t) - 1) / (t[-1] - t[0])
        else:
            fs = fs_default
        return v - np.nanmean(v), t, float(np.clip(fs, 20, 4000))
    col = raw.select_dtypes(include="number").columns
    if len(col) == 0:
        return None, None, fs_default
    v = pd.to_numeric(raw[col[0]], errors="coerce").dropna().to_numpy(dtype=float)
    t = np.arange(len(v)) / fs_default
    return v - np.mean(v), t, fs_default


def _hr_from_peaks(idx: np.ndarray, fs: float) -> float:
    idx = np.asarray(idx, dtype=float)
    idx = idx[np.isfinite(idx)]
    if len(idx) < 2:
        return np.nan
    d = np.diff(idx) / float(fs)
    d = d[d > 0]
    if len(d) == 0:
        return np.nan
    return 60.0 / float(np.median(d))


def _first_cycle_row(feat: dict, t_ms: np.ndarray, cyc: np.ndarray) -> dict:
    """把第一个周期的特征点换算到"平均周期图"的横坐标上。

    平均周期图已被 roll 到舒张期最低点为 0 点，因此不能直接用绝对时刻；
    正确做法是：以平均周期图中的主波峰为锚点，把各特征点相对主波的
    时间偏移加上去，再对周期长度取模。
    """
    row = {}
    dt = float(t_ms[1] - t_ms[0]) if len(t_ms) > 1 else 1.0
    T_ms = float(t_ms[-1]) + dt
    peak_ms = float(t_ms[int(np.argmax(cyc))])
    mains = feat.get("main_ms")
    main0 = float(mains[0]) if (mains is not None and len(mains)
                                and np.isfinite(mains[0])) else np.nan
    for k in ("main", "tidal", "dicrotic", "notch"):
        arr = feat.get(f"{k}_ms")
        if arr is None or len(arr) == 0 or not np.isfinite(main0):
            row[f"{k}_ms"] = np.nan
            continue
        v = arr[0]
        if not np.isfinite(v):
            row[f"{k}_ms"] = np.nan
        else:
            row[f"{k}_ms"] = (peak_ms + (float(v) - main0)) % T_ms
    return row


def _quick_bars(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=110)
    colors = ["#c0392b" if v > 0 else "#95a5a6" for v in df["ΔSNR(dB)"]]
    ax.bar(df["方法"].astype(str), df["ΔSNR(dB)"], color=colors)
    ax.axhline(0, color="#2c3e50", lw=0.9)
    ax.set_ylabel("ΔSNR (dB)")
    ax.set_title("八种方法的去噪增益对比（基线 = 未去噪）")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y")
    return fig, ax


if __name__ == "__main__":
    main()

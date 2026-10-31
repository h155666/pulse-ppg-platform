# -*- coding: utf-8 -*-
"""
tools/gen_dataset.py —— 用平台自带的仿真内核生成一份新的模拟数据集

产物（默认写入 ../data/）：
  1. sim_ppg_v2_raw.csv    48 列，字段与 raw_p11_ppg_sim_signal.csv 完全同构，
                           可直接喂给平台页面 ④ / ⑤ 做治理与分析
  2. sim_ppg_v2_truth.csv  真值表：随机种子 + 生成参数 + 真值主波位置 +
                           加噪后实测 SNR / 去噪后 SNR / ΔSNR / RMSE / 命中率
                           （raw_p11 没有真值，这张表正是它的补充）
  3. sim_ppg_v2_dirty.csv  （--inject N > 0 时）按指定条数注入可控异常，
                           用于验证 data_lab 治理模块的检出率与修复率

与 raw_p11 的关键差别（也是本数据集的价值所在）：
  · 库内形态学字段（systolic_peak_ms / pulse_width_ms / rise_time_ms /
    pulse_amplitude / augmentation_index / stiffness_index / perfusion_index /
    dpdt_max）全部由同一份 ppg_wave 按 pulse_core 的统一口径算出，
    因此与波形重算值完全自洽，不存在 raw_p11 里"字段与波形不相关"的问题；
  · snr_db / denoise_snr_gain 是真实测量值而非随机数；
  · ppg_wave 是干净的单周期模板（真值模板），不是带噪波形。

用法：
    python tools/gen_dataset.py                 # 默认 3000 条，不注入异常
    python tools/gen_dataset.py -n 5000
    python tools/gen_dataset.py --inject 150     # 额外产出 dirty 版本
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import pulse_core as pc  # noqa: E402

ORIG_CSV = os.path.join(ROOT, "data", "raw_p11_ppg_sim_signal.csv")
OUT_DIR = os.path.join(ROOT, "data")

FS = 250.0          # 采样率 Hz
SEC = 10.0          # 每条信号时长 s
TOL_MS = 50.0       # 主波命中容差 ms

# --------------------------------------------------------------------------
# 1. 受试者画像（生理参数取值范围，参考临床常识而非照抄 raw_p11 的异常分布）
# --------------------------------------------------------------------------
SUBJECT_PROFILES: Dict[str, dict] = {
    "健康成人":   dict(hr=(60, 85),  sbp=(100, 125), dbp=(65, 80),  h=(158, 180), bmi=(18.5, 24.0), age=(20, 45)),
    "高血压患者": dict(hr=(65, 95),  sbp=(140, 180), dbp=(88, 105), h=(158, 178), bmi=(24.0, 32.0), age=(40, 70)),
    "冠心病患者": dict(hr=(55, 85),  sbp=(120, 160), dbp=(70, 95),  h=(160, 178), bmi=(22.0, 29.0), age=(45, 75)),
    "糖尿病患者": dict(hr=(65, 95),  sbp=(115, 145), dbp=(70, 90),  h=(155, 176), bmi=(23.0, 31.0), age=(38, 68)),
    "运动员":     dict(hr=(40, 60),  sbp=(100, 120), dbp=(60, 75),  h=(170, 190), bmi=(19.0, 25.0), age=(18, 30)),
    "老年人群":   dict(hr=(60, 85),  sbp=(130, 165), dbp=(70, 90),  h=(152, 172), bmi=(20.0, 28.0), age=(65, 88)),
    "儿童青少年": dict(hr=(85, 130), sbp=(90, 115),  dbp=(55, 75),  h=(120, 170), bmi=(15.0, 24.0), age=(8, 17)),
    "肥胖人群":   dict(hr=(70, 95),  sbp=(120, 150), dbp=(78, 95),  h=(158, 180), bmi=(30.0, 42.0), age=(25, 58)),
}
SUBJECT_WEIGHTS = np.array([0.12, 0.15, 0.12, 0.11, 0.14, 0.12, 0.15, 0.09])
SUBJECT_WEIGHTS = SUBJECT_WEIGHTS / SUBJECT_WEIGHTS.sum()

# --------------------------------------------------------------------------
# 2. 生成模型 → 三波形态参数（这是"不同仿真模型"在波形上的实际体现）
#    phis 为相对相位、ws 为相对脉宽、amps 为三波幅值
# --------------------------------------------------------------------------
MODEL_SHAPES: Dict[str, dict] = {
    "三阶Windkessel模型": dict(phis=(0.24, 0.45, 0.63), ws=(0.062, 0.060, 0.078),
                          amps=(1.00, 0.36, 0.22), jitter=0.02),
    "双高斯脉冲合成法":   dict(phis=(0.24, 0.46, 0.00), ws=(0.055, 0.058, 0.070),
                          amps=(1.00, 0.32, 0.00), jitter=0.02),
    "传递函数重建法":     dict(phis=(0.22, 0.44, 0.62), ws=(0.045, 0.052, 0.068),
                          amps=(1.00, 0.30, 0.20), jitter=0.015),
    "弹性腔理论模型":     dict(phis=(0.25, 0.46, 0.66), ws=(0.060, 0.062, 0.082),
                          amps=(1.00, 0.42, 0.30), jitter=0.025),
    "实测波形回放加噪":   dict(phis=(0.24, 0.45, 0.64), ws=(0.056, 0.058, 0.074),
                          amps=(1.00, 0.35, 0.23), jitter=0.035),
    "LSTM生成模型":       dict(phis=(0.24, 0.45, 0.64), ws=(0.058, 0.060, 0.076),
                          amps=(1.00, 0.37, 0.25), jitter=0.055),
}
MODEL_WEIGHTS = np.array([0.30, 0.20, 0.14, 0.12, 0.16, 0.08])
MODEL_WEIGHTS = MODEL_WEIGHTS / MODEL_WEIGHTS.sum()

# --------------------------------------------------------------------------
# 3. 噪声与去噪设置
# --------------------------------------------------------------------------
NOISE_TYPES = ["高斯白噪", "基线漂移", "运动伪影", "工频干扰", "复合噪声", "无噪声"]
NOISE_TYPE_WEIGHTS = np.array([0.20, 0.20, 0.19, 0.13, 0.12, 0.16])
NOISE_TYPE_WEIGHTS = NOISE_TYPE_WEIGHTS / NOISE_TYPE_WEIGHTS.sum()

# 噪声等级 → 目标加噪 SNR(dB)
LEVEL_SNR = {"低": 25.0, "中": 15.0, "高": 8.0, "极高": 3.0}
LEVEL_WEIGHTS = np.array([0.26, 0.33, 0.25, 0.16])
LEVEL_WEIGHTS = LEVEL_WEIGHTS / LEVEL_WEIGHTS.sum()

# 噪声等级 → 平台实际支持的 7 种去噪算法（VMD 未实现，故不出现）
# "无"噪声的记录不做去噪处理（SNR 已足够高，再去噪只会引入误差，
#  强行计算 ΔSNR 会得到 −70 dB 这类无工程意义的数值）
ALGO_BY_LEVEL = {
    "无": ["未处理"],
    "低": ["未处理", "Savitzky-Golay滤波", "小波阈值去噪", "0.5-8Hz带通滤波"],
    "中": ["小波阈值去噪", "0.5-8Hz带通滤波", "Savitzky-Golay滤波", "中值滤波"],
    "高": ["小波阈值去噪", "EMD经验模态分解", "0.5-8Hz带通滤波", "自适应陷波滤波"],
    "极高": ["EMD经验模态分解", "小波阈值去噪", "自适应陷波滤波", "中值滤波"],
}
# 「无噪声」记录仍保留 55 dB 的电子噪声底，保证 SNR 有定义（真实采集也不可能无噪）
NOISE_FLOOR_SNR = 55.0
# 数据集里的算法名 → pulse_core 的方法名
ALGO_TO_METHOD = {
    "未处理": "无（基线）",
    "小波阈值去噪": "小波阈值去噪",
    "EMD经验模态分解": "EMD经验模态分解",
    "Savitzky-Golay滤波": "Savitzky-Golay滤波",
    "0.5-8Hz带通滤波": "0.5-8Hz带通滤波",
    "中值滤波": "中值滤波",
    "自适应陷波滤波": "自适应陷波+低通",
}

DEVICE_MODELS = ["PPG-仿真采集卡 SA-200", "仿真引擎虚拟通道 vCH-01",
                 "医用监护仪 PM-9000 PPG通道", "MAX30102 光电模块",
                 "腕带式光电传感器 WB-7", "指夹式脉搏传感器 PX-11"]
EXPORT_FORMATS = ["CSV", "JSON", "TXT", "MAT", "HDF5"]
OPERATOR_ROLES = ["本科生", "研究生", "指导教师", "实验员"]
SIM_STATUS = ["成功"] * 88 + ["部分收敛"] * 6 + ["失败重算"] * 4 + ["超时中断"] * 2
ORG_NAMES = ["广东轻工职业技术大学", "东南大学", "武汉大学", "南昌大学", "华南理工大学"]
SURNAMES = list("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜")
GIVEN = list("伟芳娜秀敏静丽强磊洋艳勇军杰娟涛明超霞平刚桂英")


# --------------------------------------------------------------------------
# 波形工具
# --------------------------------------------------------------------------
def single_cycle_template(clean: np.ndarray, fs: float, hr: float,
                          n_points: int = 64) -> np.ndarray:
    """从干净信号里切出一个完整周期（以 tp=0 为起点），重采样到 n_points 点。

    模板第 0 点对应周期起点（前一心搏的舒张期末），因此主波峰落在
    φ₁·T 处，systolic_peak_ms 才有物理意义。
    """
    T = 60.0 / hr
    n = max(8, int(round(T * fs)))
    start = int(round(T * fs))          # 从第 2 个周期开始取，躲开起始瞬态
    if start + n > len(clean):
        start = max(0, len(clean) - n)
    seg = clean[start:start + n]
    if len(seg) < 8:
        seg = clean[-n:] if len(clean) >= n else clean
    t_old = np.linspace(0.0, 1.0, len(seg))
    t_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(t_new, t_old, seg)


def cycle_peak_ms(template: np.ndarray, hr: float) -> float:
    """单周期模板中主波所在的时刻 (ms)。"""
    if len(template) < 4:
        return np.nan
    T_ms = 60000.0 / hr
    i = int(np.argmax(template))
    return float(i) / (len(template) - 1) * T_ms


# --------------------------------------------------------------------------
# 主生成流程
# --------------------------------------------------------------------------
def generate(n: int, seed: int, verbose: bool = True) -> Tuple[List[str], List[list], List[dict]]:
    rng = np.random.default_rng(seed)

    # 原表头：与 raw_p11 保持完全一致的 48 列
    with open(ORIG_CSV, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    assert len(header) == 48, f"原表头 {len(header)} 列，预期 48"

    rows: List[list] = []
    truth: List[dict] = []
    t_start = time.perf_counter()

    for i in range(n):
        seed_i = int(seed * 100000 + i)
        subj = str(rng.choice(list(SUBJECT_PROFILES.keys()), p=SUBJECT_WEIGHTS))
        prof = SUBJECT_PROFILES[subj]
        model = str(rng.choice(list(MODEL_SHAPES.keys()), p=MODEL_WEIGHTS))
        shape = MODEL_SHAPES[model]

        hr = float(np.round(rng.uniform(*prof["hr"]), 1))
        sbp = float(np.round(rng.uniform(*prof["sbp"]), 1))
        dbp = float(np.round(rng.uniform(*prof["dbp"]), 1))
        height = float(np.round(rng.uniform(*prof["h"]), 1))
        # 由「身高 + 目标 BMI」反推体重：BMI 必然落在生理区间内，
        # 也避免生成"170cm / 24kg"这类身高体重互不相容的样本。
        bmi = float(np.round(rng.uniform(*prof["bmi"]), 1))
        weight = float(np.round(bmi * (height / 100.0) ** 2, 1))
        age = int(rng.uniform(*prof["age"]))

        noise_type = str(rng.choice(NOISE_TYPES, p=NOISE_TYPE_WEIGHTS))
        if noise_type == "无噪声":
            level = "无"
            snr_in = NOISE_FLOOR_SNR          # 名义电子噪声底，保证 SNR 有定义
        else:
            level = str(rng.choice(list(LEVEL_SNR.keys()), p=LEVEL_WEIGHTS))
            # 目标 SNR 加抖动；下钳到 0.5 dB，避免"极高"档偶尔出现负 SNR
            # 违反 snr_db ∈ [0,60] 的工程区间。
            snr_in = float(np.clip(LEVEL_SNR[level] + rng.normal(0, 0.8), 0.5, 60.0))
        algo = str(rng.choice(ALGO_BY_LEVEL[level]))
        method = ALGO_TO_METHOD[algo]

        # ---- 生成（模型差异 + 个体抖动） ----
        j = shape["jitter"]
        amps = tuple(a * (1 + rng.normal(0, j)) for a in shape["amps"])
        phis = tuple(float(np.clip(p * (1 + rng.normal(0, j)), 0.05, 0.95))
                     for p in shape["phis"])
        ws = tuple(float(np.clip(w * (1 + rng.normal(0, j)), 0.02, 0.20))
                   for w in shape["ws"])
        _, clean, tr = pc.gen_pulse(FS, hr, SEC, amps=amps, phis=phis, ws=ws,
                                    return_truth=True)

        # ---- 注入生理直流基线，使灌注指数 PI = AC/DC×100% 落在真实区间 ----
        # 真实指端 PPG 的 DC（组织/光强基线）远大于搏动分量 AC。纯高斯脉冲串
        # 的 baseline=0，其周期均值只有 ~0.24，直接算 PI 会得到 340%~580% 的
        # 虚高值。这里按每条记录随机设定目标 PI（0.8%~6.5%，覆盖真实灌注水平），
        # 反解出应有的 DC 后整体抬升波形。
        # 【为什么安全】下游所有指标都与 DC 无关：add_noise 用方差定噪声功率、
        # snr_db/rmse 为 AC 耦合口径、detect_features 用相对突出度找峰、
        # 形态学特征（除 PI）也都在去均值/归一化后计算。
        pi_target = float(rng.uniform(0.8, 6.5))
        ac = float(clean.max() - clean.min())
        dc_needed = ac / (pi_target / 100.0)
        clean = clean + (dc_needed - float(clean.mean()))

        # ---- 加噪 + 去噪 ----
        noise_rng = np.random.default_rng(seed_i)
        # 「无噪声」记录按电子噪声底加极弱白噪，其余按指定噪声类型叠加
        actual_type = "高斯白噪" if noise_type == "无噪声" else noise_type
        noisy = pc.add_noise(clean, snr_in, noise_rng, fs=FS,
                             noise_type=actual_type)
        den = pc.denoise(noisy, method, fs=FS)

        # ---- 指标（真值） ----
        snr_noisy = pc.snr_db(clean, noisy)
        snr_den = pc.snr_db(clean, den)
        gain = pc.snr_gain(clean, noisy, den)
        rmse_v = pc.rmse(clean, den)
        feat = pc.detect_features(den, FS, hr=hr)
        det = feat["main_idx"][np.isfinite(feat["main_idx"])]
        hit = pc.hit_rate(det, tr["main"], TOL_MS, FS)

        # ---- 单周期真值模板 + 由它统一口径计算库内形态学字段 ----
        tmpl = single_cycle_template(clean, FS, hr, 64)
        mor = pc.morphology_from_cycle(tmpl, hr, height_cm=height)

        dicrotic_notch_ms = (float(mor["notch_ms"]) if np.isfinite(mor["notch_ms"])
                             else np.nan)
        # 重搏切迹的绝对时刻 = 主波时刻 + 主波到切迹的间隔
        sys_ms = cycle_peak_ms(tmpl, hr)
        notch_abs = sys_ms + dicrotic_notch_ms if np.isfinite(dicrotic_notch_ms) else np.nan

        signal_id = f"SIM{2026_0000_0000 + i:013d}"[:16]
        signal_id = f"SIM{seed}{i:07d}"[:16]
        point_count = int(round(FS * SEC))
        patient = f"VS{200000 + i:06d}"
        sn = "SN" + "".join(f"{int(v):02X}" for v in rng.integers(0, 256, 5))
        t_no = "T" + str(int(rng.integers(30000, 99999)))
        op_name = str(rng.choice(SURNAMES)) + str(rng.choice(GIVEN))

        sim_params = json.dumps({
            "model": model, "hr": f"{hr:g}", "sbp": f"{sbp:g}", "dbp": f"{dbp:g}",
            "compliance": f"{rng.uniform(0.4, 1.7):.3f}", "fs": f"{FS:g}",
            "noise": level,
        }, ensure_ascii=False)

        row = [
            i + 1,                                  # id
            signal_id,                              # signal_id
            patient,                                # subject_id
            subj,                                   # subject_type
            "男" if rng.random() < 0.49 else "女",   # gender
            age, height, weight, bmi,               # age/height/weight/bmi
            hr, sbp, dbp,                           # heart_rate/sbp/dbp
            round(float(hr * rng.uniform(0.055, 0.085)), 2),   # cardiac_output
            round(float(rng.uniform(0.35, 1.70)), 3),          # vessel_compliance
            model,                                  # sim_model
            int(FS),                                # sample_rate
            int(SEC),                               # duration_sec
            point_count,                            # point_count
            ",".join(f"{v:.4f}" for v in tmpl),      # ppg_wave（64 点干净模板）
            round(float(mor["dpdt_max"]), 3) if np.isfinite(mor["dpdt_max"]) else "",
            round(sys_ms, 1),                        # systolic_peak_ms
            round(notch_abs, 1) if np.isfinite(notch_abs) else "",
            round(float(mor["pulse_width_ms"]), 1) if np.isfinite(mor["pulse_width_ms"]) else "",
            round(float(mor["rise_time_ms"]), 1) if np.isfinite(mor["rise_time_ms"]) else "",
            round(float(mor["sys_amplitude"]), 4),   # pulse_amplitude
            round(float(mor["augmentation_index"]), 2) if np.isfinite(mor["augmentation_index"]) else "",
            round(float(mor["stiffness_index"]), 3) if np.isfinite(mor["stiffness_index"]) else "",
            round(float(mor["perfusion_index"]), 3) if np.isfinite(mor["perfusion_index"]) else "",
            round(snr_noisy, 2),                     # snr_db（加噪后实测）
            noise_type,                              # noise_type
            level,                                   # noise_level
            algo,                                    # denoise_algo
            round(gain, 2),                          # denoise_snr_gain（实测 ΔSNR）
            sim_params,
            "PPG-SimEngine v4.0.0",
            str(rng.choice(DEVICE_MODELS)),
            sn, t_no, op_name, str(rng.choice(OPERATOR_ROLES)),
            str(rng.choice(ORG_NAMES)),
            f"信号仿真课题组{i % 30 + 1}班",
            str(rng.choice(EXPORT_FORMATS)),
            int(rng.integers(5, 60)),
            str(rng.choice(SIM_STATUS)),
            "本记录由平台仿真内核生成，库内形态学字段与 ppg_wave 同一口径，可作为回归基准。",
            f"2026-10-16 {8 + i % 12:02d}:{i % 60:02d}:{i % 60:02d}",
            f"2026-10-16 {9 + i % 12:02d}:{i % 60:02d}:{i % 60:02d}",
        ]
        assert len(row) == 48, f"第 {i} 行列数 {len(row)} != 48"
        rows.append(row)

        truth.append({
            "id": i + 1, "signal_id": signal_id, "subject_type": subj,
            "sim_model": model, "seed": seed_i, "fs": FS, "hr": hr, "sec": SEC,
            "amps": json.dumps(amps), "phis": json.dumps(phis),
            "ws": json.dumps(ws), "noise_type": noise_type,
            "noise_level": level, "snr_in_target": round(snr_in, 2),
            "denoise_algo": algo,
            "真值主波数": len(tr["main"]),
            "真值主波索引": json.dumps([int(v) for v in tr["main"]]),
            "加噪后SNR(dB)": round(snr_noisy, 3),
            "去噪后SNR(dB)": round(snr_den, 3),
            "ΔSNR(dB)": round(gain, 3),
            "RMSE": round(rmse_v, 5),
            "主波命中率": round(float(hit), 4),
            "检出主波数": int(len(det)),
        })

        if verbose and (i + 1) % max(1, n // 10) == 0:
            print(f"  已生成 {i+1}/{n}  最近一条：{subj}/{model}/{noise_type}/{level}/{algo}")

    if verbose:
        print(f"生成完成：{n} 条，耗时 {time.perf_counter()-t_start:.1f}s")
    return header, rows, truth


def inject_anomalies(header: List[str], rows: List[list], n_inject: int,
                     seed: int = 7) -> Tuple[List[list], List[dict]]:
    """向数据副本注入可控异常，并返回异常清单（用于验证治理模块）。"""
    rng = np.random.default_rng(seed)
    out = [list(r) for r in rows]
    idx = {name: k for k, name in enumerate(header)}
    n = len(out)
    log: List[dict] = []

    def pick(k, exclude=None):
        cand = rng.choice(n, size=k * 6, replace=True)
        res = []
        for c in cand:
            c = int(c)
            if c in res or (exclude and c in exclude):
                continue
            res.append(c)
            if len(res) == k:
                break
        return res

    specs = [
        ("point_count", -1000.0, "负数"),
        ("point_count", 1073741824.0, "2^30 哨兵值"),
        ("dbp_mmhg", -8.0, "负值"),
        ("augmentation_index", -1267.77, "负值"),
        ("augmentation_index", 99786.25, "极大离群"),
        ("snr_db", -99.0, "负值"),
        ("snr_db", 999.99, "哨兵值"),
        ("cardiac_output", 99649.61, "极大离群"),
        ("pulse_amplitude", 99031.35, "极大离群"),
        ("dpdt_max", 99479.63, "极大离群"),
        ("age", 160.0, "超龄"),
        ("height_cm", 264.9, "超身高"),
        ("weight_kg", 319.9, "超体重"),
    ]
    for item in ("gender", "sim_status", "noise_level", "export_format"):
        specs.append((item, None, "口径混写"))

    dirty_variants = {
        "gender": ["Male", "f", "1", "先生", "FEMALE", "m"],
        "sim_status": ["success", "OK", "重算", "timeout"],
        "noise_level": ["LOW", "mid", "1", "轻度", "极高"],
        "export_format": ["csv", ".CSV", "csv;utf8", "excel", "dat"],
    }

    used: set = set()
    per = max(1, n_inject // len(specs))
    for field, value, kind in specs:
        for c in pick(per, exclude=used):
            used.add(c)
            if field in dirty_variants:
                out[c][idx[field]] = str(rng.choice(dirty_variants[field]))
            else:
                out[c][idx[field]] = value
            log.append({"行号": c + 1, "字段": field, "注入类型": kind,
                        "注入值": out[c][idx[field]]})
    return out, log


def read_desc_row() -> List[str]:
    """读取 raw_p11 的第 2 行（字段说明行）。新数据集沿用同一份字段说明，
    使两者结构完全同构 —— data_lab.load_raw 依赖 `skiprows=[1]` 剔除这一行，
    若新文件缺少该行，会把第 1 条真实记录误删（记录数少 1）。"""
    with open(ORIG_CSV, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r)
        return next(r)


def write_csv(path: str, header: List[str], rows: List[list],
              desc: Optional[List[str]] = None):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        if desc is not None:
            w.writerow(desc)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=3000, help="生成条数")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--inject", type=int, default=0, help="额外注入的异常条数")
    ap.add_argument("--outdir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"目标：{args.num} 条，采样率 {FS} Hz，时长 {SEC} s，随机种子 {args.seed}")

    header, rows, truth = generate(args.num, args.seed)
    desc = read_desc_row()

    raw_path = os.path.join(args.outdir, "sim_ppg_v2_raw.csv")
    write_csv(raw_path, header, rows, desc=desc)
    print(f"✔ 主数据集  {raw_path}  ({os.path.getsize(raw_path)/1024/1024:.2f} MB, {len(rows)} 行 × {len(header)} 列)")

    import pandas as pd
    tp = os.path.join(args.outdir, "sim_ppg_v2_truth.csv")
    pd.DataFrame(truth).to_csv(tp, index=False, encoding="utf-8-sig")
    print(f"✔ 真值表    {tp}  ({os.path.getsize(tp)/1024:.0f} KB, {len(truth)} 行 × {len(truth[0])} 列)")

    if args.inject > 0:
        dirty, log = inject_anomalies(header, rows, args.inject)
        dp = os.path.join(args.outdir, "sim_ppg_v2_dirty.csv")
        write_csv(dp, header, dirty, desc=desc)
        lp = os.path.join(args.outdir, "sim_ppg_v2_injected_anomalies.csv")
        pd.DataFrame(log).to_csv(lp, index=False, encoding="utf-8-sig")
        print(f"✔ 异常版本  {dp}  （注入 {len(log)} 处）")
        print(f"✔ 异常清单  {lp}")

    # 简单自检（注意 skiprows=[1] 剔除字段说明行，与 data_lab.load_raw 口径一致）
    import pandas as pd
    df = pd.read_csv(raw_path, encoding="utf-8-sig", skiprows=[1], low_memory=False)
    print("\n--- 自检 ---")
    print(f"形状            : {df.shape}")
    print(f"受试者类型分布  : {dict(df['subject_type'].value_counts())}")
    print(f"生成模型分布    : {dict(df['sim_model'].value_counts())}")
    print(f"缺失单元格总数  : {int(df.isna().sum().sum())}")
    for c in ("perfusion_index", "pulse_width_ms", "bmi", "snr_db",
              "denoise_snr_gain", "augmentation_index"):
        print(f"  {c:18s}: [{df[c].min():.3f}, {df[c].max():.3f}]  中位 {df[c].median():.3f}")
    print(f"点数字段唯一值  : {sorted(df['point_count'].unique())}")
    print(f"ppg_wave 点数   : {sorted(df['ppg_wave'].str.count(',').add(1).unique())}")
    print(f"命中率均值(真值表): {pd.DataFrame(truth)['主波命中率'].mean():.3f}")
    print(f"平均ΔSNR(真值表) : {pd.DataFrame(truth)['ΔSNR(dB)'].mean():.3f}")


if __name__ == "__main__":
    main()

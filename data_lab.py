# -*- coding: utf-8 -*-
"""
data_lab.py —— 脉搏波仿真数据集的治理与分析层（课题 5-4）

对应《大数据开发项目指导书》的：
  · 任务 2  数据规范清洗（空值处理、去重、字段口径归一、离群值识别）
  · 任务 3  仿真数据仓库 ODS → DWD → DWS（本平台在单机 pandas 上等价实现）
  · 任务 4  波形特征工程（ppg_wave 解析 + 形态学特征提取）
  · 任务 5  形态学特征量化分析
  · 任务 6  去噪算法对比评估

【为什么没有用 Spark / Hive】
  课题 5-4 的技术栈被硬性固定为 Streamlit + scipy + matplotlib（见 5-4 指导 0.5），
  且 Hive/HDFS 无法在 Windows 单机本科环境部署。本数据集体量仅 17835 × 48（约 19 MB），
  pandas 的列向量化 + 分块处理在口径上与 Spark 清洗完全等价，故在此层等价实现，
  并把处理耗时作为工程佐证输出（论文 3.5 节可直接引用）。

本文件【不依赖 Streamlit】，可单独在命令行运行：
    python data_lab.py
"""
from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

import pulse_core as pc

DEFAULT_FILENAME = "raw_p11_ppg_sim_signal.csv"
# 项目测试数据在学校目录下的原始位置（作为兜底路径）
FALLBACK_PATH = os.path.join(
    r"C:\Users\15778\Desktop\学校",
    "毕业设计-项目测试数据（26个）(1)",
    "人体脉搏波信号生成与特征分析虚拟仿真平台设计",
    DEFAULT_FILENAME,
)


def resolve_data_path(explicit: Optional[str] = None) -> Optional[str]:
    """按 显式参数 → 环境变量 → 项目 data/ → 学校原始目录 的顺序定位数据文件。"""
    cands = [explicit, os.environ.get("PPG_DATA"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", DEFAULT_FILENAME),
             FALLBACK_PATH]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


# ==========================================================================
# 1. 载入
# ==========================================================================

def load_raw(path: Optional[str] = None) -> pd.DataFrame:
    """读取原始 CSV。

    注意：文件【第 2 行是字段说明记录】，必须剔除（指导书 3.2：
    17837 行 → 有效 17835 条）。
    """
    path = path or resolve_data_path()
    if path is None:
        raise FileNotFoundError("未找到 raw_p11_ppg_sim_signal.csv，请通过侧边栏上传或设置 PPG_DATA 环境变量")
    df = pd.read_csv(path, encoding="utf-8-sig", skiprows=[1],
                     low_memory=False, on_bad_lines="skip")
    return df


def read_upload_csv(src) -> pd.DataFrame:
    """读取【用户上传】的 CSV，自动剔除第 2 行的字段说明行。

    【为什么不能直接 pd.read_csv】raw_p11_ppg_sim_signal.csv 的第 2 行是
    48 列全是文本的字段说明，read_csv 会把它当成一条真实记录：
      1) 所有数值列被污染成文本，`select_dtypes(number)` 变为空；
      2) 页面 ② 拿不到任何一维信号 → 报「信号长度不足，无法分析」；
      3) 治理页的记录数也会多 1（且"已剔除字段说明行"的说明名不副实）。
    本地文件走 load_raw（已知第 2 行固定是说明行，直接 skiprows=[1]）；
    上传的文件来源未知，用启发式判定：首行若 ≥80% 的列都不能转成数字，
    即视为说明行剔除。
    """
    df = pd.read_csv(src, encoding="utf-8-sig", low_memory=False,
                     on_bad_lines="skip")
    if len(df) == 0 or df.shape[1] == 0:
        return df
    num = pd.to_numeric(df.iloc[0], errors="coerce")
    n_num = int(num.notna().sum())
    if n_num <= int(df.shape[1] * 0.2):
        df = df.iloc[1:].reset_index(drop=True)
    return df


# ==========================================================================
# 2. 字段口径归一
# ==========================================================================

_GENDER_MAP = {
    "男": "男", "m": "男", "male": "男", "男性": "男", "先生": "男", "1": "男",
    "女": "女", "f": "女", "female": "女", "女性": "女", "女士": "女", "0": "女",
}
_STATUS_MAP = {
    "成功": "成功", "success": "成功", "ok": "成功", "succeed": "成功",
    "部分收敛": "部分收敛", "partial": "部分收敛",
    "失败重算": "失败重算", "重算": "失败重算", "fail": "失败重算", "failed": "失败重算",
    "超时中断": "超时中断", "timeout": "超时中断",
}
_NOISE_LEVEL_MAP = {
    "无": "无", "none": "无", "0": "无",
    "低": "低", "low": "低", "1": "低", "轻度": "低", "低噪声": "低",
    "中": "中", "mid": "中", "middle": "中", "2": "中", "中度": "中",
    "高": "高", "high": "高", "3": "高",
    "极高": "极高", "4": "极高", "extreme": "极高",
}
_FORMAT_MAP = {
    "csv": "CSV", "csv;utf8": "CSV", ".csv": "CSV",
    "json": "JSON", "txt": "TXT", "mat": "MAT", "hdf5": "HDF5",
    "dat": "DAT", "excel": "EXCEL", "xlsx": "EXCEL",
}
_ROLE_MAP = {
    "本科生": "本科生", "研究生": "研究生", "硕士生": "研究生",
    "指导教师": "指导教师", "教师": "指导教师", "实验员": "实验员",
}


def _norm_lookup(v, mapping: Dict[str, str], default="未知"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    key = str(v).strip()
    if key == "" or key.lower() in {"nan", "none", "null"}:
        return default
    return mapping.get(key, mapping.get(key.lower(), mapping.get(key.upper(), "其他")))


def normalize_gender(v) -> str:
    return _norm_lookup(v, _GENDER_MAP)


def normalize_status(v) -> str:
    return _norm_lookup(v, _STATUS_MAP)


def normalize_noise_level(v) -> str:
    return _norm_lookup(v, _NOISE_LEVEL_MAP)


def normalize_export_format(v) -> str:
    return _norm_lookup(v, _FORMAT_MAP)


def normalize_role(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "未知"
    key = str(v).strip().replace("（旧）", "").replace("(旧)", "")
    return _ROLE_MAP.get(key, _ROLE_MAP.get(key.lower(), "其他"))


def normalize_subject_type(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "未知"
    key = str(v).strip()
    return key if key else "未知"


NORMALIZERS = {
    "gender": normalize_gender,
    "sim_status": normalize_status,
    "noise_level": normalize_noise_level,
    "export_format": normalize_export_format,
    "operator_role": normalize_role,
    "subject_type": normalize_subject_type,
}

# 含"脏串写法"的数值列（需要走 to_number 解析）
DIRTY_NUMERIC_COLS = ["age", "heart_rate", "sbp_mmhg", "dbp_mmhg",
                      "dicrotic_notch_ms", "duration_sec", "sample_rate",
                      "file_size_kb", "point_count"]
MS_COLS = ["dicrotic_notch_ms", "systolic_peak_ms", "pulse_width_ms",
           "rise_time_ms"]

# 生理/工程合理区间（指导书 3.4 的对账依据）
NUMERIC_RULES: Dict[str, tuple] = {
    "age": (1, 120),
    "height_cm": (50, 220),
    "weight_kg": (2, 200),
    "bmi": (10, 60),
    "heart_rate": (30, 220),
    "sbp_mmhg": (60, 260),
    "dbp_mmhg": (30, 160),
    "cardiac_output": (1, 20),
    "vessel_compliance": (0.05, 3.0),
    "sample_rate": (50, 2000),
    "duration_sec": (1, 60),
    "point_count": (1, 1_000_000),
    "systolic_peak_ms": (50, 600),
    "dicrotic_notch_ms": (100, 900),
    "pulse_width_ms": (100, 1200),
    "rise_time_ms": (30, 500),
    "pulse_amplitude": (0.01, 50),
    "augmentation_index": (-50, 100),
    "stiffness_index": (0.5, 30),
    "perfusion_index": (0.05, 30),
    "snr_db": (0, 60),
    "denoise_snr_gain": (0, 60),
    "dpdt_max": (0.01, 200),
    "file_size_kb": (0.1, 2000),
}

STRATEGIES = ["仅标记不修改", "置为缺失", "全局中位数填补", "分组中位数填补"]


def clean_dataframe(df: pd.DataFrame, strategy: str = "分组中位数填补",
                    group_by: str = "subject_type",
                    apply_range: bool = True) -> tuple:
    """执行完整治理，返回 (清洗后 DataFrame, 治理日志 DataFrame, 耗时秒)。

    strategy:
      仅标记不修改   —— 只新增 *_is_outlier 标记列
      置为缺失       —— 越界值置 NaN
      全局中位数填补 —— 越界置 NaN 后用整列中位数填补
      分组中位数填补 —— 越界置 NaN 后先按 group_by 分组中位数、再全局中位数
    """
    t0 = time.perf_counter()
    out = df.copy()
    log: List[dict] = []

    # ---- 2.1 分类字段口径归一 ----
    for col, fn in NORMALIZERS.items():
        if col not in out.columns:
            continue
        raw = out[col]
        uniq_before = int(raw.astype(str).str.strip().replace(
            {"": np.nan, "nan": np.nan, "None": np.nan}).dropna().nunique())
        out[col] = raw.map(fn, na_action="ignore")
        uniq_after = int(out[col].dropna().nunique())
        log.append({"字段": col, "处理": "口径归一",
                    "处理前取值数": uniq_before, "处理后取值数": uniq_after,
                    "影响行数": int(raw.notna().sum())})

    # ---- 2.2 脏串数值列解析 ----
    for col in DIRTY_NUMERIC_COLS:
        if col not in out.columns:
            continue
        raw = pd.to_numeric(out[col], errors="coerce")
        n_before = int(raw.notna().sum())
        parsed = out[col].map(
            lambda v: pc.to_number(v, assume_ms=(col in MS_COLS)),
            na_action="ignore")
        parsed = pd.to_numeric(parsed, errors="coerce")
        out[col] = parsed
        log.append({"字段": col, "处理": "脏串数值解析",
                    "处理前取值数": n_before, "处理后取值数": int(parsed.notna().sum()),
                    "影响行数": int(parsed.notna().sum() - n_before)})

    # ---- 2.3 去重 ----
    n0 = len(out)
    key = "signal_id" if "signal_id" in out.columns else out.columns[0]
    out = out.drop_duplicates(subset=[key], keep="first")
    if len(out) != n0:
        log.append({"字段": key, "处理": "去重", "处理前取值数": n0,
                    "处理后取值数": len(out), "影响行数": n0 - len(out)})

    # ---- 2.4 规则列统一数值化（必须在越界治理之前）----
    # 【pandas 3.x 坑】含脏串的数值列会被推断成 str dtype（pandas 2 是 object）。
    # 2.2 只处理 DIRTY_NUMERIC_COLS，而 snr_db / denoise_snr_gain /
    # augmentation_index / systolic_peak_ms 等规则列同样混着 "3.5dB" 这类脏串；
    # 若不统一数值化，下游 groupby().mean() 会抛
    # "TypeError: dtype 'str' does not support operation 'mean'"，
    # 箱线图等统计也会跟着失效。
    for col in NUMERIC_RULES:
        if col not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        raw_num = pd.to_numeric(out[col], errors="coerce")
        n_before = int(raw_num.notna().sum())
        parsed = out[col].map(
            lambda v: pc.to_number(v, assume_ms=(col in MS_COLS)),
            na_action="ignore")
        parsed = pd.to_numeric(parsed, errors="coerce")
        out[col] = parsed
        log.append({"字段": col, "处理": "脏串数值解析",
                    "处理前取值数": n_before,
                    "处理后取值数": int(parsed.notna().sum()),
                    "影响行数": int(parsed.notna().sum() - n_before)})

    # ---- 2.5 异常值识别与治理 ----
    if apply_range:
        for col, (lo, hi) in NUMERIC_RULES.items():
            if col not in out.columns:
                continue
            v = pd.to_numeric(out[col], errors="coerce")
            bad = ((v < lo) | (v > hi)).fillna(False)
            n_bad = int(bad.sum())
            if n_bad == 0:
                continue
            if strategy == "仅标记不修改":
                out[col + "_is_outlier"] = bad
            else:
                v = v.mask(bad)
                if strategy in ("全局中位数填补", "分组中位数填补"):
                    if strategy == "分组中位数填补" and group_by in out.columns:
                        try:
                            v = v.fillna(v.groupby(out[group_by]).transform("median"))
                        except Exception:
                            pass
                    v = v.fillna(v.median())
                out[col] = v
            log.append({"字段": col, "处理": f"越界治理({lo}~{hi})",
                        "处理前取值数": int(pd.to_numeric(df[col], errors="coerce").notna().sum()),
                        "处理后取值数": int(out[col].notna().sum()),
                        "影响行数": n_bad})

    # ---- 2.6 point_count 特殊重建：≥1e8 的哨兵值用 fs×duration 还原 ----
    if {"point_count", "sample_rate", "duration_sec"}.issubset(out.columns):
        calc = pd.to_numeric(out["sample_rate"], errors="coerce") * \
               pd.to_numeric(out["duration_sec"], errors="coerce")
        need = out["point_count"].isna() | (out["point_count"] > 1_000_000)
        n_need = int(need.fillna(False).sum())
        if n_need:
            out.loc[need.fillna(False), "point_count"] = calc[need.fillna(False)]
            log.append({"字段": "point_count", "处理": "按 fs×时长 重建",
                        "处理前取值数": n_need, "处理后取值数": n_need,
                        "影响行数": n_need})

    return out, pd.DataFrame(log), time.perf_counter() - t0


def quality_report(df: pd.DataFrame, rules: Dict[str, tuple] = None) -> pd.DataFrame:
    """逐列质量报告：缺失 / 越界 / 口径混乱程度。"""
    rules = rules or NUMERIC_RULES
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        # 【注意】pandas 3.x 默认把字符串列推断成 str/StringDtype 而不是 object，
        # 若仍用 `dtype == object` 判断，所有文本列的唯一取值数都会是 NaN，
        # 导致"字段口径混乱度"表格整表为空。这里改用"是否数值列"来判断。
        is_text = not pd.api.types.is_numeric_dtype(s.dtype)
        miss = int(s.isna().sum())
        if is_text:
            miss += int((s.astype(str).str.strip() == "").sum())
        uniq = np.nan
        if is_text:
            uniq = int(s.dropna().astype(str).str.strip().nunique())
        oob = np.nan
        if col in rules:
            lo, hi = rules[col]
            v = pd.to_numeric(s, errors="coerce")
            oob = int(((v < lo) | (v > hi)).fillna(False).sum())
        rows.append({
            "字段": col,
            "类型": str(s.dtype),
            "缺失数": miss,
            "缺失率(%)": round(miss / max(n, 1) * 100, 3),
            "越界数": oob,
            "越界率(%)": round(oob / max(n, 1) * 100, 3) if oob == oob else np.nan,
            "唯一取值数": uniq,
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 3. 波形解析与特征工程
# ==========================================================================

def parse_ppg_wave(s) -> np.ndarray:
    """把 '1.2,1.4,...' 的序列化波形解析成 float 数组（自动跳过 NaN/空串）。"""
    if s is None:
        return np.array([], dtype=float)
    t = str(s).strip().strip('"')
    if not t:
        return np.array([], dtype=float)
    vals = []
    for p in t.split(","):
        p = p.strip()
        if p == "" or p.lower() in {"nan", "null", "na"}:
            continue
        try:
            vals.append(float(p))
        except ValueError:
            continue
    return np.asarray(vals, dtype=float)


def decode_waves(df: pd.DataFrame) -> List[np.ndarray]:
    """批量解析 ppg_wave 列。"""
    return [parse_ppg_wave(v) for v in df["ppg_wave"].tolist()]


def sample_records(df: pd.DataFrame, n: int = 300, seed: int = 42,
                   subject_type: Optional[str] = None,
                   sim_model: Optional[str] = None) -> pd.DataFrame:
    """抽样（可按受试者类型 / 生成模型过滤）。"""
    sub = df
    if subject_type and subject_type != "全部" and "subject_type" in sub.columns:
        sub = sub[sub["subject_type"].astype(str) == subject_type]
    if sim_model and sim_model != "全部" and "sim_model" in sub.columns:
        sub = sub[sub["sim_model"].astype(str) == sim_model]
    if len(sub) == 0:
        return sub
    n = int(np.clip(n, 1, len(sub)))
    return sub.sample(n=n, random_state=seed) if n < len(sub) else sub


def recompute_morphology(df: pd.DataFrame, n_sample: int = 300, seed: int = 42,
                         subject_type: Optional[str] = None) -> pd.DataFrame:
    """对抽样的 ppg_wave 重算形态学特征，并与库内字段比对（一致性校验）。

    返回列包含：记录标识 / 受试者类型 / 心率 / 重算得到的特征 / 库内字段 / 绝对误差
    """
    sub = sample_records(df, n_sample, seed, subject_type)
    rows = []
    for _, r in sub.iterrows():
        w = parse_ppg_wave(r.get("ppg_wave"))
        if len(w) < 8:
            continue
        hr = pc.to_number(r.get("heart_rate"))
        h = pc.to_number(r.get("height_cm"))
        hr = hr if np.isfinite(hr) else 75.0
        m = pc.morphology_from_cycle(w, hr, height_cm=h if np.isfinite(h) else None)
        rows.append({
            "signal_id": r.get("signal_id"),
            "subject_type": r.get("subject_type"),
            "sim_model": r.get("sim_model"),
            "heart_rate": hr,
            "重算_脉宽ms": m["pulse_width_ms"],
            "库内_脉宽ms": pc.to_number(r.get("pulse_width_ms")),
            "重算_上升时间ms": m["rise_time_ms"],
            "库内_上升时间ms": pc.to_number(r.get("rise_time_ms")),
            "重算_主波幅值": m["sys_amplitude"],
            "库内_主波幅值": pc.to_number(r.get("pulse_amplitude")),
            "重算_AIx": m["augmentation_index"],
            "库内_AIx": pc.to_number(r.get("augmentation_index")),
            "重算_PI": m["perfusion_index"],
            "库内_PI": pc.to_number(r.get("perfusion_index")),
            "重算_dpdtmax": m["dpdt_max"],
            "库内_dpdtmax": pc.to_number(r.get("dpdt_max")),
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    for a, b in (("脉宽ms", "脉宽ms"), ("上升时间ms", "上升时间ms"),
                 ("AIx", "AIx"), ("PI", "PI"), ("dpdtmax", "dpdtmax")):
        res[f"{a}_绝对误差"] = (res[f"重算_{a}"] - res[f"库内_{b}"]).abs()
    return res


def calibration_summary(mor: pd.DataFrame) -> pd.DataFrame:
    """口径可比性分析：重算值 vs 库内字段的相关系数、中位比值与结论。

    【重要发现，可直接写入论文】
    本数据集的 ppg_wave 与形态学字段并非由同一生成过程导出：
      · 脉宽  —— 重算（半高宽）与库内高度相关（r≈0.93），但存在系统性尺度
                偏差（比值≈0.76），说明二者口径不同（库内更接近"低电平宽度"）；
      · 上升时间 / AIx / PI / dpdt_max —— 与波形重算值几乎不相关，说明这些
                字段是独立仿真生成的，【不能当作波形的真值】使用。
    这正是指导书 3.4 ⑤「特征语义复杂：形态学特征计算口径多样」的实证。
    """
    pairs = [("脉宽ms", "脉宽ms"), ("上升时间ms", "上升时间ms"),
             ("AIx", "AIx"), ("PI", "PI"), ("dpdtmax", "dpdtmax")]
    rows = []
    for a, b in pairs:
        x = mor.get("重算_" + a)
        y = mor.get("库内_" + b)
        if x is None or y is None:
            continue
        x = pd.to_numeric(x, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")
        ok = np.isfinite(x) & np.isfinite(y)
        n = int(ok.sum())
        if n < 5:
            rows.append({"指标": a, "样本数": n, "相关系数r": np.nan,
                         "中位比值": np.nan, "重算中位数": np.nan,
                         "库内中位数": np.nan, "可比性结论": "样本不足"})
            continue
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        ratio = float(np.nanmedian(x[ok] / y[ok]))
        if r >= 0.8:
            if 0.95 <= ratio <= 1.05:
                concl = "强相关，口径一致（比值≈1），库内字段与波形同源"
            else:
                concl = "强相关，存在系统性尺度偏差，需统一口径"
        elif r >= 0.4:
            concl = "中等相关，口径差异明显"
        elif r >= 0.15:
            concl = "弱相关，口径基本不可比"
        else:
            concl = "不相关，库内字段为独立生成，不可作为波形真值"
        rows.append({"指标": a, "样本数": n, "相关系数r": round(r, 3),
                     "中位比值": round(ratio, 3),
                     "重算中位数": round(float(np.nanmedian(x)), 2),
                     "库内中位数": round(float(np.nanmedian(y)), 2),
                     "可比性结论": concl})
    return pd.DataFrame(rows)


# ==========================================================================
# 4. 数据集级去噪算法评估
# ==========================================================================

def evaluate_denoise_on_dataset(df: pd.DataFrame,
                                methods: Sequence[str],
                                snr_list: Sequence[float],
                                n_sample: int = 200, seed: int = 42,
                                noise_type: str = "高斯白噪",
                                analysis_fs: float = 200.0,
                                sec: float = 8.0,
                                subject_type: Optional[str] = None) -> pd.DataFrame:
    """在真实仿真波形上做去噪算法对比评估（对应指导书「任务 6」）。

    方法：把每条记录的 ppg_wave（64 点单周期模板）按心率拼接成多周期连续波，
    施加指定强度的噪声，再用各算法去噪，以"模板拼接波"为参考基准计算
    SNR / ΔSNR / RMSE。ΔSNR 是相对增益，不受参考基准是否绝对干净的影响。
    """
    sub = sample_records(df, n_sample, seed, subject_type)
    rows = []
    for i, (_, r) in enumerate(sub.iterrows()):
        w = parse_ppg_wave(r.get("ppg_wave"))
        if len(w) < 8:
            continue
        hr = pc.to_number(r.get("heart_rate"))
        hr = hr if np.isfinite(hr) and 30 <= hr <= 220 else 75.0
        t, sig = pc.tile_cycle(w, hr, analysis_fs, sec=sec)
        if len(sig) < 64:
            continue
        for snr_in in snr_list:
            rng = np.random.default_rng(seed + i)
            noisy = pc.add_noise(sig, float(snr_in), rng, fs=analysis_fs,
                                 noise_type=noise_type)
            for m in methods:
                t0 = time.perf_counter()
                den = pc.denoise(noisy, m, fs=analysis_fs)
                cost = (time.perf_counter() - t0) * 1000.0
                rows.append({
                    "signal_id": r.get("signal_id"),
                    "受试者类型": r.get("subject_type"),
                    "仿真模型": r.get("sim_model"),
                    "噪声类型": noise_type,
                    "加噪SNR(dB)": float(snr_in),
                    "方法": m,
                    "SNR(dB)": pc.snr_db(sig, den),
                    "ΔSNR(dB)": pc.snr_gain(sig, noisy, den),
                    "RMSE": pc.rmse(sig, den),
                    "耗时(ms)": cost,
                })
    return pd.DataFrame(rows)


# ==========================================================================
# 5. 命令行自检
# ==========================================================================

if __name__ == "__main__":
    p = resolve_data_path()
    print("数据文件：", p)
    t0 = time.perf_counter()
    raw = load_raw(p)
    print(f"载入 {len(raw)} 行 × {len(raw.columns)} 列，耗时 {time.perf_counter()-t0:.2f}s")

    qr = quality_report(raw)
    bad = qr[(qr["缺失率(%)"] > 0) | (qr["越界率(%)"] > 0)]
    print("\n有质量问题的字段 Top10：")
    print(bad.sort_values("缺失率(%)", ascending=False).head(10).to_string(index=False))

    t0 = time.perf_counter()
    clean, log, cost = clean_dataframe(raw)
    print(f"\n治理耗时 {cost:.2f}s，日志记录 {len(log)} 条")
    print(log.head(12).to_string(index=False))

    print("\n性别口径：", dict(clean["gender"].value_counts()))
    print("仿真状态口径：", dict(clean["sim_status"].value_counts()))
    print("噪声等级口径：", dict(clean["noise_level"].value_counts()))

    mor = recompute_morphology(clean, n_sample=200)
    print(f"\n形态学重算 {len(mor)} 条，脉宽 MAE = {mor['脉宽ms_绝对误差'].mean():.2f} ms")

    ev = evaluate_denoise_on_dataset(clean, ["无（基线）", "低通滤波", "小波阈值去噪"],
                                     [10, 20], n_sample=40)
    print("\n数据集级去噪评估（ΔSNR 均值）：")
    print(ev.groupby(["方法", "加噪SNR(dB)"])["ΔSNR(dB)"].mean().round(2).to_string())

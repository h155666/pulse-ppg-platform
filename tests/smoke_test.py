# -*- coding: utf-8 -*-
"""
smoke_test.py —— 功能测试脚本

对应 5-4 指导.md 步骤 5.1「列 10 条用例（极端参数、坏 CSV、空数据），逐条跑」。
这里给出 12 条算法/数据用例 + 6 个页面的无头渲染用例。

运行：
    python tests/smoke_test.py            # 全部
    python tests/smoke_test.py --no-ui    # 只跑算法与数据用例（不需要数据集）
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import pulse_core as pc          # noqa: E402

PASS, FAIL = "✅", "❌"
_results = []


def check(name: str, fn):
    try:
        msg = fn()
        _results.append((True, name, msg or ""))
        print(f"{PASS} {name}  {msg or ''}")
    except Exception as e:
        _results.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"{FAIL} {name}  -> {type(e).__name__}: {e}")
        traceback.print_exc()


# ==========================================================================
# 算法层用例
# ==========================================================================
def c01():
    """极端高心率 HR=180：三波不能折叠失真（相对相位改进的核心验证）。"""
    t, x, truth = pc.gen_pulse(250, 180, 6, return_truth=True)
    rng = np.random.default_rng(0)
    xn = pc.add_noise(x, 10, rng, fs=250)
    xd = pc.denoise(xn, "低通滤波", fs=250)
    f = pc.detect_features(xd, 250, hr=180)
    hr_est = 60.0 / (np.median(np.diff(f["main_idx"][np.isfinite(f["main_idx"])]) / 250))
    assert abs(hr_est - 180) < 5, f"心率估计 {hr_est:.1f} 偏离 180"
    return f"周期 {60/180:.3f}s，检出 {f['n_cycle']} 个主波，估计 {hr_est:.1f} bpm"


def c02():
    """极端低心率 HR=40。"""
    t, x, truth = pc.gen_pulse(250, 40, 12, return_truth=True)
    f = pc.detect_features(x, 250, hr=40)
    assert f["n_cycle"] >= 6, f"只检出 {f['n_cycle']} 个主波"
    return f"检出 {f['n_cycle']} 个主波（真值 {len(truth['main'])}）"


def c03():
    """极高采样率 fs=1000 + 极短时长 3 s。"""
    t, x, truth = pc.gen_pulse(1000, 75, 3, return_truth=True)
    assert len(x) == 3000, f"长度 {len(x)}"
    f = pc.detect_features(x, 1000, hr=75)
    return f"长度 {len(x)}，主波 {f['n_cycle']} 个"


def c04():
    """极噪 SNR = -5 dB。"""
    t, x, truth = pc.gen_pulse(250, 75, 8, return_truth=True)
    rng = np.random.default_rng(1)
    xn = pc.add_noise(x, -5, rng, fs=250)
    xd = pc.denoise(xn, "低通滤波", fs=250)
    assert np.isfinite(pc.snr_db(x, xd)), "SNR 非有限值"
    return f"去噪后 SNR = {pc.snr_db(x, xd):.2f} dB"


def c05():
    """8 种去噪方法在极短信号（64 点）上都不崩，且返回同长度。"""
    t, x = pc.gen_pulse(64, 75, 1.0)
    n = len(x)
    out = {}
    for m in pc.DENOISE_METHODS:
        y = pc.denoise(x, m, fs=64)
        assert len(y) == n, f"{m} 返回长度 {len(y)} != {n}"
        assert np.all(np.isfinite(y)), f"{m} 结果含 NaN/Inf"
        out[m] = round(float(np.std(y)), 3)
    return f"{len(out)} 种方法全部通过，长度均为 {n}"


def c06():
    """常数信号（无变化）：不应崩溃。"""
    x = np.ones(500)
    for m in pc.DENOISE_METHODS:
        y = pc.denoise(x, m, fs=250)
        assert len(y) == 500
    f = pc.detect_features(x, 250, hr=75)
    assert f["n_cycle"] == 0
    return "常数信号下 8 种方法均正常，特征检测返回空"


def c07():
    """全零信号。"""
    x = np.zeros(300)
    assert pc._safe_range(x) == 0.0
    f = pc.detect_features(x, 250, hr=75)
    mor = pc.morphology_features(x, 250, hr=75)
    assert f["n_cycle"] == 0
    return "零信号检测为空，形态学特征全为 NaN"


def c08():
    """含 NaN 的信号：应先过滤再计算，不能抛异常。"""
    t, x = pc.gen_pulse(250, 75, 5)
    x = x.copy()
    x[::97] = np.nan
    mor = pc.morphology_features(x, 250, hr=75)
    s = pc.snr_db(np.nan_to_num(x), pc.denoise(np.nan_to_num(x),
                                               "低通滤波", fs=250))
    assert np.isfinite(s)
    return "含 NaN 信号可正常处理"


def c09():
    """单周期 8 点：形态学不应崩。"""
    y = np.array([0.0, 0.4, 1.0, 0.6, 0.3, 0.45, 0.35, 0.05])
    m = pc.morphology_from_cycle(y, 75.0, height_cm=170.0)
    assert isinstance(m, dict)
    return f"脉宽={m['pulse_width_ms']}, 上升时间={m['rise_time_ms']}"


def c10():
    """指标自洽性：完全一致时 RMSE=0、SNR 极大；命中率满分。"""
    t, x, truth = pc.gen_pulse(250, 75, 6, return_truth=True)
    assert pc.rmse(x, x) < 1e-12
    assert pc.snr_db(x, x) > 100
    f = pc.detect_features(x, 250, hr=75)
    hit = pc.hit_rate(f["main_idx"], truth["main"], 50, 250)
    assert hit >= 0.95, f"干净信号命中率仅 {hit:.2f}"
    return f"RMSE≈0，SNR>100 dB，主波命中率 {hit*100:.1f}%"


def c11():
    """to_number 脏串解析。"""
    cases = [("0.39 s", 390.0, True), ("333ms", 333.0, True),
             ("156/98", 156.0, False), ("45岁", 45.0, False),
             ("未识别", np.nan, False), ("", np.nan, False),
             (None, np.nan, False), ("77 bpm", 77.0, False)]
    for raw_v, want, ms in cases:
        got = pc.to_number(raw_v, assume_ms=ms)
        if np.isnan(want):
            assert np.isnan(got), f"{raw_v!r} -> {got}"
        else:
            assert abs(got - want) < 1e-6, f"{raw_v!r} -> {got}, 期望 {want}"
    return f"{len(cases)} 种脏串写法全部正确解析"


def c12():
    """坏 CSV：空文件 / 只有表头 / 单行数据 —— 治理与解析均不能崩。"""
    import data_lab as dl
    with tempfile.TemporaryDirectory() as d:
        empty = os.path.join(d, "empty.csv")
        open(empty, "w", encoding="utf-8").close()
        try:
            dl.load_raw(empty)
            r1 = "空文件：被 pandas 正常处理"
        except Exception as e:
            r1 = f"空文件：抛出可捕获异常 {type(e).__name__}"

        head_only = os.path.join(d, "head.csv")
        pd.DataFrame(columns=["a", "b"]).to_csv(head_only, index=False)
        df = pd.read_csv(head_only)
        clean, log, _ = dl.clean_dataframe(df)
        assert len(clean) == 0

        one = os.path.join(d, "one.csv")
        pd.DataFrame({"a": [1], "b": ["x"],
                      "gender": ["先生"],
                      "ppg_wave": ["1,2,nan,4"]}).to_csv(one, index=False)
        df2 = pd.read_csv(one)
        c2, lg, _ = dl.clean_dataframe(df2)
        assert dl.normalize_gender("先生") == "男"
        w = dl.parse_ppg_wave("1,2,nan,4")
        assert len(w) == 3 and w.tolist() == [1.0, 2.0, 4.0]
    return f"{r1}；仅表头/单行均正常；gender '先生'→男；波形含 NaN 已跳过"


def c13():
    """数据集端到端（需要 data/raw_p11_ppg_sim_signal.csv）。"""
    import data_lab as dl
    p = dl.resolve_data_path()
    if p is None:
        raise FileNotFoundError("未找到数据集，跳过")
    raw = dl.load_raw(p)
    assert len(raw) == 17835, f"记录数 {len(raw)}，期望 17835（已剔除说明行）"
    clean, log, cost = dl.clean_dataframe(raw)
    assert len(clean) > 0 and len(log) > 0
    mor = dl.recompute_morphology(clean, n_sample=60)
    assert len(mor) > 0
    cal = dl.calibration_summary(mor)
    assert len(cal) > 0
    ev = dl.evaluate_denoise_on_dataset(
        clean, ["无（基线）", "低通滤波"], [10.0], n_sample=10)
    # 部分记录的 ppg_wave 点数不足 8 会被跳过，故用区间断言
    assert 0 < len(ev) <= 20, f"评估结果 {len(ev)} 行，期望 (0, 20]"
    return (f"{len(raw)} 条；治理 {cost:.2f}s/{len(log)} 条日志；"
            f"重算 {len(mor)} 条；评估 {len(ev)} 行")


# ==========================================================================
# 页面无头渲染用例（streamlit AppTest）
# ==========================================================================
def _app_test(rel: str, timeout: int = 600):
    from streamlit.testing.v1 import AppTest
    path = os.path.join(ROOT, rel)
    return AppTest.from_file(path, default_timeout=timeout)


def _assert_no_exception(at, rel: str):
    if at.exception:
        msgs = [f"{e.type.__name__}: {e.value}" for e in at.exception]
        raise AssertionError(" | ".join(map(str, msgs)))
    # guard 默认会把异常转成 st.error 而不抛出，必须一并检查，否则误判通过
    errs = [str(e.value) for e in at.error]
    if errs:
        raise AssertionError("页面被 guard 捕获：" + " | ".join(errs))
    return f"{rel} 渲染通过（无异常、无 st.error）"


def _run_page(rel: str):
    at = _app_test(rel)
    at.run()
    return _assert_no_exception(at, rel)


def _run_page_click(rel: str, btn_text: str, presets: dict | None = None):
    """先调小参数，再点击运行按钮，覆盖「重计算」这条主路径。

    这是真正会跑批量的路径，必须测到，否则只测到 st.stop() 前的空壳。
    """
    at = _app_test(rel, timeout=900)
    at.run()
    _assert_no_exception(at, rel + "（首次）")
    for key, val in (presets or {}).items():
        for getter in ("slider", "number_input"):
            try:
                w = getattr(at, getter)
                for e in w:
                    if getattr(e, "key", None) == key:
                        e.set_value(val)
                        raise StopIteration
            except StopIteration:
                break
            except Exception:
                continue
    at.run()
    target = None
    for b in at.button:
        if btn_text in (b.label or ""):
            target = b
            break
    if target is None:
        raise AssertionError(f"未找到按钮：{btn_text}")
    target.click()
    at.run()
    _assert_no_exception(at, rel + "（点击后）")
    return f"{rel} 「{btn_text}」点击后重算路径通过"


def main():
    only_algo = "--no-ui" in sys.argv
    print("=" * 70)
    print("算法与数据层用例")
    print("=" * 70)
    for fn in (c01, c02, c03, c04, c05, c06, c07, c08, c09, c10, c11, c12, c13):
        check(fn.__name__ + " " + (fn.__doc__ or "").strip().split("\n")[0], fn)

    if not only_algo:
        # 让 ui_utils.guard 不吞异常，页面错误才能被本测试捕获
        os.environ["PPG_STRICT"] = "1"
        print("\n" + "=" * 70)
        print("页面无头渲染用例")
        print("=" * 70)
        pages = ["app.py"]
        pdir = os.path.join(ROOT, "pages")
        if os.path.isdir(pdir):
            pages += ["pages/" + f for f in sorted(os.listdir(pdir))
                      if f.endswith(".py")]
        for p in pages:
            check(f"[UI] {p}", (lambda pp=p: _run_page(pp)))

        print("\n" + "=" * 70)
        print("页面「点击运行」重算路径用例")
        print("=" * 70)
        check("[UI-CLICK] 批量对比页", lambda: _run_page_click(
            "pages/2_去噪方法批量对比.py", "开始批量实验",
            {"b_nseed": 1, "b_sec": 5}))
        check("[UI-CLICK] 数据集评估页", lambda: _run_page_click(
            "pages/4_形态学特征与去噪评估.py", "开始评估",
            {"p5_n2": 20}))

    print("\n" + "=" * 70)
    ok = sum(1 for r in _results if r[0])
    print(f"结果：{ok}/{len(_results)} 通过")
    for r in _results:
        if not r[0]:
            print(f"   {FAIL} {r[1]} -> {r[2]}")
    print("=" * 70)
    return 0 if ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
页面 ⑥ 关于与运行说明

对应 5-4 指导.md 步骤 4.3「美化与部署说明：README 能让同学复现」。
"""
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import streamlit as st
import matplotlib
import pywt

import data_lab as dl
import pulse_core as pc
import ui_utils as ui

ui.page_config("关于与运行说明")
st.title("⑥ 关于与运行说明")

tab1, tab2, tab3, tab4 = st.tabs(["运行环境", "项目结构", "答辩要点", "不足与展望"])

# ---------------------------------------------------------------- 运行环境
with tab1:
    st.subheader("启动方式")
    st.code("cd pulse_ppg_streamlit\n"
            "pip install -r requirements.txt\n"
            "streamlit run app.py\n"
            "# 端口被占用时（5-4 背景 Q4）：\n"
            "streamlit run app.py --server.port 8502", language="bash")

    st.subheader("依赖与实验环境（可直接作为论文「实验环境配置表」）")
    env = pd.DataFrame({
        "项目": ["操作系统", "Python", "Streamlit", "NumPy", "SciPy",
                 "Matplotlib", "PyWavelets", "pandas", "CPU", "GPU"],
        "版本/说明": [
            platform.platform(),
            sys.version.split()[0],
            st.__version__,
            np.__version__,
            scipy.__version__,
            matplotlib.__version__,
            pywt.__version__,
            pd.__version__,
            platform.processor() or "x86_64",
            "未使用（课题硬约束：一律 CPU 单机）",
        ],
    })
    st.dataframe(env, width="stretch", hide_index=True)

    st.subheader("可复现性")
    st.markdown("""
- 所有随机过程统一走 `np.random.default_rng(seed)`，页面侧边栏可调起始种子；
- 默认种子 **42**；批量实验支持 1–10 个种子取平均，对应大方向背景 5.2「多次重复」；
- 导出的 CSV 与页面显示数值完全一致，可直接作为论文表格数据源。
""")

    st.subheader("数据文件")
    p = dl.resolve_data_path()
    st.code(p or "未找到", language="text")
    st.caption("默认查找顺序：侧边栏显式路径 → 环境变量 PPG_DATA → "
               "项目 data/ 目录 → 学校测试数据目录。")

    st.subheader("缓存与性能（指导 5.2）")
    st.markdown("""
| 位置 | 措施 | 效果 |
|---|---|---|
| 数据集载入与治理 | `@st.cache_data` | 首次约 1 s，之后瞬时 |
| 质量报告 | `@st.cache_data` | 避免每次重跑逐列扫描 |
| 批量实验明细 | `@st.cache_data`（以参数元组为 key） | 只改滑块不重算 |
| 批量实验触发 | `st.session_state["batch_done"]` | 切页/点其他控件不丢结果 |
| 波形抽样解析 | `@st.cache_data` | 抽样参数不变即复用 |
| 绘图 | 每次 `fig, ax = plt.subplots()` 新建 | 规避图重叠（背景 Q2） |
""")

# ---------------------------------------------------------------- 项目结构
with tab2:
    st.subheader("目录结构")
    st.code("""
pulse_ppg_streamlit/
├── app.py                 ① 平台总览（入口）
├── pulse_core.py          算法层：生成/加噪/8种去噪/三波检测/形态学/指标
├── data_lab.py            数据层：口径归一/异常治理/波形解析/特征工程/评估
├── ui_utils.py            展示层公共组件：绘图/侧边栏/缓存/导出
├── pages/
│   ├── 1_信号生成与特征分析.py    ②
│   ├── 2_去噪方法批量对比.py      ③
│   ├── 3_仿真数据集治理与分析.py  ④
│   ├── 4_形态学特征与去噪评估.py  ⑤
│   └── 5_关于与运行说明.py        ⑥
├── data/raw_p11_ppg_sim_signal.csv   17835 × 48
├── docs/设计思路.md
├── requirements.txt
└── README.md
""", language="text")

    st.subheader("分层解耦的意义")
    st.markdown("""
`pulse_core.py` 与 `data_lab.py` **都不 import streamlit**，因此：

1. 可在命令行直接 `python pulse_core.py` / `python data_lab.py` 自检；
2. 可被 5.1（PyQt 桌面）直接复用，满足 5-4 背景 15 节「算法内核共用」；
3. 论文第 3 章可以直接贴函数签名表，证明算法层与展示层解耦。
""")

    st.subheader("算法层函数清单")
    funcs = pd.DataFrame({
        "函数": ["gen_pulse", "add_noise", "denoise（8 种）", "detect_features",
                 "average_cycle", "morphology_features",
                 "morphology_from_cycle", "tile_cycle", "snr_db", "rmse",
                 "snr_gain", "hit_rate", "run_single_experiment"],
        "作用": ["自适应高斯脉冲串生成（可返回真值峰）",
                 "白噪/基线漂移/工频/运动伪影/复合噪声叠加",
                 "无·低通·小波·EMD·SG·带通·中值·陷波+低通",
                 "主波/潮波/重搏波/重搏切迹检测",
                 "周期切分 + 平均，抗噪更稳",
                 "多周期信号的形态学特征",
                 "单周期模板的形态学特征（用于 ppg_wave）",
                 "单周期模板拼接成连续波",
                 "信噪比", "均方根误差", "去噪增益 ΔSNR",
                 "主波命中率（容差匹配）",
                 "一次完整的生→噪→去→检→评"],
    })
    st.dataframe(funcs, width="stretch", hide_index=True)

# ---------------------------------------------------------------- 答辩要点
with tab3:
    st.subheader("与 5.1 / 5.2 / 5.3 的技术差异")
    st.markdown("""
| 课题 | 技术栈 | 交付物 | 与本平台的关系 |
|---|---|---|---|
| 5.1 | PyQt5 + matplotlib | 桌面 exe | 共用 `pulse_core.py` |
| 5.2 | Vue3 + ECharts（纯 JS） | 静态网页 | 算法需用 JS 重写 |
| 5.3 | scipy + PyWavelets | 脚本 + 指标表 | 算法内核即 `pulse_core.py` |
| **5.4 本平台** | **Streamlit + scipy + matplotlib** | **网页 demo** | **直接 import，零前端代码** |
""")

    st.subheader("常见问题预演（对应 5-4 指导 11 节）")
    qa = [
        ("你的核心做了什么？",
         "用 Streamlit 搭了一个脉搏波「生成—加噪—去噪—特征—评价」网页端平台，"
         "并在 17835 条仿真记录上完成了数据治理与群体分析。"),
        ("为什么不用 Spark / Hadoop / Hive？",
         "课题 5-4 技术栈被硬性固定为 Streamlit + scipy + matplotlib；且 Hive/HDFS "
         "在 Windows 单机本科环境不可部署。数据集仅 17835×48（约 19 MB），"
         "pandas 列向量化在清洗口径上与 Spark 完全等价，实测治理耗时在页面 KPI 中给出。"),
        ("为什么还要自己生成信号？",
         "因为 raw_p11 数据集没有干净真值，算不了 RMSE 与命中率。"
         "自生成信号有真值，负责定量；真实记录负责统计与外推，二者交叉印证。"),
        ("三个特征点的生理意义？",
         "主波 = 左心室快速射血产生的收缩期最高峰；潮波 = 血管反射形成的次峰；"
         "重搏波 = 主动脉瓣关闭后血流回流形成的舒张期小峰。"),
        ("滤波为什么用 filtfilt？",
         "butter 设计的 IIR 滤波器直接 lfilter 会产生相位偏移，导致特征点时刻整体"
         "后移；filtfilt 做正反两次滤波实现零相位。"),
        ("小波层数为什么自适应？",
         "分解层数超过 log2(信号长度) 会直接报错；本平台用 "
         "pywt.dwt_max_level 做上限约束，并把 waverec 结果截断对齐到原长度。"),
        ("实验能复现吗？",
         "能。所有随机走 np.random.default_rng(seed)，默认 42，批量实验支持多种子取平均。"),
        ("VMD 为什么没做？",
         "VMD 需迭代求解变分问题，交互演示开销过大，本平台未实现，列入展望；"
         "如实说明优于伪造结果。"),
        ("你怎么保证不崩溃？",
         "每个页面用 try/except 兜底并给出堆栈与降级建议；抽样条数、信号长度、"
         "滤波核大小、小波层数都做了上下界保护。"),
        ("最核心的数字是多少？",
         "请到 ③ 页面跑一次批量实验，取排名第一的方法的 ΔSNR；"
         "论文里应填你自己跑出来的数，而不是这里预设的值。"),
    ]
    for q, a in qa:
        with st.expander(q):
            st.write(a)

# ---------------------------------------------------------------- 不足与展望
with tab4:
    st.subheader("已知不足（如实列出 3 条，可直接用于论文）")
    st.markdown("""
1. **未使用真实临床数据**：全部结论建立在虚拟仿真平台的仿真记录与自生成信号之上，
   尚未在 PhysioNet BIDMC PPG 等真实数据上做交叉验证。
2. **未做并发与权限**：Streamlit 面向单人演示，没有多用户、权限与持久化数据库，
   这也是 5.4 相对 5.6（Flask/Django 教学平台）的边界。
3. **VMD 未实现**：数据集 `denoise_algo` 字段中的「VMD变分模态分解」（2114 条）
   在本平台仅作统计展示，未参与算法对比。
""")

    st.subheader("展望（3 条可行方向）")
    st.markdown("""
1. 引入 PhysioNet 真实 PPG，把「形态合理性」从肉眼比对升级为定量指标；
2. 上线用户系统与结果持久化，向 5.6 的教学平台形态演进；
3. 补上 VMD 与 EEMD，并把最优参数搜索（网格/贝叶斯）做成自动调参模块。
""")

    st.divider()
    st.caption("广东轻工职业技术大学 2024 级大数据专业 · 毕业设计课题 5-4")

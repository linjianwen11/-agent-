import numpy as np

def load_csv(path):
    """加载 CSV 文件"""
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return header, data

def sample_rate(t):
    """采样率,从时间列"""
    dt = np.median(np.diff(t))
    return 1.0/dt

def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))

def window_rms_series(x, fs, window_s=1.0, rms_floor=0.05):
    """计算窗口 RMS 序列"""
    n = int(window_s * fs)
    n_windows = len(x) // n
    times, vals = [], []
    for i in range(n_windows):
        seg = x[i*n:(i+1)*n]
        r = rms(seg)
        if r >= rms_floor:
            times.append(i*window_s)
            vals.append(r)
    return np.array(times), np.array(vals)

def median_frequency(x, fs):
    """计算中位频率"""
    n = len(x)

    if n < 4:
        return 0.0
    power = np.abs(np.fft.rfft(x)) ** 2 #功率谱
    freqs = np.fft.rfftfreq(n, d=1.0/fs) #频率
    total = power.sum()
    if total <= 0:
        return 0.0
    cum = np.cumsum(power)
    idx = int(np.searchsorted(cum, total/2.0)) #频率的一半
    return float(freqs[min(idx, len(freqs)-1)])


def window_mdf_series( x, fs, window_s = 1.0, rms_floor = 0.05):
    """计算窗口中位频率序列"""
    n = int(window_s * fs) #1s * 采样率 一个窗口的采样率
    n_windows= len(x) // n #窗口数量
    times, vals = [], []
    for i in range(n_windows):
        seg = x[i * n : (i+1) * n]
        if rms(seg) < rms_floor:
            continue
        times.append(i*window_s)
        vals.append(median_frequency(seg, fs))
    return np.array(times), np.array(vals)

def slope_per_minute(time_s, values):
    """计算每分钟的斜率"""
    if len(values) < 2:
        return 0.0
    k, _ = np.polyfit(time_s, values, 1)
    return float(k*60.0)

def fatigue_report(x, fs, window_s=1.0, rms_floor=0.05, mdf_thresh = 2.0):
    """疲劳报告,rms上升,mdf下降,就是疲劳"""
    t_r, rms_s = window_rms_series(x, fs, window_s, rms_floor)
    t_m, mdf_s = window_mdf_series(x, fs, window_s, rms_floor)

    rms_slope = slope_per_minute(t_r, rms_s)
    mdf_slope = slope_per_minute(t_m, mdf_s)
    if mdf_slope < mdf_thresh and rms_slope > 0.0:
        conclusion = "重度疲劳"
    elif mdf_slope < 0 and rms_slope > 0:
        conclusion = "轻度疲劳"
    else:
        conclusion = "未见明显疲劳"
    
    return {
        "n_native_window":int(len(rms_s)),
        "rms_series": [round(float(v), 4) for v in rms_s],
        "mdf_series": [round(float(v), 1) for v in mdf_s],
        "conclusion": conclusion,
        "rms_slope_per_minute": round(rms_slope, 4),
        "mdf_slope_per_minute": round(mdf_slope, 2),
        "conclusion": conclusion,
    }
    
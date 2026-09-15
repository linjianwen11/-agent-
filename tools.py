
import os

import bio_signal as sig
import config

DATA_DIR = config.DATA_DIR

def list_data_filies():
    """列出所有数据文件"""
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

def emg_basic_stats(filename):
    """计算EMG信号的基本统计量"""
    header, data = sig.load_csv(os.path.join(DATA_DIR, filename))
    fs = sig.sample_rate(data[:, 0])
    return {
        "columns":header,
        "sample_rate_hz":round(fs, 1),
        "duration_s":round(float(data[-1, 0]), 2),
        "n_samples":int(len(data)),
        "overall_rms":round(sig.rms(data[:, 1]), 4),
    }
def analyze_emg(filename):
    """分析EMG信号并且判断疲劳疲劳程度"""
    header,data = sig.load_csv(os.path.join(DATA_DIR, filename))
    fs = sig.sample_rate(data[:, 0])

    return sig.fatigue_report(data[:, 1], fs)

def compare_emg(file_a, file_b):
    """对比两个emg 文件的疲劳指标，返回双方的报告和对比结论"""
    _, da = sig.load_csv(os.path.join(DATA_DIR, file_a))
    _, db = sig.load_csv(os.path.join(DATA_DIR, file_b))

    ra = sig.fatigue_report(da[:, 1], sig.sample_rate(da[:, 0]))
    rb = sig.fatigue_report(db[:, 1], sig.sample_rate(db[:, 0]))
    return {
        "file_a": file_a,
        "report_a": ra,
        "file_b": file_b,
        "report_b": rb,
        "comparison": {
            "mdf_slope_diff_hz_minute": round(
                rb["mdf_slope_per_minute"] - ra["mdf_slope_per_minute"], 2),
            "rms_slope_diff_hz_minute": round(
                rb["rms_slope_per_minute"] - ra["rms_slope_per_minute"], 4),
            "more_fatigued_file": (
                file_a if ra["mdf_slope_per_minute"] < rb["mdf_slope_per_minute"]
                else file_b),
        },
    }

#给llm的工具名片，简历
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_data_filies",
            "description": "列出data目录下所有的可用数据文件名",
            "parameters": {"type": "object","properties": {} },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emg_basic_stats",
            "description": "读取指定数据文件，返回采样率、时长、样本数和整体 rms",
            "parameters": {
                "type":"object", 
                "properties": {
                   "filename": {
                       "type":"string",
                       "description":"文件数据名称比如 emg_fatigue.csv"
                   }
                },
                "required": ["filename"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_emg",
            "description": "分析指定emg数据的疲劳程度,返回rms/mdf 斜率和疲劳结论",
            "parameters": {
                "type":"object", 
                "properties": {
                   "filename": {
                       "type":"string",
                       "description":"文件数据名称比如 emg_fatigue.csv"
                   }
                },
                "required": ["filename"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_emg",
            "description": "对比两个 emg 数据文件的疲劳程度，返回双方 rms/mdf 斜率和更疲劳的文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_a": {
                        "type": "string",
                        "description": "第一个文件名比如 emg_fatigue.csv"
                    },
                    "file_b": {
                        "type": "string",
                        "description": "第二个文件名比如 emg_fatigue.csv"
                    }
                },
                "required": ["file_a", "file_b"]
            }
        }
    }
]

TOOL_MAP = {
    "list_data_filies": list_data_filies,
    "emg_basic_stats": emg_basic_stats,
    "analyze_emg": analyze_emg,
    "compare_emg": compare_emg,
}


import os

# config.py —— 所有"设置"集中在这里
import os

API_KEY = os.getenv("DEEPSEEK_API_KEY")   # 密钥从环境变量读，不进代码
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

MAX_MEMORY = 20          # 最多保留多少条对话消息

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


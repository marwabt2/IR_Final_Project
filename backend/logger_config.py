# backend/logger_config.py

import logging
import os

# أنشئ مجلد logs إذا ما كان موجود
os.makedirs("logs", exist_ok=True)

# === إعداد اللوغ الأساسي ===
logger = logging.getLogger("IR_Project")
logger.setLevel(logging.DEBUG)  # غيريها حسب الحاجة (DEBUG, INFO, WARNING...)

# ✅ فورمات موحد
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S"
)

# === File Handler: يسجّل في ملف logs/system.log ===
file_handler = logging.FileHandler("logs/system.log", mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# === Stream Handler: يطبع في الكونسول ===
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

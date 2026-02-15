import re
import os

config_path = 'configs/s3dis/semseg-pt-v3m1-0-base.py'

if not os.path.exists(config_path):
    print("❌ 找不到配置文件！")
    exit()

with open(config_path, 'r') as f:
    content = f.read()

print("🔍 正在扫描并修复配置文件...")

# 1. 强制将输入通道设为 3 (RGB)
content = re.sub(r'in_channels\s*=\s*\d+', 'in_channels=3', content)

# 2. 核心修复：把所有的 feat_keys = ... 统统改成 feat_keys=['color']
# 这个正则会匹配 feat_keys=('color') 或 feat_keys='color' 等各种形式
content = re.sub(r"feat_keys\s*=\s*[\(\['\"]+.*[\)\]'\"]+", "feat_keys=['color']", content)

# 3. 防止之前的错误操作导致双重列表 (比如 [['color']])
content = content.replace("[['color']]", "['color']")

with open(config_path, 'w') as f:
    f.write(content)

print("✅ 修复完成！关键参数检查：")
for line in content.split('\n'):
    if 'feat_keys' in line or 'in_channels' in line:
        print(f"  -> {line.strip()}")

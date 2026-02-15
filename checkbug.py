# 运行这个快速检查脚本
import numpy as np
import glob
path = "data/stpls3d/processed/"
files = glob.glob(path + "/**/segment.npy", recursive=True)
if files:
    sample = np.load(files[0])
    print(f"Sample labels from training path: {np.unique(sample)}")
    print(f"Total files in training path: {len(files)}")
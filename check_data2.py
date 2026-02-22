import torch
import os

path = 'data/sensaturban/processed/val'
files = sorted([f for f in os.listdir(path) if f.endswith('.pth')])

if not files:
    print(f"Error: No .pth files found in {path}")
else:
    file_path = os.path.join(path, files[0])
    print(f"\nChecking file: {file_path}")
    d = torch.load(file_path)
    print(f"Keys: {list(d.keys())}")
    
    if 'segment' in d:
        labels = d['segment']
        # 兼容 numpy 和 tensor
        unique_labels = torch.unique(torch.as_tensor(labels))
        print(f"Unique Labels found in data: {unique_labels.tolist()}")
        print(f"Number of points: {len(labels)}")
    else:
        print("Key 'segment' not found!")
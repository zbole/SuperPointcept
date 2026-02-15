_base_ = ["../_base_/default_runtime.py"]

# 核心参数
weight = None
resume = False
evaluate = True
test_only = False
seed = 42
save_path = "exp/semantic_kitti/semseg-pt-v3m1-0-base"
num_worker = 16  # 如果显存爆了改成 8 或 4
batch_size = 4   # 5090 24G 显存应该可以跑 4 或 6
batch_size_val = None
batch_size_test = None
epoch = 100
eval_epoch = 100
sync_bn = False
enable_amp = False
empty_cache = False
find_unused_parameters = False

# 模型定义 (PTv3)
model = dict(
    type="DefaultSegmentorV2",
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True, # 5090 必须开启 Flash Attention!
        upcast_attention=False,
        upcast_softmax=False,

        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    # SemanticKITTI 有 20 类 (0是ignore)
    num_classes=19, 
    backbone_out_channels=64,
)

# 数据集加载 (这里最关键)
data = dict(
    num_classes=19, # unlabeled全部扔到 255
    ignore_index=-1,
    names=[
        "car", "bicycle", "motorcycle", "truck", "other-vehicle",
        "person", "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
        "other-ground", "building", "fence", "vegetation", "trunk", "terrain",
        "pole", "traffic-sign"
    ],
    train=dict(
        type="SemanticKITTIDataset",
        split="train",
        data_root="data/semantic_kitti", # 指向你刚才做好的链接
        transform=[
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=1000000, mode="random", sample_rate=0.8),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type="SemanticKITTIDataset",
        split="val",
        data_root="data/semantic_kitti",
        transform=[
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
    ),
)

# 1. 使用 AdamW 优化器 (PTv3 标配)
optimizer = dict(type='AdamW', lr=0.002, weight_decay=0.005)

# 2. 使用 OneCycleLR 调度策略 (先热身再衰减，收敛最快)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.002,           # 最高学习率
    pct_start=0.04,         # 热身阶段占比 4%
    anneal_strategy='cos',  # 余弦退火
    div_factor=10.0,        # 初始学习率 = max_lr / 10
    final_div_factor=10000.0 # 最终学习率 = 初始学习率 / 10000
)
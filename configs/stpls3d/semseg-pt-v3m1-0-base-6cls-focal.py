_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4  # 根据显存调整
num_worker = 8
mix_prob = 0.8
empty_cache = True
enable_amp = False

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=6,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
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
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D", "STPLS3D"),
    ),
    criteria=[
        # 🚨 ignore_index 必须改为 255，匹配预处理脚本的填充值
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=255),
        dict(type="FocalLoss", gamma=2.0, loss_weight=1.0, ignore_index=255),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=255),
    ],
)

# scheduler settings
epoch = 100
optimizer = dict(type="AdamW", lr=0.003, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.003, 0.0003],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings
dataset_type = "STPLS3DDataset"
# 🚨 指向新的预处理路径
data_root = "data/stpls3d/processed/grid0.10_chunk50x50_stride25x25"

data = dict(
    num_classes=6,
    ignore_index=255,
    names=["Ground", "Building", "Vegetation", "Vehicle", "LightPole", "Fence"],
    train=dict(
        type=dataset_type,
        split="train", # 注意：如果你的预处理文件夹里没分train/val，这里可能需要调整，或者 Pointcept 会自动搜所有文件
        data_root=data_root,
        transform=[
            # ✅ 加回这个映射！这是提分的关键！
            dict(
                type="CategoryMapping",
                mapping_dict={
                    0: 0, 15: 0, 18: 0, 19: 0,  # Ground类合并
                    1: 1, 17: 1,                # Building类合并
                    2: 2, 3: 2, 4: 2,           # Veg类合并
                    5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, # Vehicle类合并
                    11: 4,                      # LightPole
                    14: 5,                      # Fence
                    12: 255, 13: 255            # 杂物和牌子忽略 (Benchmark通常忽略)
                },
            ),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout",
                dropout_ratio=0.2,
                dropout_application_ratio=0.2
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.75),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.1,  # 训练时通常保持和预处理一致或稍大
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="SphereCrop", point_max=100000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(
                type="CategoryMapping",
                mapping_dict={
                    0: 0, 15: 0, 18: 0, 19: 0,
                    1: 1, 17: 1,
                    2: 2, 3: 2, 4: 2,
                    5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3,
                    11: 4,
                    14: 5,
                    12: 255, 13: 255
                },
            ),
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.1, # 验证集保持 0.1
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        dict(
                type="CategoryMapping",
            ),
        type=dataset_type,
        split="val",
        data_root=data_root + "",
        transform=[
            # 🚨 核心修复：必须加上这个 Mapping！
            dict(
                type="CategoryMapping",
                mapping_dict={
                    0: 0, 15: 0, 18: 0, 19: 0,  # Ground, Road, Dirt, Grass -> 0
                    1: 1, 17: 1,                # Building
                    2: 2, 3: 2, 4: 2,           # Vegetation
                    5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, # Vehicle
                    11: 4,                      # LightPole
                    14: 5,                      # Fence
                    12: 255, 13: 255            # Ignore
                },
            ),
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)
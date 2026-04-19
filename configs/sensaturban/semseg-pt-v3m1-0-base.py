weight = None
resume = False
evaluate = True
test_only = False
seed = 2026
save_path = 'exp/urbanbis/urbanbis_inst'
num_worker = 16
batch_size = 8
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 100
eval_epoch = 100
clip_grad = 1.0
sync_bn = False
enable_amp = True
amp_dtype = 'float16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = True # 🚀 端到端架构可能有丢弃的 Query，设为 True 防止报错
enable_wandb = True
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=0.0001)]

hooks = [
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    # 🚀 替换为 InstanceSegEvaluator (Pointcept 原生用于实例分割的评估器)
    dict(type='InstanceSegEvaluator'), 
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]

train = dict(type='DefaultTrainer')
test = dict(type='InstanceSegTester', verbose=True) # 🚀 改为实例分割 Tester

model = dict(
    type='QueryBasedBuildingSegmenter', 
    hidden_dim=64,       
    num_queries=150,     
    ptv3_backbone=dict(
        type='PT-v3m1',
        in_channels=7,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
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
        pdnorm_conditions=('ScanNet', 'S3DIS', 'Structured3D', 'SensatUrban')
    ),
    # 🚀 引入我们刚刚写的二分图 Loss
    criteria=[
        dict(
            type='BuildingInstanceLoss',
            building_class_id=2, # UrbanBIS 建筑物的 Label ID
            w_cls=2.0,
            w_mask=5.0,
            w_dice=5.0,
            w_sem=1.0
        )
    ]
)

optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.001, 0.0001],
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)

dataset_type = 'DefaultDataset'

# 🚀 换成你最新的 DINO 1024D + 实例标签 的路径
data_root = '/lus/lfs1aip2/projects/b6ae/datasets/UrbanBIS/processed_1025D_Inst/' 

data = dict(
    num_classes=13, 
    ignore_index=255,
    names=['Ground', 'Vegetation', 'Building', 'Wall', 'Bridge', 'Parking',
           'Rail', 'TrafficRoad', 'StreetFurniture', 'Car', 'Footpath', 'Bike', 'Water'],
    train=dict(
        type='DefaultDataset',
        split='train',
        data_root=data_root,
        transform=[
            dict(type='CenterShift', apply_z=True),
            # ... 中间的各种 Augmentation 保持不变 ... (为节省篇幅略去)
            dict(type='SphereCrop', sample_rate=0.8, mode='random'),
            dict(type='SphereCrop', point_max=200000, mode='random'),
            dict(type='CenterShift', apply_z=True),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                # 🚀 必须确保 instance 被装载到 DataLoader 里
                keys=('coord', 'grid_coord', 'segment', 'instance'),
                feat_keys=('coord', 'color', 'extra_feat')
            )
        ],
        test_mode=False,
        loop=5),
        
    val=dict(
        type='DefaultDataset',
        split='val',
        data_root=data_root,
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.1,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                return_inverse=True),
            dict(type='CenterShift', apply_z=True),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                # 🚀 VAL 集也必须带上 instance 来算 AP/AR 指标
                keys=('coord', 'grid_coord', 'segment', 'instance', 'inverse', 'origin_segment'),
                feat_keys=('coord', 'color', 'extra_feat')
            )
        ],
        test_mode=False),
    
    test=dict(
        type='DefaultDataset',
        split='test',
        data_root=data_root,
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='NormalizeColor')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.1,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True), # 🚀 删除了 keys 参数，保持原生结构
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=True),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    # 🚀 只改这一行：把 'index' 加回来！
                    keys=('coord', 'grid_coord', 'index'), 
                    feat_keys=('coord', 'color', 'extra_feat'))
            ],
            aug_transform = [
            # Angle 1: 0度 (Baseline)
            [{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }],
            # Angle 2: 120度 (2/3 Pi)
            [{
                'type': 'RandomRotateTargetAngle',
                'angle': [np.pi * 2 / 3], 
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }],
            # Angle 3: 240度 (4/3 Pi)
            [{
                'type': 'RandomRotateTargetAngle',
                'angle': [np.pi * 4 / 3],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }]
        ]
        )
    ),
)

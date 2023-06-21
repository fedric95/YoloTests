_base_ = 'mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'


data_root = '/code/mmdetection/data/coco_style/'

pad_val = 0.0

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=pad_val,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    #dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(pad_val, pad_val, pad_val))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(pad_val, pad_val, pad_val),
        prob=0.5),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(pad_val, pad_val, pad_val))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_dataloader = dict(
    batch_size=32,
    num_workers=10,
    dataset=dict(
        data_root = '/code/mmdetection/data/coco_style/',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=5,
    num_workers=10,
    dataset=dict(
        data_root='/code/mmdetection/data/coco_style/',
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/test/'),
        pipeline = val_pipeline
    )
)

test_dataloader = val_dataloader  # Testing dataloader config

val_evaluator = dict(ann_file='/code/mmdetection/data/coco_style/annotations/test.json')

test_evaluator = val_evaluator  # Testing evaluator config


model = dict(bbox_head=dict(num_classes=1))



max_epochs = 50
interval = 5

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval
)

# optimizer
# lr is set for a batch size of 8
optim_wrapper = dict(optimizer=dict(lr=0.01))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001, 
        by_epoch=False, 
        begin=0, 
        end=500
    )
]



load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
    head=dict(
        num_classes=5,
        topk=(1,),
    ),
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',  # noqa
            prefix='backbone'
        )
    )
)
# load_from='F:\\aa学习资料\\24-25大三下\\深度学习\\mmpretrain-main\\work_dirs\\resnet50_8xb32_in1kForTask\\EPOCH_14.pth'

data_processor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
    num_classes=5,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

dataset_type = 'ImageNet'
data_root = 'F:/aa学习资料/24-25大三下/深度学习/assignment1/flower_dataset'
classes = [
    'daisy', 'dandelion', 'rose', 'sunflower', 'tulip'
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,  # 典型值可以是4/8，根据CPU核心数调整
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file=f'{data_root}/train.txt',  # 如果有标注文件
        pipeline=test_pipeline,
        data_prefix=f'{data_root}/train',
        classes=classes  # 明确指定类别
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,  # 典型值可以是4/8，根据CPU核心数调整
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file=f'{data_root}/val.txt',  # 如果有标注文件
        pipeline=train_pipeline,
        data_prefix=f'{data_root}/val',
        classes=classes  # 明确指定类别
    )
)

val_ccfg= dict()
val_evaluator = dict(type='Accuracy', topk=(1,))

optim_wrapper = dict(
    optimizer=dict(type='SGD',lr=0.005, momentum=0.9, weight_decay=0.0001),)
auto_scale_lr = dict(base_batch_size=256)

train_cfg = dict(
    by_epoch=True,
    max_epochs=20,
    val_interval=1,
)

workDir='/data/test0.10.1/mmdet/'
#############################################################
######################特别注意################################
#前两行预留，可以什么都不写，但是必须留出来，否则程序会覆盖前两行
####可以修改分数阈值，可以修改nms###############################
##############################################################
classNum=1
mean=[0,0,0]
std=[127,127,127]
imageScale=(512,512)
testScale=(512,512)
classes=['bicycle']
datasetype='EhlXMLCocoDataSet'
logger_interval=1
maxepochs=100
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=10),
    dict(
        type='MultiStepLR',
        begin=0,
        end=maxepochs,
        by_epoch=True,
        milestones=[30, 60],
        gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(norm_type=2,max_norm=35),
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
model = dict(
    type='RetinaNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=mean,
        std=std,
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=classNum,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(type='PseudoSampler'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=datasetype,
        dir=workDir+'train',
        classes=classes,
        debugSep=1,
        #refresh=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=imageScale, keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=datasetype,
        dir=workDir+'valid',
        classes=classes,
        #refresh=True,
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=testScale, keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=datasetype,
        dir=workDir+'test',
        classes=classes,
        #refresh=True,
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=testScale, keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
val_evaluator = dict(
    type='CocoMetric',
    ann_file=workDir+'valid/cocofile.json',
    metric='bbox',
    format_only=False)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=workDir+'test/cocofile.json',
    metric='bbox',
    format_only=False)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=maxepochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
auto_scale_lr = dict(enable=False, base_batch_size=16)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=logger_interval),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1,max_keep_ckpts=-1,save_best='coco/bbox_mAP',
                rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',interval=10,draw=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
custom_hooks = [dict(type='DealModelPostProcess')]
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
randomness=dict(seed=1024,deterministic=True)
runner_type='mmdet.RunnerSplitGPU'
log_level = 'INFO'
load_from = '/data/projectEngine/MMDetecTVT/FineTune/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
resume = False
launcher = 'pytorch'
work_dir = 'workDir'
selfmark=dict(
    loadFrom=None,
    minthr=0.2,
    #maxthr=0.9,
    #mode='train',
    lowMaxNum=100,
    highf1score=1.0,
    recruback=-1,
    randomSep=0.35,
    iouthr=0.5)
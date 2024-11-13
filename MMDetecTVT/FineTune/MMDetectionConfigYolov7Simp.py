
workDir='/data/projectEngine/'
#############################################################
######################特别注意################################
#前两行预留，可以什么都不写，但是必须留出来，否则程序会覆盖前两行
####可以修改分数阈值，可以修改nms###############################
##############################################################
classNum=1
imageScale=(512,512)
testScale=(512,512)
halfScale=(-256,-256)
classes=['bicycle']
logger_interval=1
maxepochs=300
datasetype='EhlXMLYoloCocoDataSet'
default_scope = 'mmyolo'
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(norm_type=2,max_norm=35),
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=3),
    constructor='YOLOv7OptimWrapperConstructor')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=logger_interval),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.1,
        warmup_epochs = 1,
        warmup_bias_lr = 0.1,
        warmup_momentum = 0.8,
        warmup_mim_iter = 10,
        max_epochs=maxepochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,rule='greater',
        save_param_scheduler=False,
        save_best='coco/bbox_mAP',
        max_keep_ckpts=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = '/data/projectEngine/MMDetecTVT/FineTune/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth'
resume = False
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=1,
    img_size=testScale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)
randomness=dict(seed=1024,deterministic=True)
runner_type='RunnerSplitGPU'
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv7Backbone',
        arch='L',
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv7PAFPN',
        block_cfg=dict(
            type='ELANBlock',
            middle_ratio=0.5,
            block_ratio=0.25,
            num_blocks=4,
            num_convs_in_block=1),
        upsample_feats_cat_first=False,
        in_channels=[512, 1024, 1024],
        out_channels=[128, 256, 512],
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            num_classes=classNum,
            in_channels=[256, 512, 1024],
            featmap_strides=[8, 16, 32],
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[[(12, 16), (19, 36), (40, 28)],
                        [(36, 75), (76, 55), (72, 146)],
                        [(142, 110), (192, 243), (459, 401)]],
            strides=[8, 16, 32]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.3),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            reduction='mean',
            loss_weight=0.05,
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.7),
        obj_level_weights=[4.0, 1.0, 0.4],
        prior_match_thr=4.0,
        simota_candidate_topk=10,
        simota_iou_weight=3.0,
        simota_cls_weight=1.0),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))
pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True)
]
train_dataloader = dict(
    batch_size=3,
    num_workers=3,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type=datasetype,
        dir=workDir+'train',
        classes=classes,
        debugSep=1,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='RandomChoice',
                transforms=[[{
                    'type':
                    'Mosaic',
                    'img_scale': imageScale,
                    'pad_val':
                    114.0,
                    'pre_transform': [{
                        'type': 'LoadImageFromFile',
                        'file_client_args': {
                            'backend': 'disk'
                        }
                    }, {
                        'type': 'LoadAnnotations',
                        'with_bbox': True
                    }]
                }, {
                    'type': 'YOLOv5RandomAffine',
                    'max_rotate_degree': 0.0,
                    'max_shear_degree': 0.0,
                    'max_translate_ratio': 0.2,
                    'scaling_ratio_range': (0.1, 2.0),
                    'border': halfScale,
                    'border_val': (114, 114, 114)
                }],
                [{
                    'type':
                    'Mosaic9',
                    'img_scale': imageScale,
                    'pad_val':
                    114.0,
                    'pre_transform': [{
                        'type': 'LoadImageFromFile',
                        'file_client_args': {
                            'backend': 'disk'
                        }
                    }, {
                        'type': 'LoadAnnotations',
                        'with_bbox': True
                    }]
                }, {
                    'type': 'YOLOv5RandomAffine',
                    'max_rotate_degree': 0.0,
                    'max_shear_degree': 0.0,
                    'max_translate_ratio': 0.2,
                    'scaling_ratio_range': (0.1, 2.0),
                    'border': halfScale,
                    'border_val': (114, 114, 114)
                }]],
                prob=[0.8, 0.2]),
            dict(
                type='YOLOv5MixUp',
                alpha=8.0,
                beta=8.0,
                prob=0.15,
                pre_transform=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk')),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='RandomChoice',
                        transforms=[[{
                            'type':
                            'Mosaic',
                            'img_scale': imageScale,
                            'pad_val':
                            114.0,
                            'pre_transform': [{
                                'type': 'LoadImageFromFile',
                                'file_client_args': {
                                    'backend': 'disk'
                                }
                            }, {
                                'type': 'LoadAnnotations',
                                'with_bbox': True
                            }]
                        }, {
                            'type': 'YOLOv5RandomAffine',
                            'max_rotate_degree': 0.0,
                            'max_shear_degree': 0.0,
                            'max_translate_ratio': 0.2,
                            'scaling_ratio_range': (0.1, 2.0),
                            'border': halfScale,
                            'border_val': (114, 114, 114)
                        }],
                        [{
                            'type':
                            'Mosaic9',
                            'img_scale': imageScale,
                            'pad_val':
                            114.0,
                            'pre_transform': [{
                                'type': 'LoadImageFromFile',
                                'file_client_args': {
                                    'backend': 'disk'
                                }
                            }, {
                                'type': 'LoadAnnotations',
                                'with_bbox': True
                            }]
                        }, {
                            'type': 'YOLOv5RandomAffine',
                            'max_rotate_degree': 0.0,
                            'max_shear_degree': 0.0,
                            'max_translate_ratio': 0.2,
                            'scaling_ratio_range': (0.1, 2.0),
                            'border': halfScale,
                            'border_val': (114, 114, 114)
                        }]],
                        prob=[0.8, 0.2])
                ]),
            dict(type='YOLOv5HSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'flip', 'flip_direction'))
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=True,
    pin_memory=True,
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
            dict(type='YOLOv5KeepRatioResize', scale=testScale),
            dict(
                type='LetterResize',
                scale=testScale,
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        batch_shapes_cfg=batch_shapes_cfg))
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
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
            dict(type='YOLOv5KeepRatioResize', scale=testScale),
            dict(
                type='LetterResize',
                scale=testScale,
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        batch_shapes_cfg=batch_shapes_cfg))
param_scheduler = None
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=workDir+'valid/cocofile.json',
    metric='bbox')
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=workDir+'test/cocofile.json',
    metric='bbox')
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=maxepochs,
    val_interval=1,
    dynamic_intervals=[(270, 1)])
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(type='mmdet.DealModelPostProcess')
]
runner_type='mmdet.RunnerSplitGPU'
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
workDir = '/data/projectEngine/'
launcher = 'none'
work_dir = 'workDir'

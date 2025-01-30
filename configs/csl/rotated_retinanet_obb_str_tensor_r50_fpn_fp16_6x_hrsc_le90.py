_base_ = \
    ['../rotated_retinanet/rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90.py']

angle_version = 'le90'
model = dict(
    bbox_head=dict(
        type='STRRetinaHead',
        angle_coder=dict(
            type='STCoder',
            angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        loss_angle=dict(
            type='L1Loss', loss_weight=0.01)))

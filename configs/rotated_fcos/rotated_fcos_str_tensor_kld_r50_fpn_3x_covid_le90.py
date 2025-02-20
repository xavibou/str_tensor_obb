_base_ = '../psc/rotated_fcos_str_tensor_r50_fpn_3x_covid_le90.py'

angle_version = 'le90'
model = dict(
    bbox_head=dict(
        type='StructureTensorFCOSHead',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        h_bbox_coder=dict(type='DistancePointBBoxCoder'),
        angle_coder=dict(
            type='STCoder', 
            anisotropy=4,   # level of anisotropy or isotropy allowed
            angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            _delete_=True,
            type='GDLoss_v1',
            loss_type='kld',
            fun='log1p',
            tau=1,
            loss_weight=1.0),
        loss_angle=dict(type='L1Loss', loss_weight=0.05),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)), 
    )

    
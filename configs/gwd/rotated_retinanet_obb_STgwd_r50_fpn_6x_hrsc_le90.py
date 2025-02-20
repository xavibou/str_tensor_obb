_base_ = ['../rotated_retinanet/rotated_retinanet_ST_obb_r50_fpn_6x_hrsc_rr_le90.py']

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)))

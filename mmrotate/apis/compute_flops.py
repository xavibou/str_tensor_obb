import torch
from fvcore.nn import FlopCountAnalysis

def single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
    model.eval()

    # Create a dummy input for the image (adjust dimensions based on your data)
    input = torch.randn(1, 3, 224, 224, dtype=torch.float32).cuda()
    
    # Create dummy image metadata (this should match what your model expects)
    img_metas = [{
        'img_shape': (224, 224, 3),
        'scale_factor': 1.0,
        'flip': False,
        'filename': 'dummy.jpg',
        'ori_shape': (224, 224, 3),
        'pad_shape': (224, 224, 3),
        'flip_direction': None,
        'img_norm_cfg': {
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
            'to_rgb': True
        }
    }]

    # Ensure model is set to evaluation mode
    model.eval()

    gt_bboxes = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32).cuda()  # Example ground truth bounding boxes
    gt_labels = []

    # Create a dummy batch of data
    batch = (input, img_metas, gt_bboxes)

    # Compute FLOPs (for inference)
    flops = FlopCountAnalysis(model, batch)
    breakpoint()
    print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")  # GFLOPs = GigaFLOPs
    
    return results

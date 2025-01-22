

for i in {1..10}
do
    python tools/plot_data_from_dir.py \
        paper_experiments/h2rbox-v2_anisotropy_2_1_coder/h2rbox_v2p_r50_fpn_1x_dota_le90_str_tensor.py \
        paper_experiments/h2rbox-v2_anisotropy_2_1_coder/epoch_12.pth \
        demo/rotated_inputs/img$i \
        --out-dir demo/rotated_outputs/img$i
done

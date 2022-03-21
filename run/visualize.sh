# predition
model="siod_res18_dminer"
python ../src/demo.py ctdet \
--arch resdcn_18 \
--demo "/home/hanjun/outputs/siod/example/images1" \
--demo_save_name "$model" \
--load_model "/home/hanjun/data/SIOD/checkpoints/$model/model_last.pth"
# visualize
python ../src/utils/visualizer.py \
--imgs_path "/home/hanjun/outputs/siod/example/images1/" \
--result_file "/home/hanjun/outputs/siod/example/detect_result/$model.json" \
--prefix "dminer_s3" \
--thresh 0.3 \

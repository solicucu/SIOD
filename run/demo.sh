# detection
python ../src/demo.py ctdet \
--arch resdcn_18 \
--demo /data1/v_sungli/output/example/images \
--demo_save_name 'resdcn18_fsod' \
--load_model /data1/v_sungli/output/keyckpts/fsod/coco_resdcn18/model_last.pth 

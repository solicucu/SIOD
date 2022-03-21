# test
python ../src/visualize.py ctdet \
--exp_id coco_resdcn18_visulization \
--arch resdcn_18 \
--gpus 0 \
--not_rand_crop \
--no_color_aug \
--flip 0. \
--prefix 'keep1_' \
--sim_thresh 0.9 \
--use_plg \
--ret_copy \
--keep_res \
--vis_prefix "vis_thr09" \
--visualize_path "/home/hanjun/outputs/siod/example/images2/" \
--anno_path "../data/coco/annotations/keep1_instances_train2017.json" \
--load_model "/home/hanjun/data/SIOD/checkpoints/siod_res18_dminer/model_last.pth" \

# train
export CUDA_VISIBLE_DEVICES=4,5,6,7
# plg
python ../src/main.py ctdet \
--exp_id siod_res18_gcl2plg \
--arch resdcn_18 \
--batch_size 114 \
--master_batch 18 \
--lr 5e-5 \
--gpus 4,5,6,7 \
--num_workers 16 \
--prefix 'keep1_' \
--sim_thresh 0.6 \
--use_plg \
--load_model "/data1/v_sungli/output/siod/exp/ctdet/siod_res18_gcl/model_last.pth" \
# test
python ../src/test.py ctdet \
--exp_id siod_res18_gcl2plg \
--arch resdcn_18 \
--keep_res \
--resume

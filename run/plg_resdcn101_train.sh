# train 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python ../src/main.py ctdet \
--exp_id siod_res101_plg \
--arch resdcn_101 \
--batch_size 96 \
--master_batch 5 \
--lr 3.75e-4 \
--gpus 0,1,2,3,4,5,6,7 \
--num_workers 16 \
--save_all \
--prefix 'keep1_' \
--use_plg \
--sim_thresh 0.6 \
# test
python ../src/test.py ctdet \
--exp_id siod_res101_plg \
--arch resdcn_101 \
--keep_res \
--resume

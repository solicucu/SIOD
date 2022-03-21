# train
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python ../src/main.py ctdet \
--exp_id siod_res101_gcl \
--arch resdcn_101 \
--batch_size 96 \
--master_batch 5 \
--lr 3.75e-4 \
--gpus 0,1,2,3,4,5,6,7 \
--num_workers 16 \
--save_all \
--prefix 'keep1_' \
--use_gcl \
--gcl_loss_mask \
# test
python ../src/test.py ctdet \
--exp_id siod_res101_gcl \
--arch resdcn_101 \
--keep_res \
--resume
# plg
python ../src/main.py ctdet \
--exp_id siod_res101_gcl2plg \
--arch resdcn_101 \
--batch_size 96 \
--master_batch 5 \
--lr 3.75e-5 \
--gpus 0,1,2,3,4,5,6,7 \
--num_workers 16 \
--save_all \
--keep 'keep1_' \
--sim_thresh 0.6 \
--use_plg \
--load_model '/data1/v_sungli/output/siod/exp/ctdet/siod_res101_gcl/model_last.pth' \
# test
python ../src/test.py ctdet \
--exp_id siod_res101_gcl2plg \
--arch resdcn_101 \
--keep_res \
--resume

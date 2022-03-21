# train
export CUDA_VISIBLE_DEVICES=4,5,6,7
python ../src/main.py ctdet \
--exp_id siod_res18_gcl \
--arch resdcn_18 \
--save_all \
--batch_size 114 \
--master_batch 18 \
--lr 5e-4 \
--gpus 4,5,6,7 \
--num_workers 16 \
--prefix 'keep1_' \
--use_gcl \
--gcl_loss_mask \
# test
python ../src/test.py ctdet \
--exp_id siod_res18_gcl \
--arch resdcn_18 \
--keep_res \
--resume

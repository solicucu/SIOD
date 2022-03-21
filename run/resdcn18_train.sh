# train
export CUDA_VISIBLE_DEVICES=0,1,2,3
python ../src/main.py ctdet \
--exp_id fsod_res18 \
--arch resdcn_18 \
--save_all \
--batch_size 114 \
--master_batch 18 \
--lr 5e-4 \
--gpus 0,1,2,3 \
--num_workers 16 \
--prefix '' \
# test
python ../src/test.py ctdet \
--exp_id fsod_res18 \
--arch resdcn_18 \
--keep_res \
--resume

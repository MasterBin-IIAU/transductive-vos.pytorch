# inference
CUDA_VISIBLE_DEVICES=1 python inference.py -r davis_train.pth.tar -s predictions_rep --data .
# Compute metrics
python /data/sda/v-yanbi/cache/davis2017-evaluation/evaluation_method.py \
--task semi-supervised  --results_path ./predictions_rep/ --set val \
--davis_path /data/sdg/tracking_datasets/DAVIS/
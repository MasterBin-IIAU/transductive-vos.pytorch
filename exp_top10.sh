# inference
# 速度很慢，目前不太确定为什么
CUDA_VISIBLE_DEVICES=5 python inference.py -r davis_train.pth.tar -s predictions_top10 --data . --topk 10
# Compute metrics
python /data/sda/v-yanbi/cache/davis2017-evaluation/evaluation_method.py \
--task semi-supervised  --results_path ./predictions_top10/ --set val \
--davis_path /data/sdg/tracking_datasets/DAVIS/
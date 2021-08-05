# inference
python inference.py -r davis_train.pth.tar -s predictions_ref20_dense10 --data . --ref_num 20 --dense_num 10
# Compute metrics
python /data/sda/v-yanbi/cache/davis2017-evaluation/evaluation_method.py \
--task semi-supervised  --results_path ./predictions_ref20_dense10/ --set val \
--davis_path /data/sdg/tracking_datasets/DAVIS/
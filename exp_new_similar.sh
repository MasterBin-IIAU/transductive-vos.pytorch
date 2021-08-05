# inference
python inference.py -r davis_train.pth.tar -s predictions_new_simi --data . --new_similar 1
# Compute metrics
python /data/sda/v-yanbi/cache/davis2017-evaluation/evaluation_method.py \
--task semi-supervised  --results_path ./predictions_new_simi/ --set val \
--davis_path /data/sdg/tracking_datasets/DAVIS/
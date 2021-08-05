# inference
python inference.py -r davis_train.pth.tar -s predictions --data .
# Compute metrics
python /data/sda/v-yanbi/cache/davis2017-evaluation/evaluation_method.py \
--task semi-supervised  --results_path ./predictions/ --set val \
--davis_path /data/sdg/tracking_datasets/DAVIS/
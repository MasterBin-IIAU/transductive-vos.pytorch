# inference
python inference.py -r davis_train.pth.tar -s predictions_add1 --data . --add_init 1
# Compute metrics
python /data/sda/v-yanbi/cache/davis2017-evaluation/evaluation_method.py \
--task semi-supervised  --results_path ./predictions_add1/ --set val \
--davis_path /data/sdg/tracking_datasets/DAVIS/
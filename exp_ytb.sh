# inference
python inference.py -r youtube_train.pth.tar -s predictions_ytb --data .
# Compute metrics
python /data/sda/v-yanbi/cache/davis2017-evaluation/evaluation_method.py \
--task semi-supervised  --results_path ./predictions_ytb/ --set val \
--davis_path /data/sdg/tracking_datasets/DAVIS/
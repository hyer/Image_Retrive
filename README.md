0. Split the dataset into 1:9 by [dataset_div.py];
1. Fine-tuning model for the specified datasets; Get acc on validation set; [train_start.sh]
2. extract feat of register images; [gen_feats_db.py]
2. launch server on 19.34; Using KNN=1 for top-100;



# optimize
1. flip;
2. KTree;
3. Cluster;
4. Directly FlowerR's softmax -- /home/hyer/workspace/algo/FlowerR/demo.py

import pickle

chk_file = "./models/LightCNN_29Layers_V2_checkpoint.pth.tar"
label_file = "./data/names.txt"

print "==> Load labels."
with open(label_file, "r") as f_l:
    data = f_l.readlines()

labels = []
for l in data:
    labels.append(l.strip())

print "==> Load feats."
with open("./data/lcnn29v2_mj.pkl", "r") as f:
    feats_data = pickle.load(f)

feats = []
names = []
for i in feats_data:
    names.append(i[0])
    feats.append(i[1])


# cal top-1 and top5 acc with knn

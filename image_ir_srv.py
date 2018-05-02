#coding=utf-8
import time

import tornado.ioloop
import tornado.web
import tornado.httpserver
from torch.autograd import Variable
from tornado.options import define, options
import json

import os
import torch
import pickle
import sys
from collections import OrderedDict
import sklearn
from PIL import ImageFont, Image, ImageDraw
from sklearn.decomposition import PCA

sys.path.append("../")

import torchvision.transforms as transforms

import time
import cv2
import numpy as np
import mxnet as mx

port = 9800
conf_thres = 0.5

model = None

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

def init_server(chk_file):
    global model




def get_feat(model, img_cv2, bbox, landmarks, use_cuda=True):

    if bbox is None:
        print("Failed to get face bbox.")
        return None
    if landmarks is None:
        print("Failed to get face landmarks.")
        return None

    if bbox.shape[0] == 0:
        return None
    bbox = bbox[0, 0:4]
    landmarks = landmarks[0, :].reshape((2, 5)).T
    # print(bbox)
    # print(points)
    nimg = face_preprocess.preprocess(img_cv2, bbox, landmarks, image_size='128,128')
    # cv2.imwrite("tmp.jpg", nimg)

    # img = cv2.imread("tmp.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(nimg, cv2.COLOR_BGR2GRAY)
    img = np.reshape(img, (128, 128, 1))
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    input = torch.zeros(1, 1, 128, 128)
    input[0, :, :, :] = img

    start = time.time()
    if use_cuda:
        input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    _, features = model(input_var)

    if use_cuda:
        feature = features.data.cpu().numpy()
    else:
        feature = features.data.numpy()
    feature = sklearn.preprocessing.normalize(feature).flatten()
    return feature, nimg



def get_n_top(X_tr, Y_tr, x, top_n=1):
    num_train = X_tr.shape[0]
    A = np.tile(x, (num_train, 1))
    B = X_tr

    tmp1 = np.sum(A * B, axis=1)
    tmp2 = (np.sum(A ** 2, axis=1) ** 0.5) * (np.sum(B ** 2, axis=1) ** 0.5)
    similarity = tmp1 / (tmp2 + 1e-12)

    sim_list = similarity.tolist()
    sim_np = np.array(sim_list)
    sim_sorted_idx = np.argsort(-sim_np)  # descend

    person_name_score_list = []
    sim_list = list(sim_list)
    for i in range(sim_np.size):
        person_name_score_list.append((Y_tr[sim_sorted_idx[i]], sim_list[sim_sorted_idx[i]]))
    # sim_label_list = zip(sim_list, Y_tr)
    #
    # sim_label_list_sorted = sorted(sim_label_list, key=lambda x: x[0], reverse=True)

    return person_name_score_list


def recognition_identify(feat, n_top, feat_list, person_name_list):
    response = {}

    X_tr = np.array(feat_list)
    y_tr = np.array(person_name_list)

    # print('predict......')

    # feat = deep_net.extract_face_feat_legacy(img_file_name)
    person_name_score_list = get_n_top(X_tr, y_tr, feat)

    candidate_list = []

    if person_name_score_list is not None:

        min_n_top = min(n_top, len(person_name_score_list))

        person_name_set = set([])
        for person_name, score in person_name_score_list:

            if (person_name not in person_name_set) and len(person_name_set) < min_n_top:
                candidate = {'confidence': score, 'person_name': person_name}
                candidate_list.append(candidate)
                person_name_set.add(person_name)
            elif len(person_name_set) == min_n_top:
                break

        response['candidate'] = candidate_list
        response['error_code'] = 0

    return response


def save_image_str(img_str, img_save_name):
    img_data = np.fromstring(img_str, dtype='uint8')
    decimg = cv2.imdecode(img_data, 1)
    cv2.imwrite( img_save_name, decimg )


define("port", default=port, help='run a test')
class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        """
        Returns:

        """
        # print "setting headers!!!"
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')


    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self, *args, **kwargs):
        save_home = "./images"
        save_file_name = ""
        if "img_str" in self.request.arguments:
            img_str_data = self.request.arguments["img_str"][0]
            save_file_name = time.strftime('%Y-%m-%d %H%M%S', time.localtime(time.time())) + ".jpg"
            save_image_str(img_str_data, save_home +  "/" + save_file_name)

        if "up_img" in self.request.files:
            imgfile = self.request.files.get('up_img')
            img_post = imgfile[0].filename.split(".")[-1]
            save_file_name = time.strftime('%Y-%m-%d %H%M%S', time.localtime(time.time())) + img_post
            with open(save_home +  "/" + save_file_name, 'wb') as f:
                f.write(imgfile[0]['body'])

        response = {}
        img_path =save_home +  "/" + save_file_name
        if not os.path.exists(img_path):
            response["is_face"] = False
            response["success"] = 0
        else:
            img = cv2.imread(img_path)
            bbox, landmarks = detector.detect_face(img)
            if bbox is not None:
                for i, box in enumerate(bbox):
                    feat, nimg = get_feat(fr_m, img, np.array([np.array(box)]), np.array([np.array(landmarks[i])]))
                    # cv2.imshow("capture", nimg)
                    # cv2.waitKey()
                    res = recognition_identify(feat, 1, feats, names)
                    print res
                    # print "recog time: ", time.time() - t1
                    if res["error_code"] == 0:
                        person_name = res["candidate"][0]["person_name"]
                        confidence = res["candidate"][0]["confidence"]
                        # stu_name = stu_names[int(person_name.split("N")[-1]) - 1]
                        if confidence > conf_thres:
                            response["person_name"] = person_name
                            response["confidence"] = confidence
                            response["is_face"] = True
                            response["success"] = 1
                        else:
                            response["is_face"] = True
                            response["confidence"] = confidence
                            response["success"] = 0
                    else:
                        response["is_face"] = True
                        # response["confidence"] = confidence
                        response["success"] = 0
            else:
                response["is_face"] = False
                response["success"] = 0

        self.finish(json.dumps(response))



application = tornado.web.Application([
    (r"/facer/", MainHandler),

])

if __name__ == "__main__":
    print("init server")
    init_server(chk_file)
    application.listen(port)
    print("Srv started at %d." % port)
    tornado.ioloop.IOLoop.instance().start()
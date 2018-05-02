# -*- coding: utf-8 -*-

import os


if __name__ == '__main__':
    with open("../data/register.txt", "w") as f:
        img_root = "/home/hyer/datasets/class01_small"

        dirs = os.listdir(img_root)
        for dir in dirs:
            img_names = os.listdir(img_root + "/" + dir)
            for img_name in img_names:
                f.write(dir + " " + img_root + "/" + dir + "/" + img_name + "\n")
    print "done."
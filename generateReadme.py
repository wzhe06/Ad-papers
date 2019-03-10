#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import urllib

""" generate readme.md """
__author__ = 'Wang Zhe'

paper_class_map = {}
paper_map = {}

file_object = open('./README.md')
all_lines = file_object.readlines()
file_object.close()

out_file = open('./README.md', 'w')

paper_class_flag = 0
paper_class_name = ""
paper_flag = 0
paper_name = ""
catalog_flag = 0

for line in all_lines:
    if catalog_flag != 1:
        out_file.write(line)
    if line.startswith("##"):
        catalog_flag = 1

    if paper_class_flag == 1 and not line.startswith("*") and not line.startswith("#"):
        paper_class_map[paper_class_name] = line.strip()
        print paper_class_name, line.strip()
    paper_class_flag = 0

    if paper_flag == 1 and not line.startswith("*") and not line.startswith("#"):
        paper_map[paper_name] = line.strip()
        print "\t", paper_name, line.strip()

    paper_flag = 0

    if catalog_flag == 1:
        if line.startswith("*"):
            paper_flag = 1
            paper_name = line[line.find("[")+1:line.find("]")].strip()

    if line.startswith("###"):
        paper_class_flag = 1
        paper_class_name = line[3:].strip()

github_root = "https://github.com/wzhe06/Ad-papers/blob/master/"
all_dir = os.listdir("./")
for one_dir in all_dir:
    if os.path.isdir(one_dir) and not one_dir.startswith('.'):
        out_file.write("\n### " + one_dir+"\n")
        if one_dir.strip() in paper_class_map:
            out_file.write(paper_class_map[one_dir.strip()] + "\n")
        all_sub_files = os.listdir(one_dir)
        for one_file in all_sub_files:
            if not os.path.isdir(one_file) and not one_file.startswith('.'):
                out_file.write("* [" + ('.').join(one_file.split('.')[:-1]) + "]("+github_root + urllib.quote(one_dir.strip())+"/"
                               + urllib.quote(one_file.strip())+") <br />\n")
                if one_file.strip() in paper_map:
                    out_file.write(paper_map[one_file.strip()] + "\n")

out_file.close()

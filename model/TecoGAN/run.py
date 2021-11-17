import os, subprocess, sys, datetime, signal, shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="path to input video")
parser.add_argument("--output", type=str, required=True, help="path to output video")

def preexec():
        os.setpgrp()

def mycall(cmd, block=False):
        if not block:
                return subprocess.Popen(cmd)
        else:
                return subprocess.Popen(cmd, preexec_fn = preexec)

def folder_check(path):
        try_num = 1
        oripath = path[:-1] if path.endswith('/') else path
        while os.path.exists(path):
                print("Delete existing folder " + path + "?(Y/N)")
                decision = input()
                if decision == "Y":
                        shutil.rmtree(path, ignore_errors=True)
                        break
                else:
                        path = oripath + "_%d/"%try_num
                        try_num += 1
                        print(path)

        return path

args = parser.parse_args()

dirstr = args.output
input_dir = args.input
if (not os.path.exists(dirstr)): os.makedirs(dirstr)
cmd1 = ["python", sys.path[0] +"/main.py",
        "--output_dir",  dirstr,
        "--summary_dir", os.path.join(dirstr, 'log/'),
        "--input_dir_LR", input_dir,
        "--checkpoint", sys.path[0] +'/model/TecoGAN',
        ]
mycall(cmd1).communicate()

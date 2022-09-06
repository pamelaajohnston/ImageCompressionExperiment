import numpy as np
import os
import random
import shutil
import cv2
import re
import argparse
import sys
from PIL import Image


def createFileList(myDir, formats=['.tif', '.png', '.tiff', '.jpg', 'jpeg']):
    fileList = []
    #print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            for format in formats:
                if name.endswith(format):
                    fullName = os.path.join(root, name)
                    fileList.append(fullName)
    return fileList

def makeFreshDir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)





if __name__ == "__main__":
    source = "UCIDPatches/"
    destDir = "UCIDsinglecompressionDataSet"
    testSplit = 15 #out of 100
    # JPEG quality settings
    high_q = 95
    low_q = 70
    step_q = 5
    doSecondCompression = False
 
    parser = argparse.ArgumentParser(description="Takes a folder of images and creates patches from them")
    parser.add_argument("-s", "--source", help="the source directory")
    parser.add_argument("-d", "--dest",   help="the destination directory (will make it if it doesn't exist, wipe it if it does)")
 
    args = parser.parse_args()

    if args.source:
        print("Getting pictures from {}".format(args.source))
        source = args.source
    if args.dest:
        destDir = args.dest
        print("Storing patch pictures to {}.".format(args.dest))
        makeFreshDir(destDir)

    #Note that first you should patch the images using patchRGB.py (we're using UCID)
    imageNames = createFileList(source)

    #Make the test and train dirs:
    testDir = os.path.join(destDir, "test")
    trainDir = os.path.join(destDir, "train")

    uncompTestDir = os.path.join(testDir, "uncompressed")
    uncompTrainDir = os.path.join(trainDir, "uncompressed")
    comp1TestDir = os.path.join(testDir, "comp1")
    comp1TrainDir = os.path.join(trainDir, "comp1")
    comp2aTestDir = os.path.join(testDir, "comp2a")
    comp2aTrainDir = os.path.join(trainDir, "comp2a")
    comp2bTestDir = os.path.join(testDir, "comp2b")
    comp2bTrainDir = os.path.join(trainDir, "comp2b")
    makeFreshDir(uncompTestDir)
    makeFreshDir(uncompTrainDir)
    makeFreshDir(comp1TestDir)
    makeFreshDir(comp1TrainDir)
    makeFreshDir(comp2aTestDir)
    makeFreshDir(comp2aTrainDir)
    makeFreshDir(comp2bTestDir)
    makeFreshDir(comp2bTrainDir)

    for imageName in imageNames:
        #test or train
        testOrTrain = random.randrange(100)
        if testOrTrain < testSplit:
            uncompDir = uncompTestDir
            comp1Dir = comp1TestDir
            comp2aDir = comp2aTestDir
            comp2bDir = comp2bTestDir
        else:
            uncompDir = uncompTrainDir
            comp1Dir = comp1TrainDir
            comp2aDir = comp2aTrainDir
            comp2bDir = comp2bTrainDir

        imageBaseName = os.path.splitext(os.path.basename(imageName))[0]
        uncompName = os.path.join(uncompDir, imageBaseName)
        comp1Name = os.path.join(comp1Dir, imageBaseName)
        comp2aName = os.path.join(comp2aDir, imageBaseName)
        comp2bName = os.path.join(comp2bDir, imageBaseName)


        im = Image.open(imageName)
        
        #Uncompressed
        uncompName = "{}_uncomp.jpeg".format(uncompName)
        im.save(uncompName, quality=100)

        q = random.choice(range(low_q, high_q, step_q))
        q=70
        comp1Name = "{}_q{}.jpeg".format(comp1Name, q)
        im.save(comp1Name, quality=q)

        if doSecondCompression:
            im_comp = Image.open(comp1Name)

            if q == high_q:
                qa = high_q
            else:
                qa = random.choice(range(q, high_q, step_q))
            
            comp2aName = "{}_q{}_q2{}.jpeg".format(comp2aName, q, qa)
            im_comp.save(comp2aName, quality=qa)

            if q == low_q:
                qb = low_q
            else: 
                qb = random.choice(range(low_q, q, step_q))
            comp2bName = "{}_q{}_q2{}.jpeg".format(comp2bName, q, qb)
            im_comp.save(comp2bName, quality=qb)


        
 
 


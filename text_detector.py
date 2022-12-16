#import cPickle
from utils import *
from darknet import Darknet
import os
from tqdm import tqdm
import cv2
# Detect if there is text present in an image and store it

def detect(cfgfile, weightfile, imgfolder, text_file):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    namesfile = 'data/recognition.names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    class_names = load_class_names(namesfile)
    image_list = os.listdir(imgfolder)

    #words, neigh = KNNclassifier()
    total_seen = 0
    total_text = 0

    with open(text_file, 'a') as f:

        for imgfile in tqdm(image_list):
            total_seen += 1
            img_full_path = imgfolder + imgfile
            try:
                img = Image.open(img_full_path).convert('RGB')
            except IOError:
                #os.remove(img_full_path)
                print("Error Opening: %s ... Continuing next image" %(img_full_path))
                continue

            sized = img.resize((m.width, m.height))

            conf_threshold = 0.3
            # ORIGINAL
            nms_threshold = 0.7

            # TEST  with better results
            # nms_threshold = 0.2
            img = cv2.imread(img_full_path)
            for i in range(1):
                start = time.time()
                boxes = do_detect(m, sized, conf_threshold, nms_threshold, use_cuda)
                finish = time.time()
                #print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
                for box in boxes:
                    x = box[0].item()
                    y = box[1].item()
                    w = box[2].item()
                    h = box[3].item()
                    x = x - w/2
                    y = y - h/2
                    cv2.rectangle(img, (int(img.shape[1]*x),int(img.shape[0]*y)),  (int(img.shape[1]*(x+w)),int(img.shape[0]*(y+h))),(255, 0, 255) , 2)
                cv2.imshow("test", img) 
                cv2.waitKey(0)
            # RESULT PATH
            #result_image_path = destination_folder + imgfile

            if len(boxes) >= 1:
                total_text += 1
                # print("Text found on %s .... Images: %d/%d " % (imgfile, total_text, total_seen))
                f.write(str(imgfile)+'\n')

            #os.remove(img_full_path)
    f.close()


        # Copy images if there is text:
        # count_boxes(img_full_path, boxes, result_image_path)



import torch
import torch.nn as nn
if __name__ == '__main__':
    imgfolder = "C:\\Users\\bliu\\Pictures\\test\\"
    text_file = "C:\\Users\\bliu\\Pictures\\testImages_with_Text.txt"
    model = torch.nn.Sequential()
    model.add_module('conv{0}'.format(1), nn.Conv2d(3, 32, 3, 2, 1, bias=False))
    model.add_module('bn{0}'.format(1), nn.BatchNorm2d(32))
    model.add_module('leaky{0}'.format(1), nn.LeakyReLU(0.1, inplace=True))
    test = torch.rand(1,3,608,608)
    model(test)
    #if not os.path.exists(destination_folder):
    #   os.mkdir(destination_folder)

    cfgfile = 'cfg/yolo-recognition-13anchors.cfg'
    weightfile = 'backup/000041.weights'

    detect(cfgfile, weightfile, imgfolder, text_file)
    print ("OPERATION COMPLETE..!!")


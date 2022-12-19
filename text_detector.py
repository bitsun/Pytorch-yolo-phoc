from utils import *
from darknet import Darknet
import os,glob
from tqdm import tqdm
import cv2
import argparse
# This script does text localization, in addition it also extracts a 604 dimensional vector descriptor for each detected text

def detect(cfgfile, weightfile, image_files,conf_threshold,nms_threshold):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    # for n in range(len(m.anchors)):
    #     m.anchors[n] = 320.0*m.anchors[n]/608
    print('Loading weights from %s... Done!' % (weightfile))

    namesfile = 'data/recognition.names'

    use_cuda = 1
    
    class_names = load_class_names(namesfile)
    m.eval()
    if use_cuda:
        m.cuda()
    start = time.time()
    for imgfile in tqdm(image_files):
        try:
            img = cv2.imread(imgfile)
        except IOError:
            #os.remove(img_full_path)
            print("Error Opening: %s ... Continuing next image" %(imgfile))
            continue

        sized = cv2.resize(img,(m.width,m.height))

        #conf_threshold = 0.3
        #nms_threshold = 0.7
        
        boxes = do_detect(m, sized, conf_threshold, nms_threshold, use_cuda)
        finish = time.time()
        for box in boxes:
            x = box[0].item() # x,y is the center coordinate of detected text bbox
            y = box[1].item()
            w = box[2].item()
            h = box[3].item()
            x = x - w/2
            y = y - h/2
            cv2.rectangle(img, (int(img.shape[1]*x),int(img.shape[0]*y)),  (int(img.shape[1]*(x+w)),int(img.shape[0]*(y+h))),(255, 0, 255) , 2)
        cv2.imshow("test", img) 
        cv2.waitKey(0)
    finish = time.time()
    print(finish-start)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',help="this is the input image file or input image folder",required=True)
    parser.add_argument('--model_file',help="this is the model file",required=True)
    parser.add_argument('--conf_th',help="this is the threshold for text box detection confidence,default is 0.9 from the original author, but 0.3 seems to better",required=False,default=0.9)
    parser.add_argument('--nms_th',help="this is the threshold for text boxes suppresion,default is 0.8",required=False,default=0.8) 
    args = parser.parse_args()
    if not os.path.exists(args.input):
        print(args.input_file,' does not exist')
        sys.exit()
    if not os.path.exists(args.model_file):
        print(args.model_file,'model file does not exist')
        sys.exit()
    image_files = []
    if os.path.isfile(args.input):
        image_files.append(args.input)
    elif os.path.isdir(args.input):
        file_extensions = ['*.jpg','*.png']
        for ext in file_extensions:
            image_files.extend(glob.glob(os.path.join(args.input,ext)))

    cfgfile = 'cfg/yolo-recognition-13anchors.cfg'
    weightfile = 'backup/000041.weights'

    detect(cfgfile, weightfile, image_files, conf_threshold=float(args.conf_th),nms_threshold=float(args.nms_th))
    print ("OPERATION COMPLETE..!!")


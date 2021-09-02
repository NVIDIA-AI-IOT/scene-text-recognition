import argparse
import os
import easyocr
import cv2

if __name__ == '__main__':
    #Arg parse and setup
    parser = argparse.ArgumentParser(description="EasyOCR Label Images")
    parser.add_argument('image',type=str, help='path to input image or directory of images')
    parser.add_argument('-t', '--trt', default=False, type=bool, help='accelerates detection and recognition models by converting them to TensorRT')
    args = parser.parse_args()
    if os.path.isfile(args.image):
        images = [args.image]
    else: #if it's not a file, assume it's a directory of images
        images = [os.path.join(args.image, file) for file in filter(lambda x: not x.endswith('.ipynb_checkpoints'),os.listdir(args.image))]
    
    #intialize output directory
    out_directory = args.image.split('/')[-1].split('.')[0]+'-labeled-images'
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    txt_file = open(out_directory+'/image_labels.txt','w')
    print('output directory:',out_directory)

    #load the scene text ocr models
    reader = easyocr.Reader(['en'], use_trt=args.trt) # need to run only once to load model into memory


    for image in images:
        #use cv2 to check if it is a valid image
        imageData = cv2.imread(image)
        if imageData is None:
            print("reading image %s failed" % image)
            continue

        #perform inference and read the models
        print("on image",image)
        txt_file.write(image+'\n')
        result = reader.readtext(image)
        print('result',result, '\n')
        
        #draw bounding boxes and ouptut result to txt file
        color = (0,0,255)
        imageData = cv2.imread(image)
        imgHeight, imgWidth, _ = imageData.shape
        thick = 2
        font_scale = 1
        for res in result:
            top_left, btm_right = res[0][0],res[0][2]
            to_int = lambda items: [int(x) for x in items]
            top_left = to_int(top_left)
            btm_right = to_int(btm_right)
            label = res[1]
            cv2.rectangle(imageData,top_left, btm_right, color, thick)
            cv2.putText(imageData, label, (top_left[0], top_left[1] - 12), 0, font_scale, color, thick)
            txt_file.write(str(res)+'\n')
        txt_file.write('\n')
        
        #write image and notify user
        check = cv2.imwrite(out_directory+"/labeled_"+image.split('/')[-1], imageData)
        if check:
            print("successfully wrote image:",out_directory+"/labeled_"+image.split('/')[-1])
        else:
            print("failed to write image:",out_directory+"/labeled_"+image.split('/')[-1])

    txt_file.close()
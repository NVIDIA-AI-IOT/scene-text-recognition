"""
This program uses EasyOCR to read a file or directory of files and output a labeled image. The output is in the labeled-images/ directory
"""
import argparse
import os
import easyocr
import cv2

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="EasyOCR Label Images")
    parser.add_argument('image',type=str, help='path to input image or directory of images')
    args = parser.parse_args()
    
    if os.path.isfile(args.image):
        images = [args.image]
    else:
        images = [os.path.join(args.image, file) for file in filter(lambda x: not x.endswith('.ipynb_checkpoints'),os.listdir(args.image))]
        
    for image in images:
        print("on image",image)
        reader = easyocr.Reader(['en']) # need to run only once to load model into memory
        result = reader.readtext(image)

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
        
        if not os.path.exists('labeled-images'):
            os.makedirs('labeled-images')
            
        check = cv2.imwrite("labeled-images/labeled_"+image.split('/')[-1], imageData)
        if check:
            print("successfully wrote image:","labeled-images/labeled_"+image.split('/')[-1])
        else:
            print("failed to write image:","labeled-images/labeled_"+image.split('/')[-1])

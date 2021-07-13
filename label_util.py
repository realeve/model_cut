
import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image

img_width = 180
img_height = 120

handleimgType = "fake_image"

CROP_IMG_SIZE = 112

FAKE_IMG_DIR = 'g:\\data\\'
OUTPUT_DIR = 'g:\\data_train_202001-202106\\'


# FAKE_IMG_DIR = 'g:\\test2\\'+handleimgType+'\\'
# OUTPUT_DIR = 'g:\\test3\\'+handleimgType+'\\'

def flip180(url):
    im = Image.open(url)
    out = im.transpose(Image.ROTATE_180)
    # out.save(url)
    img =cv2.cvtColor(np.asarray(out),cv2.COLOR_RGB2BGR)
    return img

def voc_get_image_info(annotation_root):
    filename = annotation_root.findtext('filename')
    assert filename is not None
    img_name = os.path.basename(filename)

    size = annotation_root.find('size')
    width = float(size.findtext('width'))
    height = float(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'img_name': img_name
    }
    return image_info


def rewriteXML(xml_file, xml_output_url, pt, size):
    x1, y1, x2, y2 = pt
    w, h = size
    tree = ET.parse(xml_file)

    tree.find('size').find('width').text = str(w)
    tree.find('size').find('height').text = str(h)

    bndbox = tree.find('object').find('bndbox')

    # print(pt, size)

    bndbox.find('xmin').text = str(x1)
    bndbox.find('ymin').text = str(y1)
    bndbox.find('xmax').text = str(x2)
    bndbox.find('ymax').text = str(y2)

    tree.write(xml_output_url, encoding='utf-8', xml_declaration=True)


def getBlob(xml_file):
    tree = ET.parse(xml_file)
    route = int(tree.find('blob').find('route').text)
    px1 = int(tree.find('blob').find('px1').text)
    px2 = int(tree.find('blob').find('px2').text)
    py1 = int(tree.find('blob').find('py1').text)
    py2 = int(tree.find('blob').find('py2').text)
    return route, (px1, py1, px2, py2)


def xml(xml_file):
    tree = ET.parse(xml_file)
    ann_root = tree.getroot()

    # img_info = voc_get_image_info(ann_root)
    # print(img_info)

    objs = tree.findall('object')
    im_w = float(tree.find('size').find('width').text)
    im_h = float(tree.find('size').find('height').text)

    filename = ann_root.findtext('filename')
    assert filename is not None
    img_name = os.path.basename(filename)

    if im_w < 0 or im_h < 0:
        print(
            'Illegal width: {} or height: {} in annotation, '
            'and {} will be ignored'.format(im_w, im_h, xml_file))

    gt_bbox = []
    gt_class = []
    class_name = []

    cname2cid = {
        'fake_image': 0,
        'normal_image': 1
    }

    for i, obj in enumerate(objs):
        cname = obj.find('name').text

        x1 = float(obj.find('bndbox').find('xmin').text)
        y1 = float(obj.find('bndbox').find('ymin').text)
        x2 = float(obj.find('bndbox').find('xmax').text)
        y2 = float(obj.find('bndbox').find('ymax').text)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(im_w - 1, x2)
        y2 = min(im_h - 1, y2)

        x2 = im_w if x2<0 else x2
        y2 = im_h if y2<0 else y2

        if abs(int(x2)-int(x1))<3:
            x1 = x1-5
            x2 = x2+5

        if abs(int(y2)-int(y1))<3:
            y1 = y1-5
            y2 = y2+5

        if x2 > x1 and y2 > y1:
            gt_bbox.append([x1, y1, x2, y2])
            class_name.append([cname])
            gt_class.append([cname2cid[cname]])
        else:
            print(
                'Found an invalid bbox in annotations: xml_file: {}'
                ', x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                    xml_file, x1, y1, x2, y2))
    gt_bbox = np.array(gt_bbox).astype('int32')
    gt_class = np.array(gt_class).astype('int32')
    class_name = np.array(class_name).astype('str')

    voc_rec = {
        'im_file': img_name,
        'h': im_h,
        'w': im_w,
        'gt_bbox': gt_bbox,
        'gt_class': gt_class
    }

    return voc_rec


def getBlackImg(CROP_IMG_SIZE=112):
    return np.zeros([CROP_IMG_SIZE, CROP_IMG_SIZE, 3], dtype=np.uint8)


def getCombineImg(CROP_IMG_SIZE=112):
    return np.zeros([CROP_IMG_SIZE, CROP_IMG_SIZE*2, 3], dtype=np.uint8)

# 图片横向扩充


def imgCombine(img1, img2, CROP_IMG_SIZE=112):
    combine = np.zeros([CROP_IMG_SIZE, CROP_IMG_SIZE*2, 3], dtype=np.uint8)
    combine[0:CROP_IMG_SIZE, 0:CROP_IMG_SIZE] = img1
    h,w = img2.shape[:2]    
    combine[0:min(CROP_IMG_SIZE,h), CROP_IMG_SIZE:min(w+CROP_IMG_SIZE,2*CROP_IMG_SIZE)] = img2
    return combine


def showImg(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


""" 匹配模板图像
    template_rgb: 待检图像数据
    model_url: 模板图像url
"""


def imgMatchWithROI(template_rgb, model_img):

    img_rgb = model_img.copy()

    img2 = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


    template = cv2.cvtColor(template_rgb.copy(), cv2.COLOR_BGR2GRAY)
    img3 = img2.copy()
    tmp = template.copy()

    # 匹配

    res = cv2.matchTemplate(img3, tmp, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 使用不同的比较方法，对结果的解释不同
    top_left = max_loc

    w, h = template.shape[::-1]

    image_crop = img_rgb[top_left[1]:top_left[1] +
                         h, top_left[0]:top_left[0] + w]
    return image_crop


def imgMatch(template_rgb, model_url):
    img = cv2.imread(model_url)
    img_rgb = img.copy()

    img2 = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    template = cv2.cvtColor(template_rgb.copy(), cv2.COLOR_BGR2GRAY)
    img3 = img2.copy()
    tmp = template.copy()

    # 匹配
    res = cv2.matchTemplate(img3, tmp, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 使用不同的比较方法，对结果的解释不同
    top_left = max_loc

    w, h = template.shape[::-1]

    image_crop = img_rgb[top_left[1]:top_left[1] +
                         h, top_left[0]:top_left[0] + w]
    return image_crop


def autoCrop(img_rgb, xml_url, xml_output_url):
    label = xml(xml_url)

    _, w, h = img_rgb.shape[::-1]


    thread = (w-CROP_IMG_SIZE)//2
    threadY = (h-CROP_IMG_SIZE)//2

    [x1, y1, x2, y2] = label['gt_bbox'][0]

    # x1小于 (180-112) = 34
    # y1 小于 (120 - 112)/2 = 4
    dst_img = np.zeros([CROP_IMG_SIZE, CROP_IMG_SIZE, 3], dtype=np.uint8)

    y1 = y1-threadY
    y2 = y2-threadY

    croptype = 0

    if x1 < thread:
        dst_img = img_rgb[threadY:threadY+CROP_IMG_SIZE, 0:CROP_IMG_SIZE]
        # y1、y2减 4，x1,x2不变
        croptype = 0

    elif x1 >= thread and x2 < w-thread:
        dst_img = img_rgb[threadY:threadY +
                          CROP_IMG_SIZE, thread:thread+CROP_IMG_SIZE]
        # y1,y2减4,x1,x2减thread
        x1 = x1-thread
        x2 = x2-thread
        croptype = 1

    elif x1 >= thread and x2 > w-thread:
        dst_img = img_rgb[threadY:threadY +
                          CROP_IMG_SIZE, 2*thread:2*thread+CROP_IMG_SIZE]
        # y1,y2减4，x1,x2减2*thread
        x1 = x1-2*thread
        x2 = x2-2*thread
        croptype = 2

    if(x2 > CROP_IMG_SIZE):
        x2 = CROP_IMG_SIZE
    
    if abs(int(x2)-int(x1))<3:
        x1 = x1-5
        x2 = x2+5

    if abs(int(y2)-int(y1))<3:
        y1 = y1-5
        y2 = y2+5

    rewriteXML(xml_url, xml_output_url, (x1, y1, x2, y2),
               (CROP_IMG_SIZE, CROP_IMG_SIZE))

    return dst_img, (x1, y1, x2, y2),croptype


def matchModel(tmp_filename,filetype='fake_image'):
    img_url = FAKE_IMG_DIR+filetype+'\\'+tmp_filename+'.bmp'
    xml_url = img_url.replace('.bmp','.xml')
    
    # print(xml_url)

    xml_output_url = OUTPUT_DIR+filetype+'\\'+tmp_filename+'.xml'

    _, (px1, py1, px2, py2) = getBlob(xml_url)

    img_route = tmp_filename[:3]

    routetype = img_route[:2]

    routetype = int(routetype)

    model_url = 'standard/9607T/'+img_route+'.bmp'

    model = cv2.imread(model_url)
    _, w, h = model.shape[::-1]

    # 17为背面荧光，直接用检测系统给出的位置

    padding = 20

    # 7T丝印
    if routetype == 12:
        padding = 300

    # 13为正面红外，扩充100
    # padding = 100 if routetype == 16 else padding

    if(routetype == 16 or routetype == 13):

        padding = 100

        _px1 = px1//2
        _py1 = py1//2

        px1 = _px1-180
        py1 = _py1-120
        px2 = _px1
        py2 = _py1

        bx1 = px1-padding
        bx2 = px2+padding
        by1 = py1-padding
        by2 = py2+padding

        bx1_nopadding = px1
        by1_nopadding = py1

    elif routetype == 17 or routetype==15:
        bx1 = w-px2//2-padding
        bx2 = w-px1//2+padding
        by1 = h-py2//2-padding
        by2 = h-py1//2+padding
    
        bx1_nopadding = w-px2//2
        by1_nopadding = h-py2//2

    else:
        bx1 = w-px2 - padding
        bx2 = w-px1+padding
        by1 = h-py2-padding
        by2 = h-py1+padding

        bx1_nopadding = w-px2
        by1_nopadding = h-py2

    bx1 = max(bx1, 0)
    by1 = max(by1, 0)
    bx2 = min(bx2, w)
    by2 = min(by2, h)

    bx2 = w if bx2<0 else bx2
    by2=  h if by2<0 else by2

    # print(w, h,img_route)
    # print(px1, px2, py1, py2)
    # print(bx1, bx2, by1, by2)

    model_img_crop = model[by1:by2, bx1:bx2]

    # showImg(model_img_crop)

    # 读取需要处理的文件 15需要图像翻转

    img_rgb = flip180(img_url) if routetype == 15 else cv2.imread(img_url)

    img_rgb, (x1, y1, x2, y2),croptype = autoCrop(img_rgb, xml_url, xml_output_url)
    try:
        image_crop = imgMatchWithROI(img_rgb, model_img_crop)
    except:
        xStart = bx1_nopadding
        thread = (180-CROP_IMG_SIZE)/2

        if croptype ==1:
            xStart = int(xStart- thread)
        elif croptype == 2:
            xStart = int(xStart- 2*thread)
        
        xStart = max(0,xStart)
        bx1_nopadding = max(bx1_nopadding,0)
        by1_nopadding = max(by1_nopadding,0)

        blackimg = getBlackImg()
        
        blackimg[0:CROP_IMG_SIZE, 0:min(w-xStart,CROP_IMG_SIZE)] =  model[by1_nopadding+4:by1_nopadding+4+CROP_IMG_SIZE, xStart:xStart+CROP_IMG_SIZE]
       
        image_crop = blackimg
        

    # need handle
    combine = imgCombine(img_rgb, image_crop)
    
    # showImg(combine)
    
    cv2.imwrite(OUTPUT_DIR+filetype+'\\'+tmp_filename+'.bmp', combine)

    return combine

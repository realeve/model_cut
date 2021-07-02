import cv2
from matplotlib import pyplot as plt
import label_util as util
import glob

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 从模板图像中匹配出原始图像


img_width = util.img_width
img_height = util.img_height


FAKE_IMG_DIR = '0630/'
OUTPUT_DIR = 'out/'

tmp_filename = "131_1_2075B630_1663"

model_url = 'standard/9607T/101.bmp'

xml_output_url = OUTPUT_DIR+tmp_filename+'.xml'

CROP_IMG_SIZE = 112

filelist = glob.glob('.\\0630\\*.bmp')

for i, filename in enumerate(filelist):
    file = filename.split('\\')[2]
    bmpfileName = file.replace('.bmp', '')

    print("开始处理{}".format(bmpfileName))

    util.matchModel(bmpfileName)

# img = util.matchModel(tmp_filename)
# util.showImg(img)


def matchImg():

    img_url = FAKE_IMG_DIR+tmp_filename+'.bmp'
    xml_url = FAKE_IMG_DIR+tmp_filename+'.xml'
    img_rgb = cv2.imread(img_url)

    # 自动裁剪为112*112
    img_rgb, (x1, y1, x2, y2) = util.autoCrop(img_rgb, xml_url, xml_output_url)

    image_crop = util.imgMatch(img_rgb, model_url)

    subtracted = cv2.absdiff(img_rgb, image_crop)

    roi = subtracted[y1:y2, x1:x2]

    black_img = util.getBlackImg(CROP_IMG_SIZE)

    black_img[y1:y2, x1:x2] = roi

    black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)

    combine = util.imgCombine(img_rgb, image_crop)

    # black_img = cv2.Sobel(black_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)

    # 阈值
    # ret, black_img = cv2.threshold(black_img, 40, 255, cv2.THRESH_BINARY)

    # 腐蚀
    # black_img = cv2.erode(black_img, None, iterations=1)

    # 中值滤波
    # black_img = cv2.medianBlur(black_img, ksize=3)

    plt.figure(figsize=(15, 15))

    plt.subplot(151), plt.imshow(
        cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)),
    plt.title('模板匹配结果'), plt.axis('off')

    plt.subplot(152), plt.imshow(
        cv2.cvtColor(combine, cv2.COLOR_BGR2RGB)),
    plt.title('待检图像'), plt.axis('off')

    plt.subplot(153), plt.imshow(
        cv2.cvtColor(subtracted, cv2.COLOR_BGR2RGB)),
    plt.title('差异图像'), plt.axis('off')

    plt.subplot(154), plt.imshow(black_img, cmap='gray'),
    plt.title('blob roi image '), plt.axis('off')

    cv2.imwrite(OUTPUT_DIR+tmp_filename+'.bmp', combine)

    plt.show()


# matchImg()

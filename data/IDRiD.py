
import os
import glob
import cv2
import numpy as np
from config import parse_args

def get_path_list(data_root_path,img_path,label_path,fov_path, vessel_path):
    tmp_list = [img_path,label_path,fov_path,vessel_path]
    res = []
    for i in range(len(tmp_list)):
        data_path = os.path.join(data_root_path,tmp_list[i])
        filename_list = os.listdir(data_path)
        filename_list.sort()
        res.append([os.path.join(data_path,j) for j in filename_list])
    return res

def write_path_list(name_list, save_path, file_name):
    f = open(os.path.join(save_path, file_name), 'w')
    for i in range(len(name_list[0])):
        f.write(str(name_list[0][i]) + " " + str(name_list[1][i]) + " " + str(name_list[2][i]) + " " + str(name_list[3][i]) + '\n')
    f.close()

def generate_fov(image_dir):
    for setname in ['TrainingSet', 'TestingSet']:
        mask_dir = os.path.join(image_dir, 'Groundtruths', setname, 'FOV')
        os.mkdir(mask_dir)

        imgs_ori = glob.glob(os.path.join(image_dir, 'OriginalImages/' + setname + '/*.jpg'))

        for image_path in imgs_ori:
            image = cv2.imread(image_path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            black_mask = np.uint8((image_gray > 15) * 255.)
            ret, thresh = cv2.threshold(black_mask, 127, 255, 0)
            im2, contours, her = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros(image.shape[:2], dtype='uint8') * 255
            cn = []
            for contour in contours:
                if len(contour) > len(cn):
                    cn = contour
            cv2.drawContours(mask, [cn], -1, 255, -1)
            mask = 255-mask
            # print(mask[10,10],mask[1500,1500])
            # cv2.imshow('img',cv2.resize(mask,(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST))
            # cv2.waitKey(0)
            image_name = os.path.split(image_path)[-1].split('.')[0]
            mask_path = os.path.join(mask_dir, image_name + '_FOV.tif')
            cv2.imwrite(mask_path, mask)


def clahe_gridsize(image_path, mask_path, denoise=False, contrastenhancement=False, brightnessbalance=None,
                   cliplimit=None, gridsize=8):
    """This function applies CLAHE to normal RGB images and outputs them.
    The image is first converted to LAB format and then CLAHE is applied only to the L channel.
    Inputs:
      image_path: Absolute path to the image file.
      mask_path: Absolute path to the mask file.
      denoise: Toggle to denoise the image or not. Denoising is done after applying CLAHE.
      cliplimit: The pixel (high contrast) limit applied to CLAHE processing. Read more here: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
      gridsize: Grid/block size the image is divided into for histogram equalization.
    Returns:
      bgr: The CLAHE applied image.
    """
    bgr = cv2.imread(image_path)

    # brightness balance.
    if brightnessbalance:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask_img = cv2.imread(mask_path, 0)
        mask_img = 255 - mask_img
        brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
        bgr = np.uint8(np.minimum(bgr * brightnessbalance / brightness, 255))

    if contrastenhancement:
        # illumination correction and contrast enhancement.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(gridsize, gridsize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if denoise:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 1, 3)
        bgr = cv2.bilateralFilter(bgr, 5, 1, 1)

    return bgr

def generate_process(image_dir):
    limit = 2
    grid_size = 8
    if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE')):
        os.mkdir(os.path.join(image_dir, 'Images_CLAHE'))
        # compute mean brightess
        meanbright = 0.
        images_number = 0
        for tempsetname in ['TrainingSet', 'TestingSet']:
            imgs_ori = glob.glob(os.path.join(image_dir, 'OriginalImages/' + tempsetname + '/*.jpg'))
            imgs_ori.sort()
            images_number += len(imgs_ori)
            # mean brightness.
            for img_path in imgs_ori:
                img_name = os.path.split(img_path)[-1].split('.')[0]
                mask_path = os.path.join(image_dir, 'Groundtruths', tempsetname, 'FOV', img_name + '_FOV.tif')
                gray = cv2.imread(img_path, 0)
                mask_img = cv2.imread(mask_path, 0)
                mask_img = 255-mask_img
                print(mask_img[10,10])
                brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
                meanbright += brightness
        meanbright /= images_number
        # print(meanbright,images_number)

        for setname in ['TrainingSet', 'TestingSet']:
            if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE', setname)):
                os.mkdir(os.path.join(image_dir, 'Images_CLAHE', setname))
            imgs_ori = glob.glob(os.path.join(image_dir, 'OriginalImages/' + setname + '/*.jpg'))

            for img_path in imgs_ori:
                img_name = os.path.split(img_path)[-1].split('.')[0]
                mask_path = os.path.join(image_dir, 'Groundtruths', setname, 'FOV', img_name + '_FOV.tif')
                clahe_img = clahe_gridsize(img_path, mask_path, denoise=True, contrastenhancement=True, brightnessbalance=meanbright,
                                       cliplimit=limit, gridsize=grid_size)
                cv2.imwrite(os.path.join(image_dir, 'Images_CLAHE', setname, os.path.split(img_path)[-1]), clahe_img)

def generate_Colabels(image_dir):
    dir = os.path.join(image_dir, 'Groundtruths')
    lesion = ['HardExudates','Haemorrhages','Microaneurysms','SoftExudates']
    ab = ['EX','HE','MA','SE']
    for setname in ['TrainingSet', 'TestingSet']:
        if not os.path.exists(os.path.join(dir, setname, 'Co')):
            os.mkdir(os.path.join(dir, setname, 'Co'))

            imgs = glob.glob(os.path.join(dir, setname,'Microaneurysms','*'))
            for img in imgs:
                name = os.path.split(img)[-1].split('.')[0].split('MA')[0]
                Co = np.zeros_like(cv2.imread(img))
                for i in range(len(lesion)):
                    path = os.path.join(dir,setname, lesion[i],name+ab[i])
                    if os.path.exists(path):
                        label = cv2.imread(path)
                        Co += label*(i+1)
                cv2.imwrite(os.path.join(dir, setname, 'Co',name+'CO.tif'),Co)


def generate_list(image_dir,args):

    if args.preprocess:
        img_train = "Images_CLAHE/TrainingSet/"
        img_test = "Images_CLAHE/TestingSet"
    else:
        img_train = "OriginalImages/TrainingSet/"
        img_test = "OriginalImages/TestingSet"

    gt_train = "Groundtruths/TrainingSet/CO/"
    fov_train = "Groundtruths/TrainingSet/FOV/"
    vessel_train = "Groundtruths/TrainingSet/vessel/"

    gt_test = "Groundtruths/TestingSet/CO/"
    fov_test = "Groundtruths/TestingSet/FOV/"
    vessel_test = "Groundtruths/TestingSet/vessel/"
    # ----------------------------------------------------------
    save_path = "./data_path_list/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = "./data_path_list/IDRiD/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)


    train_list = get_path_list(image_dir, img_train, gt_train, fov_train, vessel_train)
    print('Number of train imgs:', len(train_list[0]))
    write_path_list(train_list, save_path, 'train.txt')

    test_list = get_path_list(image_dir, img_test, gt_test, fov_test, vessel_test)
    print('Number of test imgs:', len(test_list[0]))
    write_path_list(test_list, save_path, 'test.txt')



if __name__ == '__main__':
    args = parse_args()
    image_dir = r'D:\Datasets\IDRiD\Segmentation\Segmentation'
    if os.path.isdir("./data_path_list/IDRiD/"):
        print('data path lists have been already generated!')
    else:
        generate_list(image_dir,args)
        if not os.path.exists(os.path.join(image_dir,'Groundtruths', 'TrainingSet', 'FOV')):
            generate_fov(image_dir)
            print('FOVs have been already generated!')
        if not os.path.exists(os.path.join(image_dir,'Groundtruths', 'TrainingSet', 'Co')):
            generate_Colabels(image_dir)
            print('CoLabels have been already generated!')
        if args.preprocess:
                generate_process(image_dir)
                # print('Preprocessed Images have been already generated!')



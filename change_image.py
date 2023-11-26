import cv2
import numpy as np
import math

def filter(image, data):
    #마스크 만들기
    mask_size = int(math.sqrt(len(data)))
    mask = np.array(data, dtype=np.float32).reshape(mask_size, mask_size)

    #결과물 배열
    dst = np.copy(image).astype(np.float32)
    #색상별 분해
    dst_split = cv2.split(dst)

    #이미지 사이즈, 마스크 사이즈 중간값
    row, col = image.shape[:2]
    center = mask_size // 2

    #색상별 회선
    for c in range(3):
        for i in range(center, row - center):
            y1 = i - center
            y2 = i + center + 1
            for j in range(center, col - center):
                x1 = j - center
                x2 = j + center + 1

                t2 = dst[y1:y2, x1:x2, c]
                t2 = cv2.multiply(t2, mask)
                dst_split[c][i, j] = cv2.sumElems(t2)[0]

    dst = cv2.merge(dst_split).astype(np.uint8)

    return dst

def filter_max(image, ksize):
    #색상별 분리
    dst = np.copy(image)
    dst_split = cv2.split(dst)

    #이미지 사이즈, 마스크 중간값
    row, col = image.shape[:2]
    center = ksize // 2

    #회선
    for c in range(3):
        for i in range(center, row - center):
            y1 = i - center
            y2 = i + center + 1
            for j in range(center, col - center):
                x1 = j - center
                x2 = j + center + 1

                tmp = image[y1:y2, x1:x2, c]
                dst_split[c][i, j] = cv2.minMaxLoc(tmp)[1]

    #합침
    dst = cv2.merge(dst_split)

    return dst

def change_canny(image, face, threshold1=50, threshold2=150):
    if face:
        # 얼굴 부분 분리
        x1, y1, x2, y2 = face
        dst = image[y1:y1 + y2, x1:x1 + x2]
    else:
        x1, y1 = 0, 0
        y2, x2 = image.shape[:2]
        dst = image.copy()
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    #캐니 에지
    dst = cv2.Canny(dst, threshold1, threshold2)
    dst2 = cv2.merge((dst, dst, dst))
    image[y1:y1 + y2, x1:x1 + x2] = dst

def change_blur(image, face, level=1):
    #블러에 사용할 마스크
    data1 = [1/9, 1/9, 1/9,
             1/9, 1/9, 1/9,
             1/9, 1/9, 1/9]

    data2 = [1/25, 1/25, 1/25, 1/25, 1/25,
             1/25, 1/25, 1/25, 1/25, 1/25,
             1/25, 1/25, 1/25, 1/25, 1/25,
             1/25, 1/25, 1/25, 1/25, 1/25,
             1/25, 1/25, 1/25, 1/25, 1/25]

    if face:
        # 얼굴 부분 분리
        x1, y1, x2, y2 = face
        dst = image[y1:y1 + y2, x1:x1 + x2]
    else:
        x1, y1 = 0, 0
        y2, x2 = image.shape[:2]
        dst = image.copy()

    #마스크 고르기
    if level == 1:
        dst = filter(dst, data1)
    else:
        dst = filter(dst, data2)

    dst2 = cv2.merge((dst, dst, dst))

    image[y1:y1 + y2, x1:x1 + x2] = dst

def change_max(image, face, ksize=10):
    if face:
        # 얼굴 부분 분리
        x1, y1, x2, y2 = face
        dst = image[y1:y1 + y2, x1:x1 + x2]
    else:
        x1, y1 = 0, 0
        y2, x2 = image.shape[:2]
        dst = image.copy()

    dst = filter(dst, ksize)
    image[y1:y1 + y2, x1:x1 + x2] = dst

def change_mosaic(image, face=None, ksize=10):
    if face:
        #얼굴 부분 분리
        x1, y1, x2, y2 = face
        dst = image[y1:y1 + y2, x1:x1 + x2]
    else:
        x1, y1 = 0, 0
        y2, x2 = image.shape[:2]
        dst = image.copy()

    #이미지 크기 확인
    row, col = dst.shape[:2]

    #변수 하나
    isSame = False

    for i in range(row):
        if i % ksize == 0:
            isSame = False
        else:
            isSame = True
        for j in range(col):
            #isSame이면 윗줄이랑 똑같이
            if isSame:
                dst[i, j] = dst[i - 1, j]
            else:
                if j % ksize != 0:
                    dst[i, j] = dst[i, j - 1]

    image[y1:y1+y2, x1:x1+x2] = dst

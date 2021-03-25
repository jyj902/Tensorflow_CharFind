import cv2
from keras.models import load_model
import numpy as np



model = load_model('char_model_v1.h5')
cap = cv2.VideoCapture(0)
class_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
              'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e',
              'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
              'x', 'y', 'z']

list_src = [[],[]]

while (True):
    count =0

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mor_gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

    adp_gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)

    close_gray = cv2.morphologyEx(adp_gray, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    dst = cv2.medianBlur(close_gray, 5)

    _, src_bin = cv2.threshold(close_gray, 0, 255, cv2.THRESH_BINARY_INV+16)

    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)
    text_l1 = ''
    text_l2 = ''
    for i in range(1, cnt):
        (x, y, w, h, area) = stats[i]
        if area < 150 or w<15 or h<15:
            continue
        elif area > 2000 or w>70 or h > 70:
            continue
        if y >130 :
            list_src[1].append(stats[i])
        else :
            list_src[0].append(stats[i])
        count = count + 1

    list_src[0].sort(key=lambda x:x[0])
    list_src[1].sort(key=lambda x:x[0])

    for i in range(0, len(list_src[0])):
        try:
            (x, y, w, h, area) = list_src[0][i]
            src = gray[y-3:y+h+3, x-3:x+w+3]
            dst = cv2.resize(src, dsize=(180, 180), interpolation=cv2.INTER_AREA)
            x_test = np.array([dst])
            y_predict = model.predict_classes(x_test)
            test_pd = model.predict(x_test)
            max_pd = np.argmax(test_pd)
            print(max_pd)
            if max_pd < 0:
                text_l1 += '?'
                continue
            text_l1 += class_name[y_predict[0]]
            file_name_path = 'test/test_l0' + str(i) + '.png'
            cv2.imwrite(file_name_path, src)
        except Exception as e:
            print(str(e))

    for i in range(0, len(list_src[1])):
        try:
            (x, y, w, h, area) = list_src[1][i]
            src = gray[y-3:y+h+3, x-3:x+w+3]
            dst = cv2.resize(src, dsize=(180, 180), interpolation=cv2.INTER_AREA)
            x_test = np.array([dst])
            y_predict = model.predict_classes(x_test)
            test_pd = model.predict(x_test)
            max_pd = np.argmax(test_pd)
            print(max_pd)
            if max_pd < 0:
                text_l2 += '?'
                continue
            text_l2 += class_name[y_predict[0]]
            file_name_path = 'test/test_l1' + str(i) + '.png'
            cv2.imwrite(file_name_path, src)
        except Exception as e:
            print(str(e))

    print(text_l1)
    print(text_l2)
    break
    #cv2.imshow('gray', gray)
    #cv2.imshow('mor_gray', mor_gray)
    #cv2.imshow('adp_gray', adp_gray)
    #cv2.imshow('close_gray', close_gray)
    #cv2.imshow('dst', dst)
    #cv2.imshow("test",dst)



    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
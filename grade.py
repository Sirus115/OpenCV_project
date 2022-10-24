import cv2
import cv2 as cv
import numpy as np

import globalVar


class ans_correct:
    def cvshow(self, name, img):
        cv.imshow(name, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def four_point_transform(self, img, four_points):
        rect = self.order_points(four_points)
        (tl, tr, br, bl) = rect

        # 计算输入的w和h的值
        widthA = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
        widthB = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
        heightB = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
        maxHeight = max(int(heightA), int(heightB))

        # 变换后对应的坐标位置
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype='float32')

        # 最主要的函数就是 cv2.getPerspectiveTransform(rect, dst) 和 cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))
        return [warped, M, maxWidth, maxHeight]

    def order_points(self, points):
        res = np.zeros((4, 2), dtype='float32')
        # 按照从前往后0，1，2，3分别表示左上、右上、右下、左下的顺序将points中的数填入res中

        # 将四个坐标x与y相加，和最大的那个是右下角的坐标，最小的那个是左上角的坐标
        sum_hang = points.sum(axis=1)
        res[0] = points[np.argmin(sum_hang)]
        res[2] = points[np.argmax(sum_hang)]

        # 计算坐标x与y的离散插值np.diff()
        diff = np.diff(points, axis=1)
        res[1] = points[np.argmin(diff)]
        res[3] = points[np.argmax(diff)]

        # 返回result
        return res

    def sort_contours(self, contours, method="l2r"):
        # 用于给轮廓排序，l2r, r2l, t2b, b2t
        reverse = False
        i = 0
        if method == "r2l" or method == "b2t":
            reverse = True
        if method == "t2b" or method == "b2t":
            i = 1

        boundingBoxes = [cv.boundingRect(c) for c in contours]
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda a: a[1][i], reverse=reverse))
        return contours, boundingBoxes

    # 计算最终得分并在图中标注
    def start(self, file):
        # 正确答案
        right_key = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
        # 输入图像
        img = cv.imread(file)
        img_copy = img.copy()
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # cvshow('img-gray', img_gray)

        # 图像预处理
        # 高斯降噪
        img_gaussian = cv.GaussianBlur(img_gray, (5, 5), 1)
        # cvshow('gaussianblur', img_gaussian)
        # canny边缘检测
        img_canny = cv.Canny(img_gaussian, 80, 150)
        # cvshow('canny', img_canny)
        # 轮廓识别——答题卡边缘识别
        cnts, hierarchy = cv.findContours(img_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(img_copy, cnts, -1, (0, 0, 255), 3)
        # cvshow('contours-show', img_copy)

        docCnt = None

        # 确保检测到了
        if len(cnts) > 0:
            # 根据轮廓大小进行排序
            cnts = sorted(cnts, key=cv.contourArea, reverse=True)

            # 遍历每一个轮廓
            for c in cnts:
                # 近似
                peri = cv.arcLength(c, True)  # arclength 计算一段曲线的长度或者闭合曲线的周长；
                # 第一个参数输入一个二维向量，第二个参数表示计算曲线是否闭合

                approx = cv.approxPolyDP(c, 0.02 * peri, True)
                # 用一条顶点较少的曲线/多边形来近似曲线/多边形，以使它们之间的距离<=指定的精度；
                # c是需要近似的曲线，0.02*peri是精度的最大值，True表示曲线是闭合的

                # 准备做透视变换
                if len(approx) == 4:
                    docCnt = approx
                    break

        # 透视变换——提取答题卡主体
        docCnt = docCnt.reshape(4, 2)
        warped = self.four_point_transform(img_gray, docCnt)[0]
        M = self.four_point_transform(img_gray, docCnt)[1]
        maxWidth = self.four_point_transform(img_gray, docCnt)[2]
        maxHeight = self.four_point_transform(img_gray, docCnt)[3]
        crop = cv.warpPerspective(img ,M,(maxWidth,maxHeight))

        # cvshow('warped', warped)

        # 轮廓识别——识别出选项
        thresh = cv.threshold(warped, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        # cvshow('thresh', thresh)
        thresh_cnts, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        w_copy = warped.copy()
        cv.drawContours(w_copy, thresh_cnts, -1, (0, 0, 255), 2)
        # cvshow('warped_contours', w_copy)

        questionCnts = []
        # 遍历，挑出选项的cnts
        for c in thresh_cnts:
            (x, y, w, h) = cv.boundingRect(c)
            ar = w / float(h)
            # 根据实际情况指定标准
            if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
                questionCnts.append(c)

        # 检查是否挑出了选项
        w_copy2 = warped.copy()
        cv.drawContours(w_copy2, questionCnts, -1, (0, 0, 255), 2)
        # cvshow('questionCnts', w_copy2)

        # 检测每一行选择的是哪一项，并将结果储存在元组bubble中，记录正确的个数correct
        # 按照从上到下t2b对轮廓进行排序
        questionCnts = self.sort_contours(questionCnts, method="t2b")[0]
        correct = 0
        # 每行有5个选项
        for (i, q) in enumerate(np.arange(0, len(questionCnts), 5)):
            # 排序
            cnts = self.sort_contours(questionCnts[q:q + 5])[0]

            bubble = None
            # 得到每一个选项的mask并填充，与正确答案进行按位与操作获得重合点数
            for (j, c) in enumerate(cnts):
                mask = np.zeros(thresh.shape, dtype='uint8')
                cv.drawContours(mask, [c], -1, 255, -1)
                # cvshow('mask', mask)

                # 通过按位与操作得到thresh与mask重合部分的像素数量
                bitand = cv.bitwise_and(thresh, thresh, mask=mask)
                totalPixel = cv.countNonZero(bitand)

                if bubble is None or bubble[0] < totalPixel:
                    bubble = (totalPixel, j)

            k = bubble[1]
            color = (0, 0, 255)
            if k == right_key[i]:
                correct += 1
                color = (0, 255, 0)

            # 绘图
            cv.drawContours(crop, [cnts[right_key[i]]], -1, color, 3)
            # cvshow('final', warped)
        # print(correct)
        score = (correct / 5.0) * 100
        print(f"Score: {score}%")
        cv.putText(crop, f"Score: {score}%", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        result = globalVar.get_value('result')
        cv2.imwrite(result + 'chapter8/' + 'grade.jpg', crop)

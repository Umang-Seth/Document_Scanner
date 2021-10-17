import cv2
import numpy as np

def empty(a):
    pass

def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 640, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("Threshold2", "Trackbars", 255, 255, empty)

def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1, Threshold2
    return src

def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(3,3),1)
    imgCanny = cv2.Canny(imgBlur, valTrackbars()[0], valTrackbars()[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (0, 0, 255), 10)
    #print(biggest)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    #print("diff",diff)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print(myPointsNew)
    return myPointsNew


def getWarp(img, biggest):
    if biggest.size != 0:
        biggest = reorder(biggest)
    else:
        return 0
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [Width, 0], [0, Height], [Width, Height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (Width, Height))
    return imgOutput



cap = cv2.VideoCapture(0)
cap.set(10, 150)
Width = 480
Height = 640
initializeTrackbars()
count = 0

while True:
    #success, img = cap.read()
    img = cv2.imread("12.jpeg")
    img = cv2.resize(img, (Width, Height))
    imgContour = img.copy()
    imgThres = preProcessing(img)
    Biggest = getContours(imgThres)
    imgWarp = getWarp(img, Biggest)

    cv2.imshow("Original",img)
    cv2.imshow("Original", imgContour)
    cv2.imshow("Canny", imgThres)

    if Biggest.size != 0:
        imgCropped = imgWarp[10:630, 10:430]
        cv2.imshow("Output", imgCropped)
        imgWarpGray = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 0, 1, 7, 7)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        cv2.imshow("Final", imgAdaptiveThre)
    else:
        cv2.imshow("Blank", np.zeros((Height, Width, 3), np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(count)
        cv2.imwrite("ScannedImage" + str(count) + ".jpg", imgAdaptiveThre)
        cv2.imwrite("OriginalImage" + str(count) + ".jpg", imgCropped)
        cv2.waitKey(300)
        count += 1

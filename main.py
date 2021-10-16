import cv2
import numpy as np

cap = cv2.VideoCapture(0)
#cap.set(3,1080)
#cap.set(4,1920)
cap.set(10,150)
Width = 480
Height = 640

def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,50,200)
    kernel = np.ones((5,5))
    #imgDial = cv2.dilate(imgCanny,kernel,iterations=1)
    #imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgCanny

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>5000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area>maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 255, 0), 10)
    print(biggest)
    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    print("diff",diff)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    print(myPointsNew)
    return myPointsNew

def getWarp(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [Width, 0], [0, Height], [Width, Height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (Width, Height))
    return imgOutput

while True:
    #success, img = cap.read()
    img = cv2.imread("WhatsApp Image 2021-10-15 at 16.20.49.jpeg")
    img = cv2.resize(img,(Width,Height))
    imgContour = img.copy()

    imgThres = preProcessing(img)
    Biggest = getContours(imgThres)
    imgWarp = getWarp(img,Biggest)

    cv2.imshow("Video",imgThres)
    cv2.imshow("V", imgContour)
    cv2.imshow("Vi", imgWarp)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
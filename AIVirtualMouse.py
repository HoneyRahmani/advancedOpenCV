import cv2
import numpy as np
import time
import HandTrackingModules as htm
# to move around mouse
import autopy

####################
wCam, hCam = 640, 480
pTime = 0
frameR = 100
smoothening = 10
plocX, plocY = 0, 0
clocX, clocY = 0, 0
####################

cap = cv2.VideoCapture(0)
# address = "https://192.168.42.129:8080/video"
# cap.open(address)

cap.set(3, 640)
cap.set(4, 480)

detector = htm.handDetector(maxHands=1)

wScr , hScr = autopy.screen.size()
#print(wScr, hScr)

while True:
    # 1.Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    #print(lmList)

    # 2. Get the tip of the index and middle finger
    if len(lmList)!=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #print(x1, y1, x2, y2)
        # 3. Check which fingers are up
        fingers = detector.fingerUp()
        #print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        # 4. Only index finger: Moving mode
        if fingers[1]==1 and fingers[2]==0:
            # 5. Convert coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            # 6. Smooth value
            clocX = plocX + (x3 - plocX)/smoothening
            clocY = plocY + (y3 - plocY)/smoothening

            # 7. Move mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        # 8. Both index and middle fingers are up : clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.fingerDistance(img, 8, 12)
            #print(length)
            # 10. click mouse if distance short
            if length < 48:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15,
                           (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()



    # 11. Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN
                , 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

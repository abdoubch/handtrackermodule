import cv2
import mediapipe as mp
import time
import math
import numpy as np

class handTracker():
    def __init__(self,mode=False,maxHands = 2 ,detectionConf = 0.5,trackConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        self.mpHands = mp.solutions.hands
        self.tipIds = [4, 8, 12, 16, 20]
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConf,
            min_tracking_confidence=self.trackConf,
        )
        self.mpdraw = mp.solutions.drawing_utils
    def trackHands(self,img, showDraw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlLms in self.results.multi_hand_landmarks:
                if showDraw:
                    self.mpdraw.draw_landmarks(img, handlLms, self.mpHands.HAND_CONNECTIONS)
        return img
    def handPointPosition(self,img,handNumber=0,showDraw=True):
        xlist =[]
        ylist = []
        bbox =[]
        self.landMarkList = []
        if self.results.multi_hand_landmarks:
            handCHosen = self.results.multi_hand_landmarks[handNumber]
            for id, landMark in enumerate(handCHosen.landmark):
                height, width, channel = img.shape
                cx, cy = int(landMark.x * width), int(landMark.y * height)
                xlist.append(cx)
                ylist.append(cy)
                self.landMarkList.append([id,cx,cy])
                if showDraw:
                    cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
        xmin,xmax= min(xlist),max(xlist)
        ymin, ymax = min(ylist), max(ylist)
        bbox = xmin, ymin, xmax, ymax
        if showDraw:
            cv2.rectangle(
                img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2
            )

        return self.landMarkList,bbox
    def fingerUp(self):
        fingers = []
        # Pouce
        if self.landMarkList[self.tipIds[0]][1] > self.landMarkList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Other Fingers
        for id in range(1,5):
            if self.landMarkList[self.tipIds[id]][2] < self.landMarkList[self.tipIds[id]-2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    def distance(self,img,p1,p2,showDraw=True,r=15,t=3):
        x1, y1 = self.landMarkList[p1][1:]
        x2, y2 = self.landMarkList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if showDraw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    ctime = 0
    ptime = 0
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
    while True:
        success, img = cap.read()
        img = tracker.trackHands(img)
        landMarkList = tracker.handPointPosition(img,showDraw=False)
        if len(landMarkList) != 0:
            print(landMarkList[4])
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img,"fps :" + str(int(fps)),(10, 70),cv2.FONT_HERSHEY_COMPLEX,1,(255, 0, 255),2,)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()

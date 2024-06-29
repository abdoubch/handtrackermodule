import cv2
import mediapipe as mp
import time


class handTracker():
    def __init__(self,mode=False,maxHands = 2 ,detectionConf = 0.5,trackConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        self.mpHands = mp.solutions.hands
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
        landMarkList = []
        if self.results.multi_hand_landmarks:
            handCHosen = self.results.multi_hand_landmarks[handNumber]
            for id, landMark in enumerate(handCHosen.landmark):
                height, width, channel = img.shape
                cx, cy = int(landMark.x * width), int(landMark.y * height)
                landMarkList.append([id,cx,cy])
                if showDraw:
                    cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
        return landMarkList


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

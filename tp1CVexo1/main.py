import cv2
import imutils

video = cv2.VideoCapture("sans_titre2.mp4")

firstFrame = None

mirror = False

while True:  # do while simuler
    ret, frame = video.read()

    if ret:
        if mirror:
            frame = cv2.flip(frame, 1)

        niveaux_de_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if firstFrame is None:  # si la premiere image n'exite pas
            firstFrame = niveaux_de_gris

        # Difference absolue entre les 2 frames
        frame_differences = cv2.absdiff(firstFrame, niveaux_de_gris)
        threshold = cv2.threshold(frame_differences, 20, 100, cv2.THRESH_BINARY)[1]

        contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]

        for c in contours:
            # Si le contour est petit on l'ignore
            if cv2.contourArea(c) < 50000:
                continue
            # Dessiner les contours du rectangle
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        cv2.imshow("Différence entre les frames (niveaux de gris)", frame_differences)
        cv2.imshow("Segmentation B&W", threshold)
        cv2.imshow("Capture des mouvements", frame)

        cv2.namedWindow("Capture des mouvements", cv2.WINDOW_AUTOSIZE)

    key = cv2.waitKey()

    if key == 27:
        break

cv2.destroyAllWindows()

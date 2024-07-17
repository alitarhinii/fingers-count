import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import math

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(maxHands=2, detectionCon=0.6)

# Positions of the operation 
box_positions = [(850, 70, 60, 50),  # division
                 (930, 70, 60, 50),  # Subtraction
                 (1020, 70,60, 50),# Multiplication
                 (1110,70,60,50)]  #addition

operation_symbols = ['/', '-', '*','+']  
current_operation = '+'

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        total_finger_count = 0  # Initialize total finger count

        # Get finger count for each hand
        first_hand_finger_count = detector.fingersUp(hands[0]).count(1)
        second_hand_finger_count = detector.fingersUp(hands[1]).count(1) if len(hands) == 2 else 0

        if current_operation == '+':
            total_finger_count = first_hand_finger_count + second_hand_finger_count if len(hands) == 2 else first_hand_finger_count
        elif current_operation == '-':
            total_finger_count = first_hand_finger_count - second_hand_finger_count if len(hands) == 2 else first_hand_finger_count
        elif current_operation == '*':
            total_finger_count = first_hand_finger_count * second_hand_finger_count if len(hands) == 2 else first_hand_finger_count
        elif current_operation == '/':
            total_finger_count = first_hand_finger_count / second_hand_finger_count if len(hands) == 2 and second_hand_finger_count != 0 else first_hand_finger_count

        # Display finger count for each hand
        for i, hand in enumerate(hands):
            bbox = hand["bbox"]
            text_x = bbox[0] + 60
            text_y = bbox[1] - 30
            if len(hands) == 1:
                cv.putText(img, 'Fingers up ' + str(first_hand_finger_count), (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            if len(hands) == 2:
                if i == 0:
                    cv.putText(img, 'Fingers up ' + str(first_hand_finger_count), (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                else:
                    cv.putText(img, 'Fingers up ' + str(second_hand_finger_count), (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Display total finger count
        cv.putText(img, 'Total Fingers ' + str(total_finger_count), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw operation boxes
        for i, (x, y, w, h) in enumerate(box_positions):
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), cv.FILLED)
            cv.putText(img, operation_symbols[i], (x + 20, y + 35), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Check if hand is close to any operation box
        for i, (x, y, w, h) in enumerate(box_positions):
            for hand in hands:
                lmList = hand["lmList"]
                if lmList:
                    # Calculate distance between landmark 4 and landmark 8
                    distance, _, _ = detector.findDistance((lmList[8][0], lmList[8][1]), (lmList[12][0], lmList[12][1]))
                    if distance < 30:  # If hand is close to the box
                        x, y, w, h = box_positions[i]
                        if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                            # Display operation result
                            current_operation = operation_symbols[i]

    cv.imshow("Image", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
cap.release()

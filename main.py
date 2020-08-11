import f_main
import cv2 
import time
import argparse
import imutils

# instacio recognizer
recognizer = f_main.rec()

def main(parse):
    if parse.input == "webcam":
        cam = cv2.VideoCapture(0)
        while True:
            # read the frame from the camera and send it to the server
            star_time = time.time()
            ret, frame = cam.read()
            frame = imutils.resize(frame, width=720)

            res = recognizer.recognize_face(frame)
            print(res)
            frame = f_main.bounding_box(frame,res["faces"],res["names"])

            end_time = time.time() - star_time    
            FPS = 1/end_time
            cv2.putText(frame,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.imshow("face_recognition",frame)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break

    elif parse.input == "image":
            frame = cv2.imread(parse.path_im)
            res = recognizer.recognize_face(frame)
            print(res)
            frame = f_main.bounding_box(frame,res["faces"],res["names"])
            cv2.imshow("face_recognition",frame)
            cv2.imwrite("test.jpg",frame)
            cv2.waitKey(0)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="face recognition implementation")
    parse.add_argument("--input",help="webcam or image")
    parse.add_argument("--path_im",help="path of image")
    parse = parse.parse_args()
    main(parse)
    
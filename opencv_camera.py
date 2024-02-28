import cv2
import numpy as np

def find_ball(frame):
    print("find ball")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours.count)

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            break
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.5:
            ((x,y), radius) = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 255), 2)
    return frame

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))


    print("vidcap")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    #input_image = cv2.imread('WIN_20240221_10_30_19_Pro.jpg')

    #if input_image is None:
        #print("Error: Could not open or read the image")
        #return
    
    #output_image = find_ball(input_image)

    #cv2.imshow('Output Image', output_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    while True:
        print("in while")
        ret, frame = cap.read()


        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
                
        frame = find_ball(frame)
        
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
            

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
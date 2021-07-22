import cv2
import inspect
import os
import numpy as np
import matplotlib.pyplot as plt

#reading locally the image file
dirname = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
filepath_image = os.path.join(dirname, 'img/test_image.jpg')
filepath_video = os.path.join(dirname, 'img/test_video.mp4')

def canny(image):
    #make img grayscale (simplifies the channels the computer needs to process 
    # - RGB (3 channels of 0-255) 
    # - GRAYSCALE (1 channel of (0-255) 
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    #make img blurred by applying a gaussian kernel of 5 by 5 
    #(averages the pixels by its neighbours pixels thus removing sharpness)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    #canny sets the edges as a threshold of max and min values 
    #(if the gradient of the matrix derivative pixels changes higher than the high_threshold,
    # then it is an edge otherwise it isn't)
    # Image comes out as: White - Edge (>high_threshold), Black - not a sufficient gradience to be an edge (<low_threshold)
    canny = cv2.Canny(blur,50,150)
    return canny


def region_of_interest(image):
    #gets the height of the image
    height = image.shape[0]
    #creates a region_of_interest and since the area in which the lanes will show are triangular shape 
    # we set three vertices (also you need to have an array of polygons)
    polygons = np.array([
        [(200,height ),(1100,height ),(550,250)]
    ]) 
    #A mask which has the same dimensions as the image, but with pixels intensity set to 0
    mask = np.zeros_like(image)
    # Fills the mask with the triangle we created above with the max intensity (255)
    cv2.fillPoly(mask,polygons,255)
    # This function will merge the mask & the image and only return the merge of both of them.
    # If there are pixels inside the mask area then they will appear in the final image.
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)      
    return line_image


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1* (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2  = line.reshape(4)
        parameters =  np.polyfit((x1,x2),(y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])


image = cv2.imread(filepath_image)
lane_image = np.copy(image)


#canny_img = canny(lane_image)
#cropped_image = region_of_interest(canny_img)
#lines = cv2.HoughLinesP(cropped_image,2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#averaged_lines = average_slope_intercept(lane_image,lines)
#line_image = display_lines(lane_image,averaged_lines)
#final_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)


#show the images
#cv2.imshow("result", final_image)
#cv2.waitKey(0)
#print(canny_img)

cap = cv2.VideoCapture(filepath_video)
while (cap.isOpened()):
    _, frame = cap.read()
    canny_img = canny(frame)
    cropped_image = region_of_interest(canny_img)
    lines = cv2.HoughLinesP(cropped_image,2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame,lines)
    line_image = display_lines(frame,averaged_lines)
    final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", final_image)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

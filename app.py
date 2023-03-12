import numpy as np
import cv2
import math

def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

def dist_2_pts(x1, y1, x2, y2):
	return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def rescaleImage(img, h):
    scale = 1
    if (h > 850 and h < 1100):
        scale = 0.6
    elif (h >= 1100):
        scale = 0.45
    else:
        scale = 0.8
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    dimension = (width, height)

    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA) 

def calvec(a):
    x1 = a[0]
    y1 = a[1]
    x2 = a[2]
    y2 = a[3]
    v = []
    v.append(x2 - x1)
    v.append(y2 - y1)
    return v
    
def calcos(x1, y1, x2, y2):
    a = x1*x2 + y1*y2
    b = np.sqrt(x1*x1 + y1*y1) * np.sqrt(x2*x2 + y2*y2)
    if (a > b):
        return 0
    else:
        return (math.acos(a / b) / np.pi) * 180
    
def readImage(sourceImage):
    image = cv2.imread(sourceImage)
    h = image.shape[0]
    img = rescaleImage(image, h)
    height, width = img.shape[:2]   
    return img, height, width
    
def imageProcess(preImg):
    grayColor = cv2.cvtColor(preImg, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayColor, (7, 7), cv2.BORDER_DEFAULT)
    canny = cv2.Canny(blur, 125, 175)

    # Convert the gray scaled image to BGR image
    img = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    grayCanny = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, grayColor, grayCanny
    
def findClock(img, grayColor, height):
    circles = cv2.HoughCircles(grayColor, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 150, 50, int(height*0.35), int(height*0.48)) #find all the circles in the image
    a, b, c = circles.shape
    x,y,r = avg_circles(circles, b) #calculate the center and radius of the circle
    cv2.circle(img, (x, y), r, (0, 255, 0), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(img, (x, y), 2, (0, 255, 0), 2, cv2.LINE_AA)  # draw center of circle
    return x, y, r

def lineDetect(grayCanny):
    minLineLength = 80
    lines = cv2.HoughLinesP(grayCanny, rho=1, theta=np.pi / 180, threshold=100, minLineLength = minLineLength, maxLineGap = 10)
    return lines

def findClockHands(lines, xCircleCenter, yCircleCenter, radius):
    line = []
    diff1LowerBound = 0.01 #diff1LowerBound and diff1UpperBound determine how close the line should be from the center
    diff1UpperBound = 0.45
    diff2LowerBound = 0.1 #diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
    diff2UpperBound = 1

    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            diff1 = dist_2_pts(xCircleCenter, yCircleCenter, x1, y1)  # x, y is center of circle
            diff2 = dist_2_pts(xCircleCenter, yCircleCenter, x2, y2)  # x, y is center of circle
            #set diff1 to be the smaller (closest to the center) of the two), makes the math easier
            if (diff1 > diff2):
                temp = diff1
                diff1 = diff2
                diff2 = temp
            # check if line is within an acceptable range
            if (((diff1<diff1UpperBound*radius) and (diff1>diff1LowerBound*radius) and (diff2<diff2UpperBound*radius)) and (diff2>diff2LowerBound*radius)):
                # add to final list
                dist = dist_2_pts(x1, y1, x2, y2)
                line.append([x1, y1, x2, y2, dist])

    line.sort(key = lambda x : x[-1])
    return line

def mergeClockHand(img, init_line_list):
    res = init_line_list[0] #res to contain the current line
    xAvg = (res[0] + res[2]) / 2 #Avg x coordinate of the line
    yAvg = (res[1] + res[3]) / 2 #Avg y coordinate of the line
    v1 = calvec(res) 
    final_line_list = [] #list to contain the final lines
    ch = 0
    for i in range(1,len(init_line_list)):
        v2 = calvec(init_line_list[i])
        deg = calcos(v1[0], v1[1], v2[0], v2[1])
    
        if (deg <= 2) or ((deg >= 178) and (deg <= 180)):
            res = init_line_list[i]
            ch = 1
        else:
            cv2.line(img, (res[0], res[1]), (res[2], res[3]), (0, 0, 255), 2)
            thickness = dist_2_pts(xAvg, yAvg, (res[0] + res[2]) / 2, (res[1] + res[3]) /2)
            length = dist_2_pts(res[0], res[1], res[2], res[3])
            final_line_list.append([res[0], res[1], res[2], res[3], thickness, length])
            xAvg = (init_line_list[i][0] + init_line_list[i][2]) / 2
            yAvg = (init_line_list[i][1] + init_line_list[i][3]) / 2
            res = init_line_list[i]
            ch = 0
        v1 = v2

    if (ch == 1):
        cv2.line(img, (res[0], res[1]), (res[2], res[3]), (0, 0, 255), 2)
        thickness = dist_2_pts(xAvg, yAvg, (res[0] + res[2]) / 2, (res[1] + res[3]) /2)
        length = dist_2_pts(res[0], res[1], res[2], res[3])
        final_line_list.append([res[0], res[1], res[2], res[3], thickness, length])

    return final_line_list

def calculateMinuteAndSecond(line, xCircleCenter, yCircleCenter):
    res = []
    # xCircleCenter, yCircleCenter is center of circle
    diff1 = dist_2_pts(xCircleCenter, yCircleCenter, line[0], line[1])  
    diff2 = dist_2_pts(xCircleCenter, yCircleCenter, line[2], line[3])
    if (diff1 < diff2):
        res.append([line[0], line[1], line[2], line[3]])
    else:
        res.append([line[2], line[3], line[0], line[1]])
    tmp = res[0]
    v = calvec(tmp)
    deg = calcos(v[0], v[1], 0, -100)

    if (deg <= 2):
        value = 0
    elif ((deg >= 178) and (deg <= 180)):
        value = 30
    else:
        if (tmp[2] < xCircleCenter):
            value = 60 - int(deg / 6)
        else:
            value = int(deg / 6)
    
    return value, deg

def detectTime(line, xCircleCenter, yCircleCenter):
    hour, minute, second = 0, 0, 0
    degH, degM, degS = 0, 0, 0
    line.sort(key = lambda x : x[-1])

    # Calculate hour
    res = []
    # xCircleCenter, yCircleCenter is center of circle
    diff1 = dist_2_pts(xCircleCenter, yCircleCenter, line[0][0], line[0][1])  
    diff2 = dist_2_pts(xCircleCenter, yCircleCenter, line[0][2], line[0][3])
    if (diff1 < diff2):
        res.append([line[0][0], line[0][1], line[0][2], line[0][3]])
    else:
        res.append([line[0][2], line[0][3], line[0][0], line[0][1]])
    tmp = res[0]
    v = calvec(tmp)
    deg = calcos(v[0], v[1], 0, -100)

    if (deg <= 2):
        hour = 0
    elif ((deg >= 178) and (deg <= 180)):
        hour = 30
    else:
        if (tmp[2] < xCircleCenter):
            if ((deg % 30) / 30 >= 0.25):
                hour = 12 - math.ceil(deg / 30)
            else:     
                hour = 12 - int(deg / 30)
        else:
            hour = int(deg / 30)  
    degH = deg

    # Calculate minute and second
    if (line[1][4] < line[2][4]):
        second, degS = calculateMinuteAndSecond(line[1], xCircleCenter, yCircleCenter)
        minute, degM = calculateMinuteAndSecond(line[2], xCircleCenter, yCircleCenter)
    else:
        second, degS = calculateMinuteAndSecond(line[2], xCircleCenter, yCircleCenter)
        minute, degM = calculateMinuteAndSecond(line[1], xCircleCenter, yCircleCenter)
    
    if (hour >= 10):
        hour = str(hour)
    else:
        hour = '0' + str(hour)

    if (minute >= 10):
        minute = str(minute)
    else:
        minute = '0' + str(minute)

    if (second >= 10):
        second = str(second)
    else:
        second = '0' + str(second)
    
    return hour, minute, second, degH, degM, degS

def main():
    # Read image and get image's size
    preImg, height, width = readImage('clock.jpeg')

    # Pre-process image
    img, grayColor, grayCanny = imageProcess(preImg)

    # Detect circle
    xCircleCenter, yCircleCenter, radius = findClock(img, grayColor, height)

    # Find all the lines in the image
    lines = lineDetect(grayCanny)

    # Find all the clock hands
    init_line_list = []
    init_line_list = findClockHands(lines, xCircleCenter, yCircleCenter, radius)

    # Find the 3 clock hands by merging all the lines in each clock hand
    line = mergeClockHand(img, init_line_list)

    # Calculate the time of the clock
    hour, minute, second, degH, degM, degS = detectTime(line, xCircleCenter, yCircleCenter)

    # Print the result
    print("--------------")
    print(f"H:{int(degH)}, M:{int(degM)}, S:{int(degS)}")
    print("--------------")
    print(f"{hour}:{minute}:{second}")
    print("--------------")

    # Draw the time on the clock image
    cv2.putText(img, hour + ":" + minute + ":" + second, (5, height - 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA, False)
    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__=='__main__':
    main()

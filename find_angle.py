import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

'''
1:Passed
2:Passed
3:Passed (w divide by 0 warning)
4:? (Passed but returned acute angle instead of obtuse)
5:? (Passed but returned obtuse angle instead of acute)
6: Passed (w divide by 0 warning)
7: Error - more than 2 lines found
8: Error - more than 2 lines found 
9: Passed
10: Error - more than 2 lines found
'''
def find_angle(png_path):
    '''
    Reads a png with opencv, runs canny edge detection and applies a hough transform to extract lines

    returns the angle between the 2 lines

    :param png_path: path to png image containing lines
    :return: angle between lines 
    '''
    # read image
    img = cv2.imread(png_path)
    # convert to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find edges
    edges = cv2.Canny(gray_img,0,10,L2gradient=True) # experiment with params
    '''
    check if way to remove double edges
    '''
    # apply a hough line transform
    '''
    Figure out way to select best threshold value / resolution based on thickness of lines / edges
    '''
    lines = cv2.HoughLines(edges,1,np.pi/180,90) 
    # array to store line gradients
    angles = []
    print(lines.shape)
    for line in lines:
        rho,theta = line[0]
        
        # calculate angle of inclination
        dydx = -np.cos(theta)/np.sin(theta)
        angle = np.arctan(dydx)
        angles.append(angle)
        
        # add line to img for visualising 
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    # show lines and edges
    plt.imshow(img)
    plt.show()
    plt.imshow(gray_img)
    plt.show()
    # remove repeat angles coming from parallel edges
    angles = list(set(angles))

    ''' at this step len angles should = 2, if more than 2 angles at this stage need another filtering step'''

    if len(angles) == 2:
        angle = np.abs(angles[0] - angles[1])
        angle = (180/np.pi) * angle # convert angle to degrees
        return angle
    
    else:
        print('[ERROR] More than 2 lines found')
        return sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("png_path", help="Path to a png image containing lines")
    args = parser.parse_args()

    theta = find_angle(args.png_path)
    print(f'angle found:{theta:.3f}')

if __name__ == "__main__":
    main()
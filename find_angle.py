import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as patches


'''
1:Passed
2:Passed
3:Passed (w divide by 0 warning)
4:Passed
5:Passed
6: Obtuse instead of acute (w divide by 0 warning)
7: Acute instead of obtuse
8: Error - more than 2 lines found 
9: Error - more than 2 lines found
10: Passed (w divide by 0 warning)
'''

'''
Assumes edges of equal length to set thresh (not yet)
check if min sep can be increased to acute_angle / 2


TODO: 
Check how m = inf is handled by acute checker
Check why case 7 is incorrectly classified as acute

Figure out how to handle inf gradients. Vectorisaiton of lines?
'''

def check_acute_seperation(acute_angle, edge_map, unique_lines):
    '''
    Checks seperation between edge pixels end points from intersection point to find whether acute or obtuse seperation
         
    Parameters
    ----------
    acute_angle : float
        acute angle between lines found from the hough transform
    
    edge_map : np.ndarray 
        edge map of input gray scale image
    
    unique_lines : list
        list containing gradient and y intercept parameter tuples for 2 distinct lines, [(m1,c1), (m2,c2)] .
    

    Returns
    -------
    acute_bool : bool
        Boolean value determined by whether or not seperation found to be acute
    '''
    # get index position array of edge pixels 
    pos_array = np.argwhere(edge_map)

    # calculate intersection point
    m1,c1 = unique_lines[0]
    m2,c2 = unique_lines[1]
    x_int = (c2 - c1)/(m1 - m2)
    y_int = m1*x_int + c1

    intersection_point = np.asarray([y_int,x_int]) # y=i, x=j, flipped for array indexing
    
    # find distance of edge pixels from intersection point 
    pos_diff_array = pos_array - intersection_point # find difference in i,j indices
    dist_array = np.sum(pos_diff_array**2,1)**0.5 # find abs difference 
    sep_array = [np.arctan(delta_j/delta_i) if delta_i != 0 else np.pi/2 for delta_i, delta_j in pos_diff_array]
    
    # find the pair of edge pixels that pass seperation threshold
    min_seperation = acute_angle/3 # 1/3rd since exact half sep might not exist 
    edge_point_1_idx = np.argmax(dist_array) # picks an edge end point as first edge
    
    edge_sep_array = np.abs(sep_array - sep_array[edge_point_1_idx])
    thresholded_edge_sep_array = np.where(edge_sep_array >= min_seperation, 1, 0)
    edge_point_2_idx = np.argmax(thresholded_edge_sep_array)
    
    # find angle between lines by adding seperation angles
    angle = np.abs(sep_array[edge_point_1_idx]) + np.abs(sep_array[edge_point_2_idx])
    
    # return acute bool
    if angle <= np.pi/2:
        return True

    else:
        return False
    



def find_angle(png_path):
    '''
    Reads a png with opencv, runs canny edge detection and applies a hough transform to extract lines
    Checks if acute or obtuse seperation and returns the appropriate angle.
    
    Parameters
    ----------
    png_path : str
        path to png image containing lines
    
    Returns
    -------
    angle : float
        angle between lines in degrees
    '''
    # read image
    img = cv2.imread(png_path)
    # convert to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find edges
    edges = cv2.Canny(gray_img,0,10,L2gradient=True) # experiment with params / auto canny https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    '''
    check if way to remove double edges / do we even need to?
    '''

    '''
    Figure out way to select best threshold value / resolution based on thickness of lines / edges
    currently calculating a bbox which contains the edge and setting thresh to the value of the box's diagonal
    '''
    # find edge bbox 
    pos_array = np.argwhere(edges)
    i_max, i_min = np.amax(pos_array[:,0]), np.amin(pos_array[:,0])
    j_max, j_min = np.amax(pos_array[:,1]), np.amin(pos_array[:,1])
    bbox_diag = ((i_max-i_min)**2 + (j_max-j_min)**2)**0.5

    thresh = int(round(bbox_diag/2)) # check if better relation between thresh and bbox_diag

    # apply a hough line transform
    lines = cv2.HoughLines(edges,1,np.pi/180,thresh) 
    
    # array to store line param tuples (m,c) 
    cartesian_line_param_list = []
    
    # loop over lines found via the hough line transform
    for line in lines:
        # extract rho theta values for each line
        rho,theta = line[0]
        cosx = np.cos(theta)
        sinx= np.sin(theta)

        # calc line cartesian params
        '''
        where is the angle measured from? Verify if -ve theta always works / why to get m
        '''
        m = -1/np.tan(theta) 
        x0 = cosx*rho
        y0 = sinx*rho
        c = y0 - m*x0         
        
        # append param tuple to list
        cartesian_line_param_list.append((m,c))
        
        # add line to img for visualising
        '''understand how this works  / check what happens if cos and sinx mixed up?'''
        x1 = int(x0 + 1000*(-sinx))
        y1 = int(y0 + 1000*(cosx))
        x2 = int(x0 - 1000*(-sinx))
        y2 = int(y0 - 1000*(cosx))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    # show lines and edges
    plt.imshow(img)
    plt.show()

    # remove parallel lines from cartesian_line_param_list
    '''re write this better'''
    m_list = []
    unique_line_idxs = []
    unique_lines = []
    for idx,(m,c) in enumerate(cartesian_line_param_list):
        if m not in m_list: 
            m_list.append(m)
            unique_line_idxs.append(idx)

    unique_lines = [cartesian_line_param_list[i] for i in unique_line_idxs]
    
    ''' at this step len angles should = 2, if more than 2 angles at this stage need another filtering step'''
    if len(unique_lines) == 2:
        # calculate angle of incidences to find angle between lines
        m1,c1 = unique_lines[0]
        m2,c2 = unique_lines[1]
        angle_1 = np.arctan(m1)
        angle_2 = np.arctan(m2)
        # find absolute acute angle 
        angle = np.abs(angle_1-angle_2) 
        acute_angle = np.pi - angle if angle>np.pi/2 else angle
        # check if acute seperation true
        acute_bool = check_acute_seperation(acute_angle, edges, unique_lines)
        # return angle in degrees
        if acute_bool: 
            return 180/np.pi * acute_angle
        else: 
            return 180/np.pi * (np.pi - acute_angle)
    
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

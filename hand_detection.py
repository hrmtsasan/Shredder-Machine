import cv2
import argparse
import orien_lines
import datetime
from imutils.video import VideoStream
from utils import detector_utils as detector_utils
import pandas as pd
from datetime import date
import xlrd                    ## We are trying to save this report into excel sheet as well. That's why we r calling xlrd.
from xlwt import Workbook
from xlutils.copy import copy 
import numpy as np

lst1=[]
lst2=[]
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()       ## This will be the starting point. Here graph will be loaded in the particular session.

def save_data(no_of_time_hand_detected, no_of_time_hand_crossed):

    try:   
        today = date.today()
        today=str(today)
        #loc = (r'C:\Users\rahul.tripathi\Desktop\result.xls') 
      
        rb = xlrd.open_workbook('result.xls')
        sheet = rb.sheet_by_index(0) 
        sheet.cell_value(0, 0) 
      
         
        #print(sheet.nrows)
        q=sheet.cell_value(sheet.nrows-1,1)
        
        rb = xlrd.open_workbook('result.xls') 
        #rb = xlrd.open_workbook(loc) 
        wb=copy(rb)
        w_sheet=wb.get_sheet(0)
        
        if q==today:
            w=sheet.cell_value(sheet.nrows-1,2)
            e=sheet.cell_value(sheet.nrows-1,3)
            w_sheet.write(sheet.nrows-1,2,w+no_of_time_hand_detected)
            w_sheet.write(sheet.nrows-1,3,e+no_of_time_hand_crossed)
            wb.save('result.xls')      
        else:
            w_sheet.write(sheet.nrows,0,sheet.nrows)
            w_sheet.write(sheet.nrows,1,today)
            w_sheet.write(sheet.nrows,2,no_of_time_hand_detected)
            w_sheet.write(sheet.nrows,3,no_of_time_hand_crossed)
            wb.save('result.xls')
    except FileNotFoundError:
        today = date.today()
        today=str(today)
         

        # Workbook is created 
        wb = Workbook() 

        # add_sheet is used to create sheet. 
        sheet = wb.add_sheet('Sheet 1') 

        sheet.write(0, 0, 'Sl.No')
        sheet.write(0, 1, 'Date') 
        sheet.write(0, 2, 'Number of times hand detected') 
        sheet.write(0, 3, 'Number of times hand crossed') 
        m=1
        sheet.write(1, 0, m)
        sheet.write(1, 1, today) 
        sheet.write(1, 2, no_of_time_hand_detected) 
        sheet.write(1, 3, no_of_time_hand_crossed) 

        wb.save('result.xls')
        
if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.80                         ## If the detection accuracy is more than 80% only in that case it'll show you the bounding boxes.
    
    #vs = cv2.VideoCapture('rtsp://192.168.1.64')
    vs = VideoStream(0).start()                 ## It will start the videostream from port 0.
    #Oriendtation of machine    
    Orientation= 'bt'
	#input("Enter the orientation of hand progression ~ lr,rl,bt,tb :")

    #For Machine
    #Line_Perc1=float(input("Enter the percent of screen for the line of machine :"))
    Line_Perc1=float(15)

    #For Safety
    #Line_Perc2=float(input("Enter the percent of screen for the line of safety :"))
    Line_Perc2=float(30)

    # max number of hands we want to detect/track
    num_hands_detect = 2

    # Used to calculate fps
    start_time = datetime.datetime.now()               ## To keep the information of when person hand has crossed the line.
    num_frames = 0

    im_height, im_width = (None, None)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)     ## Creating the screen that will pop up.
    def count_no_of_times(lst):                         ## Count no. of times we are passing the frames.
        x=y=cnt=0
        for i in lst:
            x=y
            y=i
            if x==0 and y==1: 
                cnt=cnt+1
        return cnt 
    try:
        while True:
            frame = vs.read()                 ## From vedio we will keep on reading the frame one by one.
            frame = np.array(frame)           ## Convert that frame into array.
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            if im_height == None:             ## im_height == None means if there is no frame.
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")
            #cv2.line(img=frame, pt1=(0, Line_Position1), pt2=(frame.shape[1], Line_Position1), color=(255, 0, 0), thickness=2, lineType=8, shift=0)
            
            #cv2.line(img=frame, pt1=(0, Line_Position2), pt2=(frame.shape[1], Line_Position2), color=(255, 0, 0), thickness=2, lineType=8, shift=0)

            ## From video streaming we will get a frame and then we will pass this frame to the model.

            # Run image through tensorflow graph
            ## This is the function which is responsible for doing detection out of that particular graph.
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)                        ## We are passing the session bcoz in tensorflow 1.x we are supposed to call everything inside the session . Everything should happen inside the same session , we aren't supposed to create another session.
            
            Line_Position2=orien_lines.drawsafelines(frame,Orientation,Line_Perc1,Line_Perc2)             
            # Draw bounding boxeses and text
            a,b=detector_utils.draw_box_on_image(
                num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame,Line_Position2,Orientation)
            lst1.append(a)
            lst2.append(b)
            no_of_time_hand_detected=no_of_time_hand_crossed=0
            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time

            if args['display']:
            
                # Display FPS on frame
                detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows() 
                    vs.stop()
                    break
        
        no_of_time_hand_detected=count_no_of_times(lst2)
        #no_of_time_hand_detected=b
        no_of_time_hand_crossed=count_no_of_times(lst1)
        #print(no_of_time_hand_detected)
        #print(no_of_time_hand_crossed)
        save_data(no_of_time_hand_detected, no_of_time_hand_crossed) 
        print("Average FPS: ", str("{0:.2f}".format(fps)))
        
    except KeyboardInterrupt: 
        no_of_time_hand_detected=count_no_of_times(lst2)
        no_of_time_hand_crossed=count_no_of_times(lst1)
        today = date.today()
        save_data(no_of_time_hand_detected, no_of_time_hand_crossed)
        print("Average FPS: ", str("{0:.2f}".format(fps)))
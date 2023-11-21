"""
Python Class to display annotated PCL files with colored bounding boxes
Using o3d library for visualization
Ethan Gage
Prabin Rath
Fall 2023
"""
import open3d as o3d
import numpy as np
import math
import time
import os
import csv
from transformations import euler_matrix


class ScenarioAnnotationVisualizer(object):
    def __init__(self, data_path,intersection):
        self.data_path = data_path
        self.earliest_frame = 9999999
        self.pcl = []
        self.boundingBoxes = {}
        self.boxes = []
        if intersection == "Intersection1":
            self.carlaLocation = [18,-124,13]
            self.carlaRotation = [-30,-133,0]
        elif intersection == "Intersection2":
            self.carlaLocation = [17.073307,11.931071,7.059668]
            self.carlaRotation = [-16.686125,-39.844498,0]
        elif intersection == "Intersection3":
            self.carlaLocation = [-62.299007, 2.646346, 6.799613] 
            self.carlaRotation = [-24.607510, 46.940891, 0]
        elif intersection == "Intersection4":
            self.carlaLocation = [-98.378456, 145.841202, 7.029282] 
            self.carlaRotation = [-19.377438, -39.743271, 0]
    
    
    def transform_points(self, points, xyz, rpy):
        x, y, z = xyz[0],xyz[1],xyz[2]
        roll, pitch, yaw = rpy[0],rpy[1],rpy[2]
        pos = points[:,:3]
        M = euler_matrix(roll, pitch, yaw).astype(np.float32)
        M[0:3,3] = np.array([x, y, z])
        pos = np.hstack((pos, np.ones((pos.shape[0],1), dtype=np.float32)))
        pos = (M @ pos.T).T
        points[:,:3] = pos[:,:3]
        return points
    
    #Returns array of detected vehicle and pedestrian bounding box data, to store in boundingBox dictionary
    def getBoundingBoxData(self,frame):
        #create empty box
        boxList = []
        filename = self.data_path + str(frame) + '.csv'
        with open(filename,'r') as csvfile:
            datareader = csv.reader(csvfile)
            for row in datareader:
                if row[0] == 'Object ID':
                    continue
                else:
                    #if object is person or car
                    if row[1] == '4' or row[1] == '10':
                        #add list to
                        tArr = [0,0,0,0,0,0,0,0,0] 
                        #Type cast from STR array to respective data type:
                        #[Object ID(int),Semantic Tag(int),XYZ(float),Yaw(float),Length,Width,Height(float)]
                        tArr[0] = int(row[0])  #ID
                        tArr[1] = int(row[1])   #Semantic Tag
                        tArr[2] = float(row[2]) #X
                        tArr[3] = float(row[3]) #Y
                        tArr[4] = float(row[4]) #Z
                        tArr[5] = float(row[5]) #Yaw
                        tArr[6] = float(row[6]) #Length
                        tArr[7] = float(row[7]) #Width
                        tArr[8] = float(row[8]) #Height
                        boxList.append(tArr)
        return boxList
    
    #Creates Bounding Box o3d geometry with the center of the box, rotation matrix, and size passed as np arrays, and semantic tag passed as int for coloring
    #Returns OrienctedBoundingBox o3d geometry object
    def makeBoundingBox(self, center, r, size,tag):
        
        sample_3d = o3d.geometry.OrientedBoundingBox(center, r, size)
        #Colors bounding box based on tag, Green for Cars, Red for Pedestrians
        if(tag == 4):
            sample_3d.color = [1,0,0]
        elif tag == 10:
            sample_3d.color = [0,1,0]
        else:
            sample_3d.color = [0,0,0]
        return sample_3d
    
    def getData(self):
        #iterate over files in data_path
        for filename in os.listdir(self.data_path): #pt_clouds are the point cloud data from several .pcf files
            f = os.path.join(self.data_path,filename)
            #check for valid file
            if os.path.isfile(f):
                #only grab .ply
                if f.endswith('.csv'):
                    frame = f[f.find('/')+1:]
                    frame = frame.strip('.csv')
                    #print(frame)
                    #setup boundingBox dictionary with frame number as key
                    #used getBoundingBoxData() to populate dictionary with data
                    print(frame)
                    self.boundingBoxes[int(frame)] = self.getBoundingBoxData(int(frame))
                    if int(frame) < self.earliest_frame:
                        #store earliest frame ID to start iterative process
                        self.earliest_frame = int(frame)
                        #print(frame)
        print(self.earliest_frame)
    
    def transform_pointcloud(self,pcd):
        #Convert PCD to Numpy array
        pointcloud = np.asarray(pcd.points)

        #Reverse Y coors to reflect Carla using LHR
        pointcloud[:,1]*=-1

        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        
        #correct Pitch
        xyz = [0,0,0]
        rpy = [0,((-1*self.carlaRotation[0])/180)*np.pi,0]
        pcd.points = o3d.utility.Vector3dVector(self.transform_points(np.asarray(pcd.points),xyz,rpy))

        #correct Yaw
        xyz = [0,0,0]
        rpy = [0,0,((-1*self.carlaRotation[1])/180)*np.pi]
        pcd.points = o3d.utility.Vector3dVector(self.transform_points(np.asarray(pcd.points),xyz,rpy))

        #Correct Translation
        xyz = [(self.carlaLocation[0]),(-1*self.carlaLocation[1]),self.carlaLocation[2]]
        rpy = [0,0,0]
        pcd.points = o3d.utility.Vector3dVector(self.transform_points(np.asarray(pcd.points),xyz,rpy))
        
        return pcd
    
    def displayData(self):
        print("Earliest frame: "+ str(self.earliest_frame))
        #Get Lidar transform 
        # XYZ = [self.lidar_transforms[0][0],self.lidar_transforms[0][1],self.lidar_transforms[0][2]]
        # RPY = [self.lidar_transforms[0][3],self.lidar_transforms[0][4],self.lidar_transforms[0][5]]
        # print(XYZ)
        # print(RPY)
        # #inverse transform to correct PCD points
        # RPY = RPY[0]*-1,RPY[1]*-1,RPY[2]*-1
        # XYZ = XYZ[0]*-1,XYZ[1]*-1,XYZ[2]*-1
        #Display earliest PCD file first
        current_frame = self.earliest_frame
        leading_zeroes = ''
        for i in range(6 - len(str(current_frame))):
            leading_zeroes = leading_zeroes + '0'
        first_file = self.data_path + leading_zeroes + str(current_frame) + '.ply'
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        pcd = o3d.io.read_point_cloud(first_file)
        pcd = self.transform_pointcloud(pcd)

        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        for i in range(100):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)
        #interate through the 300 ply files
        for i in range (299):
            #remove previous bounding boxes
            for box in self.boxes:
                vis.remove_geometry(box,reset_bounding_box=False)
            vis.update_renderer()
            self.boxes = []
            leading_zeroes = ''
            for i in range(6 - len(str(current_frame))):
                leading_zeroes = leading_zeroes + '0'
            current_file = self.data_path + leading_zeroes + str(current_frame) + '.ply'
            pcd.points = o3d.io.read_point_cloud(current_file).points
            pcd = self.transform_pointcloud(pcd)
            vis.update_geometry(pcd)

            #Iterate through boundingBoxes dictionary, make list of OrientedBoundingBoxes
            for box in self.boundingBoxes[current_frame]:
                # center = np.array([0, 0, 0])
                # r = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                # size = np.array([10, 10, 10]
                # tArr[0] = int(row[0])  #ID
                # tArr[1] = int(row[1])   #Semantic Tag
                # tArr[2] = float(row[2]) #X
                # tArr[3] = float(row[3]) #Y
                # tArr[4] = float(row[4]) #Z
                # tArr[5] = float(row[5]) #Yaw
                # tArr[6] = float(row[6]) #Length
                # tArr[7] = float(row[7]) #Width
                # tArr[8] = float(row[8]) #Height
                center = np.array([box[2],-1*box[3],box[4]+box[8]/2])               #Extract box XYZ coord, store in center
                yaw = (-box[5]/180)*np.pi
                r = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])    #Store Box Yaw in rotation matrix
                size = np.array([box[6], box[7], box[8]])               #Store Box's size
                tag = box[1]                                            #Store box's semantic tag
                self.boxes.append(self.makeBoundingBox(center, r, size, tag))
            #Add all boxes to o3d window
            for box in self.boxes:
                vis.add_geometry(box,reset_bounding_box = False)

            vis.poll_events()
            vis.update_renderer()
            
            current_frame+=1
            time.sleep(0.1)

# lidar_location = carla.Location(18,-124,13)  #Top of lightpost
# lidar_rotation = carla.Rotation(-30,-133,0)  [P,Y,R]

lidar_transforms = [[18,-124,13,0,(-30/180)*np.pi,(-133/180)*np.pi]] #lidar_transforms[lidar1[X,Y,Z,R,P,Y]]  XYZ in Meters, RPY in Radians Positive transform needs to be inverted to convert PCD

test = ScenarioAnnotationVisualizer('test_data21/',"Intersection4")
test.getData()
test.displayData()
#!/usr/bin/env python

"""
Python Class to run Carla Scenario Runner to generate 
Synthetic Annotated Lidar data to train ML models

Ethan Gage
Prabin Rath
Fall 2023
"""
import carla 
import math 
import random 
import time 
import numpy as np
import sys
    
import cv2
import open3d as o3d
from matplotlib import cm
from queue import Queue
from queue import Empty

class CarlaScenarioRunner(object):

    #default contructor
    def __init__(self, data_path, intersection):
        #Get data path to store generated files
        self.data_path = data_path
        self.intersection = intersection
        #Connect to Carla client 
        self.client = carla.Client('localhost', 2000)
        
        #Load town, set spawn points based on passed intersection argument
        if intersection == "Intersection1":
            self.client.load_world('Town03')
            self.vehicle_spawns = [ carla.Transform(carla.Location(0.700499, -189.727951, 0.039138),carla.Rotation(0,91.413528,0)),
                                    carla.Transform(carla.Location(0, -161.2, 0.003371),carla.Rotation(0,91.435211,0)),
                                    carla.Transform(carla.Location(5.778288, -113.229492, 0.067768),carla.Rotation(0,-88.876099,0)),
                                    carla.Transform(carla.Location(-3.098442, -183.513992, -0.006345),carla.Rotation(-0.003415,91.413528,0)),
                                    carla.Transform(carla.Location(-24.311588, -135.292282, -0.006437),carla.Rotation(-0.000061,1.226963,0.001293)),
                                    carla.Transform(carla.Location(9.184540, -101.743164, 0.000199),carla.Rotation(0,-88.586418,0)),
                                    carla.Transform(carla.Location(73.203979, -136.704651, 7.996281),carla.Rotation(-1.480286,-178.773956,-0.116272)),
                                    carla.Transform(carla.Location(73.266441, -139.972733, 8.045337),carla.Rotation(-1.480286,-178.773956,-0.116272)),
                                    carla.Transform(carla.Location(-15.723951, -131.572281, 0.279280),carla.Rotation(0,1.2,0)),
                                    carla.Transform(carla.Location(10.139042, -146.582550,-0.007044),carla.Rotation(0.002227,-88.586418,-0.000366)),
                                    carla.Transform(carla.Location(16.879513, -134.409973,0.066324),carla.Rotation(1.567317,1.230288,0.030370)),
                                    carla.Transform(carla.Location(-4.364792, -126.337173,0.000056),carla.Rotation(0,91.413528,0)),
                                    carla.Transform(carla.Location(-11.102114, -138.510193,-0.026897),carla.Rotation(0,-178.772690,0))]
            self.pedestrian_spawns = [[-13,-124,1.18],
                                    [16,-145,1.18],
                                    [-13,-145,1.198]]
        elif intersection == "Intersection2":
            self.client.load_world('Town05')
            self.vehicle_spawns = [ carla.Transform(carla.Location(28.240622, -30.186789, -0.002115), carla.Rotation(0.000000, 91.532082, 0.000000)),
                                    carla.Transform(carla.Location(41.941372, 5.246575, -0.005409), carla.Rotation(-0.000123, 0.250286, -0.000000)),
                                    carla.Transform(carla.Location(31.399370, -12.340173, 0.003142), carla.Rotation(0.323457, -89.800964, 0.000000)),
                                    carla.Transform(carla.Location(24.597328, 14.145382, -0.002615), carla.Rotation(0.000000, 89.488258, 0.000000)),
                                    carla.Transform(carla.Location(17.847080, -4.494134, 0.001719), carla.Rotation(0.000102, 179.860489, 0.000001)),
                                    carla.Transform(carla.Location(28.082001, 12.414189, 0.000234), carla.Rotation(0.000000, 89.488258, 0.000000)),
                                    carla.Transform(carla.Location(31.599438, 55.988182, 0.002890), carla.Rotation(0.000000, -89.977112, 0.000000)),
                                    carla.Transform(carla.Location(0.547553, 2.547991, -0.002569), carla.Rotation(0.000000, -0.139465, 0.000000)),
                                    carla.Transform(carla.Location(35.099331, 30.202101, -0.002520), carla.Rotation(0.000000, -89.977112, 0.000000)),
                                    carla.Transform(carla.Location(24.706718, -28.981133, -0.030267), carla.Rotation(-0.041992, 91.531876, 0.002927)),
                                    carla.Transform(carla.Location(0.562071, -4.457942, -0.000382), carla.Rotation(0.000000, 179.860489, 0.000000)),
                                    carla.Transform(carla.Location(-0.929380, -0.944301, 0.008695), carla.Rotation(0.000000, 179.860489, 0.000000)),
                                    carla.Transform(carla.Location(-0.943924, 6.051632, -0.011420), carla.Rotation(0.000000, -0.139465, 0.000000)),
                                    carla.Transform(carla.Location(34.907337, -14.626545, -0.002194), carla.Rotation(0.000000, -89.800896, 0.000000)),
                                    carla.Transform(carla.Location(16.355600, -0.990493, -0.011786), carla.Rotation(0.000000, 179.860489, 0.000000)),
                                    carla.Transform(carla.Location(31.599438, 31.600700, 0.042888), carla.Rotation(0.000000, -89.977112, 0.000000)),
                                    carla.Transform(carla.Location(41.056709, 1.742319, -0.054240), carla.Rotation(0.001229, 0.250308, -0.042267))]
            self.pedestrian_spawns = [[39.423889, 9.980207, 1.039616],
                                    [38.450214,-11.755219,0.461922],
                                    [18.761215, -8.827444, 0.410718],
                                    [17.712217, 11.860162, 0.413099]]
        elif intersection == "Intersection3":
            self.client.load_world('Town10HD_Opt')
            self.vehicle_spawns = [carla.Transform(carla.Location(x=-41.668877, y=48.905540, z=-0.005281), carla.Rotation(pitch=0.000000, yaw=-90.161186, roll=0.000000)),
                                    carla.Transform(carla.Location(x=-20.115099, y=16.748653, z=-0.007563), carla.Rotation(pitch=-0.003142, yaw=-179.840790, roll=-0.001862)),
                                    carla.Transform(carla.Location(x=-79.275810, y=27.963949, z=0.000450), carla.Rotation(pitch=0.001018, yaw=0.159203, roll=-0.000580)),
                                    carla.Transform(carla.Location(x=-87.276062, y=24.441530, z=0.000106), carla.Rotation(pitch=0.000000, yaw=0.159198, roll=0.000000)),
                                    carla.Transform(carla.Location(x=-17.105413, y=13.259648, z=-0.000360), carla.Rotation(pitch=0.000096, yaw=-179.841064, roll=0.026100)),
                                    carla.Transform(carla.Location(x=-48.839581, y=-17.213196, z=-0.018568), carla.Rotation(pitch=0.000560, yaw=90.432365, roll=-0.001587)),
                                    carla.Transform(carla.Location(x=-52.330765, y=-28.861359, z=0.004023), carla.Rotation(pitch=-0.000273, yaw=90.432327, roll=0.000227)),
                                    carla.Transform(carla.Location(x=-45.239609, y=54.258713, z=0.346443), carla.Rotation(pitch=-11.990997, yaw=-89.315338, roll=0.000218)),
                                    carla.Transform(carla.Location(x=-67.254601, y=27.963617, z=-0.018216), carla.Rotation(pitch=0.001106, yaw=0.159208, roll=-0.002350)),
                                    carla.Transform(carla.Location(x=-64.103928, y=16.506121, z=-0.018244), carla.Rotation(pitch=0.001147, yaw=-179.840775, roll=-0.002533)),
                                    carla.Transform(carla.Location(x=-48.819988, y=-4.795075, z=0.033633), carla.Rotation(pitch=0.000000, yaw=89.838760, roll=0.000000)),
                                    carla.Transform(carla.Location(x=-66.794197, y=12.998388, z=0.033516), carla.Rotation(pitch=0.000000, yaw=-179.840790, roll=0.000000)),
                                    carla.Transform(carla.Location(x=-52.310936, y=-1.585238, z=-0.011521), carla.Rotation(pitch=0.000000, yaw=89.838760, roll=0.000000)),
                                    carla.Transform(carla.Location(x=-64.644875, y=24.470966, z=0.039431), carla.Rotation(pitch=0.000294, yaw=0.159205, roll=-0.001251)),
                                    carla.Transform(carla.Location(x=-52.186646, y=42.565102, z=-0.018517), carla.Rotation(pitch=0.000260, yaw=89.838760, roll=-0.000854)),
                                    carla.Transform(carla.Location(x=-48.674461, y=46.955261, z=0.000653), carla.Rotation(pitch=0.000608, yaw=89.838760, roll=-0.000305)),
                                    carla.Transform(carla.Location(x=-28.726021, y=28.104218, z=0.140955), carla.Rotation(pitch=0.000000, yaw=0.159198, roll=0.000000)),
                                    carla.Transform(carla.Location(x=-25.516315, y=24.613102, z=-0.004840), carla.Rotation(pitch=0.000116, yaw=0.159194, roll=-0.001099))]
            self.pedestrian_spawns = [[-58.683865, 3.466921, 1.252516], 
                                        [-34.722759, 4.599870, 1.423768], 
                                        [-35.731003, 37.223576, 0.987376], 
                                        [-59.768127, 35.494572, 0.776283]
]
        elif intersection == "Intersection4":
            self.client.load_world('Town03')
            self.vehicle_spawns = [carla.Transform(carla.Location(x=-95.444824, y=136.115875, z=-0.004154), carla.Rotation(pitch=0.000000, yaw=-0.597992, roll=0.000000)),
                                carla.Transform(carla.Location(x=-87.967072, y=87.134247, z=0.591078), carla.Rotation(pitch=-1.020294, yaw=89.787354, roll=0.001010)),
                                carla.Transform(carla.Location(x=-43.270969, y=131.164963, z=-0.001584), carla.Rotation(pitch=0.000000, yaw=176.631271, roll=0.000000)),
                                carla.Transform(carla.Location(x=-88.254883, y=106.199615, z=0.246260), carla.Rotation(pitch=-0.931877, yaw=89.787483, roll=0.000917)),
                                carla.Transform(carla.Location(x=-85.144127, y=144.537506, z=0.005812), carla.Rotation(pitch=-0.142792, yaw=89.787666, roll=0.000000)),
                                carla.Transform(carla.Location(x=-59.360752, y=135.466934, z=0.033657), carla.Rotation(pitch=0.000000, yaw=-1.296814, roll=0.000000)),
                                carla.Transform(carla.Location(x=-77.527092, y=109.821274, z=0.204491), carla.Rotation(pitch=0.770281, yaw=-90.150970, roll=-0.054718)),
                                carla.Transform(carla.Location(x=-88.056679, y=66.842804, z=0.824996), carla.Rotation(pitch=0.114932, yaw=89.787201, roll=-0.000031)),
                                carla.Transform(carla.Location(x=-149.063583, y=107.705582, z=-0.005027), carla.Rotation(pitch=0.000000, yaw=89.622032, roll=0.000000)),
                                carla.Transform(carla.Location(x=-101.632477, y=132.628662, z=-0.000240), carla.Rotation(pitch=0.000000, yaw=178.703156, roll=0.000000)),
                                carla.Transform(carla.Location(x=-84.654854, y=114.390144, z=0.134617), carla.Rotation(pitch=-0.661101, yaw=89.785408, roll=0.060561)),
                                carla.Transform(carla.Location(x=-74.027000, y=119.116455, z=0.065049), carla.Rotation(pitch=0.330588, yaw=-90.157822, roll=-0.000031)),
                                carla.Transform(carla.Location(x=-121.301514, y=136.415878, z=0.067766), carla.Rotation(pitch=0.000000, yaw=-0.597992, roll=0.000000)),
                                carla.Transform(carla.Location(x=-56.303329, y=131.564163, z=-0.019611), carla.Rotation(pitch=-0.000307, yaw=176.631271, roll=0.001366)),
                                carla.Transform(carla.Location(x=-88.644112, y=149.850555, z=-0.000006), carla.Rotation(pitch=0.000000, yaw=89.787704, roll=0.000000)),
                                carla.Transform(carla.Location(x=-60.503540, y=187.431732, z=0.009144), carla.Rotation(pitch=0.000000, yaw=-144.446411, roll=0.000000)),
                                carla.Transform(carla.Location(x=-61.637417, y=190.780136, z=-0.028799), carla.Rotation(pitch=-0.175440, yaw=-144.447174, roll=0.000002))]
            self.pedestrian_spawns = [ [-66.479980, 124.073990, 1.497562], 
                                        [-95.097008, 144.398834, 1.179993], 
                                        [-67.918472, 143.308441, 1.794224], 
                                        [-94.995750, 124.192848, 0.784548]]
        else:
            print("Invalid Intersection")

        

        #Grab bps, spawn points, and settings
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library() 
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.original_settings = self.world.get_settings()

        
        #setup sensor data queue
        self.sensorQueue = Queue()

    #Spawns numVehicles number of random vehicles in the carla world
    #First filling the predefined locations around the specific intersection, then filling in random spawn location with overflow   
    def spawn_vehicles(self, numVehicles):
        failedSpawns = 0
        if numVehicles <= len(self.vehicle_spawns):
            for i in range(numVehicles):
                vehicle_bp = random.choice(self.bp_lib.filter('vehicle'))
                if self.world.try_spawn_actor(vehicle_bp, random.choice(self.vehicle_spawns)) == None:
                    failedSpawns+=1

        else:
            remVehicles = numVehicles - len(self.vehicle_spawns)
            for i in range(len(self.vehicle_spawns)):
                vehicle_bp = random.choice(self.bp_lib.filter('vehicle'))
                if self.world.try_spawn_actor(vehicle_bp, random.choice(self.vehicle_spawns)) == None:
                    failedSpawns+=1
            for i in range(remVehicles):
                vehicle_bp = random.choice(self.bp_lib.filter('vehicle'))
                self.world.try_spawn_actor(vehicle_bp, random.choice(self.spawn_points))
        for i in range(failedSpawns):
            vehicle_bp = random.choice(self.bp_lib.filter('vehicle'))
            self.world.try_spawn_actor(vehicle_bp, random.choice(self.spawn_points))
        
    def spawn_pedestrians(self, numPeds):
        if numPeds <= 4:
            pedestrians = numPeds
            
        else:
            pedestrians = 4
        for i in range(pedestrians):
                pedestrian_bp = random.choice(self.bp_lib.filter('*walker.pedestrian*'))
                location = random.choice(self.pedestrian_spawns)
                transform = carla.Transform(carla.Location(location[0],location[1],location[2]))
                self.world.try_spawn_actor(pedestrian_bp, transform)

    def spawn_camera(self):
        camera_bp = self.bp_lib.find('sensor.camera.instance_segmentation')
        if self.intersection == "Intersection1":
            location = carla.Location(18,-124,13)
            rotation = carla.Rotation(-30,-133,0)
        else:
            location = carla.Location(18,-124,13)
            rotation = carla.Rotation(-30,-133,0)
        camera_init_trans = carla.Transform(location,rotation)
        self.image_width = camera_bp.get_attribute("image_size_x").as_int()
        self.image_height = camera_bp.get_attribute("image_size_y").as_int()
        
        self.camera = self.world.spawn_actor(camera_bp,camera_init_trans,attach_to=None)
        

    def spawn_lidar(self,model):
        #Get simulated Semantic Lidar BP
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')

        #Set parameters based on lidar model
        if model == 'Ouster':
            lidar_bp.set_attribute('channels', '64')
            lidar_bp.set_attribute('range', '120.0')
            lidar_bp.set_attribute('points_per_second', '1310720')
            lidar_bp.set_attribute('rotation_frequency', '20.0')
            lidar_bp.set_attribute('upper_fov', '16.6')
            lidar_bp.set_attribute('lower_fov', '-16.6')
            lidar_bp.set_attribute('horizontal_fov', '360.0')
            lidar_bp.set_attribute('sensor_tick', '0.0')
        else:
            lidar_bp.set_attribute('channels', '16')
            lidar_bp.set_attribute('range', '100.0')
            lidar_bp.set_attribute('points_per_second', '1310720')
            lidar_bp.set_attribute('rotation_frequency', '20.0')
            lidar_bp.set_attribute('upper_fov', '16.6')
            lidar_bp.set_attribute('lower_fov', '-16.6')
            lidar_bp.set_attribute('horizontal_fov', '360.0')
            lidar_bp.set_attribute('sensor_tick', '0.0')
        
        #Set Lidar transform based on called intersection
        if self.intersection == "Intersection1":
            xyz = [18,-124,13]
            rpy = [0,-30,-133]
        else:
            xyz = [18,-124,13]
            rpy = [0,-30,-133]
        #Set spawn location and orientation with given transformation
        lidar_location = carla.Location(xyz[0],xyz[1],xyz[2])  #Location, Carla uses LHR 
        lidar_rotation = carla.Rotation(rpy[1],rpy[2],rpy[0])  #Rotation, Carla uses PYR
        lidar_init_trans = carla.Transform(lidar_location,lidar_rotation)

        #spawn lidar
        self.semantic_lidar = self.world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=None)
    
    def sensor_callback(self, sensor_data, sensor_queue):
        # Do stuff with the sensor_data data like save it to disk
        # Then you just need to add to the queue
        sensor_queue.put(sensor_data)
    
    def camera_callback(self,image, data):
        data['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        image.save_to_disk(self.data_path + 'camera/%.6d.png' % image.frame)

   
    def gather_simulation_data(self, simTime):
        #Code to Run simulation for simTime seconds at 20 ticks/seconds
        #Each tick captures Lidar data, and exports data as .ply and .csv files 
        #for pointcloud, and detected objects location/bounding box info
        original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        

        #Define camera data value to be pushed to callback function
        camera_data = {'image': np.zeros((self.image_height,self.image_width, 4))}

        #counting variable
        t = 1
        

        #Setup Lidar data queue
        self.sensorQueue = Queue()

        #Set Spectator to look where lidar is looking
        self.world.get_spectator().set_transform(self.semantic_lidar.get_transform())


        #Apply settings
        self.world.apply_settings(settings)
        
        #Turn on all vehicle AI
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            vehicle.set_autopilot(True)

        #setup callback to push data to queue when available
        self.semantic_lidar.listen(lambda data: self.sensor_callback(data, self.sensorQueue))
        self.camera.listen(lambda image: self.camera_callback(image, camera_data))
        self.world.tick()
        while t <= simTime / 0.05:
                        
            #retrieve first in queue Lidar Data
            data = self.sensorQueue.get(True,1.0)

            #save .ply file
            data.save_to_disk(self.data_path + '%.6d.ply' % data.frame)
            
            #create .csv
            filename = self.data_path + str(data.frame) + ".csv"
            file = open(filename, "w")
            file.write("Object ID,Semantic Tag,X,Y,Z,Yaw,Length,Width,Height")
            file.write('\n')
            
            id_list = []
            for detection in data:
                try:
                    if detection.object_idx not in id_list and detection.object_idx != self.world.get_spectator().id:
                        #add id to unique list
                        id_list.append(detection.object_idx)
                        
                        #temp actor assigned for ease of coding
                        actor = self.world.get_actor(detection.object_idx)
            
                        #write object ID to first column of csv
                        file.write(str(detection.object_idx))
                        file.write(',')
                        file.write(str(actor.semantic_tags[0]))
                        file.write(',')
                        #Write Transform information for found object
                        transform = actor.get_transform()
                        file.write(str(transform.location.x))
                        file.write(',')
                        file.write(str(transform.location.y))
                        file.write(',')
                        file.write(str(transform.location.z))
                        file.write(',')
                        file.write(str(transform.rotation.yaw))
                        file.write(',')
                        
                        #Write Bounding Box information for found object
                        bounding = actor.bounding_box
                        
                        length = bounding.extent.x*2
                        width = bounding.extent.y*2
                        height = bounding.extent.z*2
                    
                        file.write(str(length))
                        file.write(',')
                        file.write(str(width))
                        file.write(',')
                        file.write(str(height))
                                    
                        #make new line for next ID
                        file.write('\n')
                except:
                    continue
            #tick world once ~0.1s
            self.world.tick()
            file.close()
            time.sleep(0.01)
            #iterate counting variable
            t+=1
        #Remove all vehicles from sim
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            vehicle.destroy()
        #Stop and Destroy sensors
        self.semantic_lidar.stop()
        self.semantic_lidar.destroy()
        self.camera.stop()
        self.camera.destroy()
        #Return Carla to default settings after sensor grab
        self.world.apply_settings(original_settings)
        print("Done Gathering Data")

    
    
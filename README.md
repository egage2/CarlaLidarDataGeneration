# CarlaLidarDataGeneration
Repo for running carla sim, and generating training data for Lidar perception models. 

## CarlaScenarioRunner()
Main class file for generating data from Carla Simulator
<br>SEE RunCarlaScenarioRunner.py for example

### Initialize with Constructor:

newClass = CarlaScenarioRunner(data_path = "file/path", intersection = "Intersection1")
<br>data_path: path to data output folder
<br>Valid Intersections: "Intersection1","Intersection2","Intersection3","Intersection4"

### Spawn Sensors and Entities with:
spawn_lidar(model)
<br>spawn_camera()
<br>spawn_vehicles(num_vehicles)
<br>spawn_pedestrian(num_pedestrians)

### Run simulation with:
gather_simulation_data(sim_seconds)

## ScenarioAnnotationVisualizer()
Visualization Class to display Lidar Pointclouds with drawn bounding boxes
<br>SEE End of file for run example

## drawBoundingPolygons.py
Additional code which takes in instance segmentized Camera images, draws bounding polygons around vehicles and pedestrians, saves all annotated images as single video file


## Synchronous_mode.ipynb
Jupter Notebook used during code development, contains useful methods for testing the framework with Carla

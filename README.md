# CarlaLidarDataGeneration
Repo for running carla sim, and generating training data for Lidar perception models. 

## CarlaScenarioRunner()
Main class file for generating data from Carla Simulator
SEE RunCarlaScenarioRunner.py for example

### Initialize with data_path, and Intersection args
Valid Intersection = "Intersection1","Intersection2","Intersection3","Intersection4"
spawn sensors and entities with:
spawn_lidar(model)
spawn_camera()
spawn_vehicles(num_vehicles)
spawn_pedestrian(num_pedestrians)

### Run simulation with:
gather_simulation_data(sim_seconds)

## ScenarioAnnotationVisualizer.py
Visualization Class to display Lidar Pointclouds with drawn bounding boxes
SEE End of file for run example

## drawBoundingPolygons.py
Additional code which takes in instance segmentized Camera images, draws bounding polygons around vehicles and pedestrians, saves all annotated images as single video file

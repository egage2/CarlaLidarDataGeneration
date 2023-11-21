# CarlaLidarDataGeneration
Repo for running carla sim, and generating training data for Lidar perception models. 

CarlaScenarioRunner()
Main class file for generating data from Carla Simulator

Initialize with data_path, and Intersection args
Valid Intersection = "Intersection1",
                     "Intersection2",
                     "Intersection3"
                     "Intersection4"
spawn sensors and entities with:
    spawn_lidar(model)
    spawn_camera()
    spawn_vehicles(num_vehicles)
    spawn_pedestrian(num_pedestrians)

Run simulation with:
    gather_simulation_data(sim_seconds)

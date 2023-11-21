"""
File to run Carla ScenarioRunner for Data Generation
"""

import os
import inspect
import argparse
import threading
from argparse import RawTextHelpFormatter

from CarlaScenarioRunner import CarlaScenarioRunner
import RunScenarioRunner
import RunManualControl

from scenario_runner.scenario_runner import ScenarioRunner
"""
-----------------------------------------------------------Setup CarlaScenarioRunner Class----------------------------------------------------------------------------
"""

carla_scenario = CarlaScenarioRunner(data_path = 'test_data/', intersection = 'Intersection1')

model = 'Ouster'
carla_scenario.spawn_lidar(model)
carla_scenario.spawn_camera()
carla_scenario.spawn_vehicles(4)
carla_scenario.gather_simulation_data(20)

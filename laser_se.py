import os
import json
import carla
import logging
import argparse

from langchain_openai import ChatOpenAI
from laser.agent_manager import AgentManager


args = None
client = None
carla_world = None
carla_map = None
agent_manager = None

lane_wps = None
driving_lane_num = None

def get_wp(x, y, z):
    global carla_map
    return carla_map.get_waypoint(carla.Location(x, y, z), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))

def init_world():
    global client, carla_world, carla_map, lane_wps, driving_lane_num

    client.reload_world()
    carla_world = client.get_world()
    if args.road == 'T04Highway':
        carla_world = client.load_world("Town04")
        carla_map = carla_world.get_map()
        # T04Highway
        lane_wps = [get_wp(4.671636, -68, 0), get_wp(8.171656, -68, 0), get_wp(11.671675, -68, 0), get_wp(15.171694, -68, 0)]
        driving_lane_num = 4
    elif args.road == 'T05Highway':
        carla_world = client.load_world("Town05")
        carla_map = carla_world.get_map()
        # T05Highway
        lane_wps = [get_wp(105.0, 194.0, 0), get_wp(105.0, 191.0, 0), get_wp(105.0, 188.0, 0)]
        driving_lane_num = 3
    elif args.road == 'T06Highway':
        carla_world = client.load_world("Town06")
        carla_map = carla_world.get_map()
        # T06Highway
        lane_wps = [get_wp(x=544.0, y=38.2, z=0), get_wp(x=544.0, y=41.7, z=0), get_wp(x=544.0, y=45.2, z=0), get_wp(x=544.0, y=48.7, z=0), get_wp(x=544.0, y=52.1, z=0)]
        driving_lane_num = 5
    elif args.road == 'T10Urban1':
        carla_world = client.load_world("Town10HD")
        carla_map = carla_world.get_map()
        # Town10Urban1
        lane_wps = [get_wp(106, 35, 0), get_wp(109, 35, 0), get_wp(117, 35, 0)] # bus stop at (109, 55.5)
        driving_lane_num = 2
        # bus_stop_pos = 20.5 # 150 - (20.5 + 3.5) = 126
    elif args.road == 'T10Urban2':
        carla_world = client.load_world("Town10HD")
        carla_map = carla_world.get_map()
        # Town10Urban2
        lane_wps = [get_wp(10.0, -64.7, 0), get_wp(10.0, -68.2, 0), get_wp(10.0, -75.6, 0)] # bus stop at (54.1, -68.2)
        driving_lane_num = 2
        # bus_stop_pos = 44.1 # 150 - (44.1 + 3.5) = 102.4
    elif args.road == 'T05Urban':
        carla_world = client.load_world("Town05")
        carla_map = carla_world.get_map()
        # T05Urban
        lane_wps = [get_wp(-66.0, -87.7, 0), get_wp(-66.0, -84.7, 0), get_wp(-66.0, -81.7, 0)] # bus stop at (-84.6, -84.7)
        driving_lane_num = 2
        # bus_stop_pos = 18.6 # 150 - (18.6 + 3.5) = 128



def init_agents():
    global client, carla_world, carla_map, lane_wps, driving_lane_num, agent_manager

    openai_llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

    script = json.load(open(args.script, 'r'))
    print("Load script: ")
    print(script)
    
    simulation_time = args.time
    agent_manager = AgentManager(client, carla_world, simulation_time, script, lane_wps, driving_lane_num, openai_llm)

def destroy():
    global agent_manager
    
    agent_manager.destroy(None)

if __name__ == "__main__":
    logging.basicConfig(filename='run.log', level=logging.DEBUG, filemode='w')

    parser = argparse.ArgumentParser(
        description='Script Execution')
    parser.add_argument("--host", 
                        default='127.0.0.1',
                        help="carla host ip (default: 127.0.0.1)")
    parser.add_argument("-p", "--port", 
                        default=2000,
                        type=int,
                        help="carla host port (default: 2000)")
    parser.add_argument("-r", "--road", 
                        type=str,
                        help="select road segment: T04Highway, T05Highway, T06Highway, T10Urban1, T10Urban2, T05Urban", required=True)
    parser.add_argument("-s", "--script", 
                        type=str,
                        help="path/to/script.json")
    parser.add_argument("-t", "--time", 
                        default=10,
                        type=int,
                        help="simulation time")

    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20)

    init_world()
    print('carla init finished')
    init_agents()
    print('agent init finished')

    try:
        agent_manager.run_scenario()
    except KeyboardInterrupt:
        print('\nCancelled')
    finally:
        destroy()
    

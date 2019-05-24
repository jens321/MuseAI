"""
Odie is trying to get the best present. Help him to learn what he should do.

Author: Moshe Lichman and Sameer Singh
"""
from __future__ import division
import numpy as np

import MalmoPython
import os
import random
import sys
import time
import json
import random
import math
import errno
from collections import defaultdict, deque
from timeit import default_timer as timer

def teleport(agent_host, teleport_x, teleport_z):
    '''
    Teleports the agent to a specific coordinate location after a delay

    **Parameters**
    --------------
    *agent_host*: malmo agent that will be acted on
    *teleport_x*: desired x coordinate to teleport to
    *teleport_z*: desired z coordinate to teleport to

    **Notes**
    ---------
    teleport_y is not needed because we will assume a desired
    teleport height of 57, which is ground level
    '''

    print("trying a teleport")

    tp_command = "tp " + str(teleport_x)+ " 57 " + str(teleport_z)
    agent_host.sendCommand(tp_command)
    good_frame = False
    start = timer()
    while not good_frame:
        world_state = agent_host.getWorldState()
        #print(world_state.number_of_video_frames_since_last_state)
        if not world_state.is_mission_running:
            print("Mission ended prematurely - error.")
            exit(1)
        if not good_frame and world_state.number_of_video_frames_since_last_state > 0:
            frame_x = world_state.video_frames[-1].xPos
            frame_z = world_state.video_frames[-1].zPos
            print("Current frame_x: {}".format(frame_x))
            print("Current frame_z: {}".format(frame_z))
            if math.fabs(frame_x - teleport_x) < 0.001 and math.fabs(frame_z - teleport_z) < 0.001:
                good_frame = True
                end_frame = timer()



def GetMissionXML(summary):
    ''' Build an XML mission string that uses the RewardForCollectingItem mission handler.'''

    result = '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>''' + summary + '''</Summary>
        </About>

        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>6000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
                <AllowSpawning>false</AllowSpawning>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;3*7,52*3,2;1;village" forceReset="true" /> 
                '''
    notes = ['F_sharp_3','G3','G_sharp_3','A3','A_sharp_3','B3','C4','C_sharp_4','D4','D_sharp_4','E4','F4','F_sharp_4', \
            'G4','G_sharp_4','A4','A_sharp_4','B4','C5','C_sharp_5','D5','D_sharp_5','E5','F5','F_sharp_5']

    num_notes = len(notes)
    print(num_notes)
    result += '''<DrawingDecorator>
                     '''
    for index, note in enumerate(notes):
        result += '''<DrawBlock
                       type="noteblock"
                       variant="{cur_note}"
                       x="-15"
                       y="56"
                       z="{z_loc}" />
                     <DrawBlock
                       type="wooden_pressure_plate"
                       x="-14"
                       y="56"
                       z="{z_loc}" />'''.format(cur_note=note, z_loc=num_notes-index)
                
    result += '''</DrawingDecorator>
                 '''
                 
    result += '''<ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>ThePianoMan</Name>
                <AgentStart>
                    <Placement x="0" y="56.0" z="{agent_loc}" yaw="90"/>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="480"/>
                  <AbsoluteMovementCommands/>
                  <MissionQuitCommands/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''.format(agent_loc=num_notes/2)

    return result 


if __name__ == '__main__':
    random.seed(0)
    #sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
    print('Starting...', flush=True)

    my_client_pool = MalmoPython.ClientPool()
    my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))

    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)


    
    num_reps = 30000
    n=1
    print("n=",n)
    for iRepeat in range(num_reps):
        my_mission = MalmoPython.MissionSpec(GetMissionXML("The Piano Man plays piano"), True)
        my_mission_record = MalmoPython.MissionRecordSpec()  # Records nothing by default
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(0)
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Attempt to start the mission:
                agent_host.startMission(my_mission, my_client_pool, my_mission_record, 0, "ThePianoMan")
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission", e)
                    print("Is the game running?")
                    exit(1)
                else:
                    time.sleep(2)

        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()

        while world_state.is_mission_running:
            print(".", end="")
            time.sleep(0.1)
            teleport(agent_host, -13, 1)
            time.sleep(5)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)


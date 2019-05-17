from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Modification of tutorial 2 to generate xml string for piano generation
# and teleporting the agent onto the appropriate tiles (tp not ready yet)

from builtins import range
import MalmoPython
import os
import sys
import time
import math
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

    To teleport to a specific note, from the list:
    notes = ['F_sharp_3','G3','G_sharp_3','A3','A_sharp_3','B3','C4','C_sharp_4','D4','D_sharp_4','E4','F4','F_sharp_4', \
            'G4','G_sharp_4','A4','A_sharp_4','B4','C5','C_sharp_5','D5','D_sharp_5','E5','F5','F_sharp_5']

    To teleport to a note at index i, call 
        teleport(agent_host, -13, i)
    where i is the index of the note + 1.5
    '''

    print("\ntrying a teleport")

    tp_command = "tp " + str(teleport_x) + " 57 " + str(teleport_z)
    agent_host.sendCommand(tp_command)
    good_frame = False
    start = timer()
    while not good_frame:
        world_state = agent_host.getWorldState()
        if not world_state.is_mission_running:
            print("Mission ended prematurely - error.")
            exit(1)
        if not good_frame and world_state.number_of_video_frames_since_last_state > 0:
            frame_x = world_state.video_frames[-1].xPos
            frame_z = world_state.video_frames[-1].zPos
            # print("Current frame_x: {}".format(frame_x))
            # print("Current frame_z: {}".format(frame_z))
            if math.fabs(frame_x - teleport_x) < 0.001 and math.fabs(frame_z - teleport_z) < 0.001:
                good_frame = True
                end_frame = timer()


def genString():
    '''
    Generates an appropriate xml string for creating the piano in minecraft

    **Notes**
    ---------
    Given the possible pitches for a Minecraft note block, places a tuned
    note block and pressure plate to allow for easy playing of that note
    '''

    result = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

              <About>
                <Summary>Hello world!</Summary>
              </About>

              <ServerSection>
                <ServerInitialConditions>
                    <Time>
                    <StartTime>12000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                </ServerInitialConditions>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;3*7,52*3,2;1;village" forceReset="true"/>
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
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                  <AbsoluteMovementCommands/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''.format(agent_loc=num_notes/2)
    return result


if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

# More interesting generator string: "3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"

missionXML=genString()

# Create default Malmo objects:

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()
my_mission_video = my_mission.requestVideo(800,500)

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission running ", end=' ')

#agent_host.sendCommand("tp -14 57 0")

# Loop until mission ends:
# teleporting to all of the notes, test run, to be changed
i=1.5
while world_state.is_mission_running:
    print(".", end="")
    time.sleep(0.5)
    teleport(agent_host, -13, i)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)
    i += 1

print()
print("Mission ended")
# Mission has ended.

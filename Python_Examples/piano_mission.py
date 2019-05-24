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
from sklearn.ensemble import RandomForestClassifier
from music21 import *
from Instrument import Instrument

from music_generator import MusicGenerator

NOTE_TO_POS_MAP = {
    'F#3': 0.5,
    'G3': 1.5,
    'G#3': 2.5,
    'A3': 3.5,
    'A#3': 4.5,
    'B3': 5.5,
    'C4': 6.5,
    'C#4': 7.5,
    'D4': 8.5,
    'D#4': 9.5,
    'E4': 10.5,
    'E#4': 11.5, # same as F4!!
    'F4': 11.5,
    'F#4': 12.5,
    'G4': 13.5,
    'G#4': 14.5,
    'A4': 15.5,
    'B4': 16.5,
    'C5': 17.5,
    'C#5': 18.5,
    'D5': 19.5,
    'D#5': 20.5,
    'E5': 21.5,
    'F5': 22.5,
    'F#5': 23.5
}

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
        teleport(agent_host, 14, i)
    where i is the index of the note + 0.5
    '''

    print("\ntrying a teleport")

    #agent_host.sendCommand("tp -14 57 0")
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

def genString(piano):
    '''
    Generates an appropriate xml string for creating the piano in minecraft

    **Notes**
    ---------
    Given the possible pitches for a Minecraft note block, places a tuned
    note block and pressure plate to allow for easy playing of that note
    '''

    notes = piano.getNotes()
    num_notes = piano.getNoteCount()

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

                            <DrawingDecorator>'''+  piano.genNoteBlocks() + '''</DrawingDecorator>

                            <ServerQuitFromTimeUp timeLimitMs="300000"/>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>

                    <AgentSection mode="Survival">
                        <Name>ThePianoMan</Name>
                        <AgentStart>
                            <Placement x="0" y="56.0" z="{agent_loc}" yaw="270"/>
                        </AgentStart>
                        <AgentHandlers>
                            <ObservationFromFullStats/>
                            <ContinuousMovementCommands turnSpeedDegs="180"/>
                            <AbsoluteMovementCommands/>
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''.format(agent_loc=num_notes/2)
    return result

def startTheMission(agent_host, my_mission, my_mission_record, max_retries):
    '''
    Attempt to start mission in indicated number of tries (max_retries)
    If attemps fail, exit.
    '''
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

def waitingForMissionStart(world_state, agent_host):
    '''
    Loop until mission begins. Checks world state in each iteration until mission begins.
    Returns world_state once the mission began.
    '''
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    return world_state

def runMission(world_state, agent_host, piano):
    '''
    Loops until mission ends.

    # Loop until mission ends:
    # teleporting to all of the notes, test run, to be changed
    '''
    training_music = ['bach/bwv66.6',
                        'bach/bwv1.6',
                        'bwv438',
                        'bwv44.7',
                        'bwv436',
                        'bwv89.6',
                        'bwv84.5',
                        'bwv83.5']
    # Should probably just stay one?
    test_music = ['bach/bwv437']
    music_gen = MusicGenerator(training_music, test_music)
    predicted = music_gen.generate_music()
    print(predicted)
    i=0.5
    note_idx = 0
    while world_state.is_mission_running:
        print(".", end="")
        time.sleep(0.3)
        while (predicted[note_idx] not in NOTE_TO_POS_MAP):
            note_idx = (note_idx + 1) % len(predicted)
        teleport(agent_host, 14, NOTE_TO_POS_MAP[predicted[note_idx]])
        note_idx = (note_idx + 1) % len(predicted)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
        i += 1
    return world_state

def main():
    '''
    Generats world along with note blocks.

    **Parameters**
    --------------

    **Notes**
    '''
    # More interesting generator string: "3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"
    piano = Instrument()
    missionXML = genString(piano)

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

    # 1 - Attempt to start a mission:
    max_retries = 3
    startTheMission(agent_host, my_mission, my_mission_record, max_retries)

    # 2 - Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    world_state = waitingForMissionStart(world_state, agent_host)
    print("\nMission running ", end=' ')

    # 3 - Loop until mission ends:
    world_state = runMission(world_state, agent_host, piano)
    print("\nMission ended")


if __name__ == '__main__':
    if sys.version_info[0] == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
    else:
        import functools
        print = functools.partial(print, flush=True)

    main()

from collections import defaultdict

class Instrument:
    pitches = ['F_sharp_3',
               'G3',
               'G_sharp_3',
               'A3',
               'A_sharp_3',
               'B3',
               'C4',
               'C_sharp_4',
               'D4',
               'D_sharp_4',
               'E4',
               'F4',
               'F_sharp_4',
               'G4',
               'G_sharp_4',
               'A4',
               'A_sharp_4',
               'B4',
               'C5',
               'C_sharp_5',
               'D5',
               'D_sharp_5',
               'E5',
               'F5',
               'F_sharp_5']
    noteLoc = defaultdict() # will be generated once genNoteBlocks called

    def genNoteBlocks(self):
        '''
        Generates xml for note blocks and
        save location of pressure plate for the note block in noteLoc
        '''
        xml = ""
        notes = self.getNotes()
        num_notes = self.getNoteCount()

        for index, note in enumerate(notes):
            xml += '''<DrawBlock type="noteblock"
                                 variant="{cur_note}"
                                 x="15"
                                 y="56"
                                 z="{z_loc}" />
                      <DrawBlock type="wooden_pressure_plate"
                                 x="14"
                                 y="56"
                                 z="{z_loc}" />'''.format(cur_note=note, z_loc=num_notes-index)
            self.noteLoc[note] = (13, index+.5)
        print(self.noteLoc)
        return xml

    def getNotes(self):
        '''
        Return list of all the pitches
        '''
        return self.pitches

    def getNoteCount(self):
        ''' Return count of the pitches '''
        return len(self.pitches)

    def getNotePos(self, note):
        ''' Return the location of the pressure plate for the note block'''
        return self.noteLoc[note]

    def runScale(self):
        '''
        Returns list of note locations for teleportation, playing all the notes sequentially
        Ex. [(13,1.5), (13,2.5)]
        '''
        command = list()
        for note in self.pitches:
            x_loc, z_loc = self.noteLoc[note]
            #tp_command = "tp " + str(x_loc) + " 57 " + str(z_loc)
            #command.append(tp_command)
            command.append(self.noteLoc[note])
        return command

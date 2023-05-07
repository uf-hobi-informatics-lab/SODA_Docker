###
 # <p>Title:  </p>
 # <p>Create Date: 19:44:14 01/10/22</p>
 # <p>Copyright: College of Medicine </p>
 # <p>Organization: University of Florida</p>
 # @author Yonghui Wu
 # @version 1.0
 # <p>Description: </p>
 ##

class TemporalExpression:
    def __init__(self):
        # context in [self.start, self.end) in self.row 
        self.text = ''
        # starts from 0
        # character index of the start and end
        self.start = -1
        self.end = -1
        # word index of the start and the end
        self.word_start = -1
        self.word_end = -1
        self.value = 'other'
        self.type = 'DATE'
        self.mod = 'NA'

    # 0 represents using character index; 1 represents using word index
    def tostring(self):
        te_str = 'TIMEX3="' 
        te_str += self.text + '" '

        te_str += str(self.start) + ':' + str(self.end)

        te_str += '||' + 'type="' + self.type + '"'
        te_str += '||' + 'val="' + self.value + '"'
        te_str += '||' + 'mod="' + self.mod + '"'
        return te_str



if __name__ == "__main__":
    import sys
 

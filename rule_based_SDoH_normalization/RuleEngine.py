###
 # <p>Title:  </p>
 # <p>Create Date: 15:14:06 01/10/22</p>
 # <p>Copyright: College of Medicine </p>
 # <p>Organization: University of Florida</p>
 # @author Yonghui Wu
 # @version 1.0
 # <p>Description: </p>
 ##
import os
import sys
import re
from pathlib import Path
from .TemporalExpression import *

class RuleEngine(object):
    '''
    The rule engine
    '''

    def __init__(self):
        print ("-----init RuleEngine")
        self._seperators=[' ', '\t','\n'] #determine the end of words
        self._patterns={} # dictionary patterns
        self._normPatterns={} # normalization patterns
        self._rules={} # extraction rules
        
        self._loadPatterns()
        # self._loadNormPatterns()
        self._loadRules()

        ###
        self._ruleExpression=[]
        self._ruleAttribute=[]



    def _loadPatterns(self, pattern_dir = './patterns'):
        print ("\n-----start _loadPatterns")
        tmpdir = pattern_dir
        if not os.path.isdir(tmpdir):
            tmpdir = str(Path(os.path.dirname(os.path.abspath(__file__))) / Path(tmpdir))            
        if not tmpdir.endswith('/'):
            tmpdir += '/'
        for fi in os.listdir(tmpdir):
            fi = tmpdir + fi
            if os.path.isdir(fi):
                self._loadPatterns(fi)
            else:
                tmp_pattern = []
                cur_pattern = ''
                for line in open(fi):
                    line = line.strip()
                    if line == '' or line.startswith('#') or line.startswith('//'):
                        continue
                    if line.endswith(':'):
                        if cur_pattern != '':
                            self._patterns[cur_pattern] = '|'.join(tmp_pattern)
                        tmp_pattern = []
                        cur_pattern = line[:-1].strip()
                    else:
                        tmp_pattern.append(line)
                if (cur_pattern != '') and tmp_pattern:
                    self._patterns[cur_pattern] = '|'.join(tmp_pattern)

        print ("----- loaded : "+str(len(self._patterns))+" patterns")

    def _loadNormPatterns(self, norm_dir = './normpatterns'):
        print ("\n----- start _loadNormPatterns")
        tmpdir = norm_dir
        if not os.path.isdir(tmpdir):
            tmpdir = str(Path(os.path.dirname(os.path.abspath(__file__))) / Path(tmpdir))    
        if not tmpdir.endswith('/'):
            tmpdir += '/'
        for fi in os.listdir(tmpdir):
            fi = tmpdir + fi
            if os.path.isdir(fi):
                self._loadNormPatterns(fi)
            else:
                cur_pattern = ''
                for line in open(fi): 
                    line = line.strip()
                    if line == '' or line.startswith('#') or line.startswith('//'):
                        continue
                    if line.endswith(':'):
                        cur_pattern = line[:-1].strip()
                        if cur_pattern != '':
                            self._normPatterns[cur_pattern] =  {}
                    else:
                        if cur_pattern == '':
                            print ('none norm pattern')
                            return
                        #token = line.split(None)
                        token = line.split('=>')
                        self._normPatterns[cur_pattern][token[0]] = token[1]

        print ("----- load: %s normPatterns",str(len(self._normPatterns)))

    def _loadRules(self, rule_dir="./rules"):#, ruleset):
        print ("\n----- _loadRules")
        tmpdir = rule_dir
        if not os.path.isdir(tmpdir):
            tmpdir = str(Path(os.path.dirname(os.path.abspath(__file__))) / Path(tmpdir))    
        if not tmpdir.endswith('/'):
            tmpdir += '/'
        for fi in os.listdir(tmpdir):
            fi = tmpdir + fi
            if os.path.isdir(fi):
                self._loadRules(fi)
            else:
                cur_type = ''
                for line in open(fi):
                    line = line.strip()
                    if line == '' or line.startswith('#') or line.startswith('//'):
                        continue
                    if line.endswith(':'):
                        cur_type = line[:-1].strip()
                    else:
                        if cur_type == '':
                            print ('load Rules error: no type')
                            return
                        else:
                            if cur_type not in self._rules:
                                self._rules[cur_type]=[]
                            token = line.split('",')
                            tlt=[None,None,None]
                            for item in token:
                                (attr, val) = item.split('="')
                                val = val.strip('"')
                                if attr == 'expression':
                                    # parse patterns in the expression
                                    tlt[0]=re.sub(r'%(\w+)', lambda m: '(' + self._patterns[m.group(1)] + ')', val)
                                elif attr == 'val':
                                    tlt[1]=val
                                elif attr == 'mod':
                                    tlt[2]=val
                                else:
                                    print ("----- _loadRules error, unknown type : %s",attr)
                                    return
                                            
                            self._rules[cur_type].append(tlt)

        for key,val in self._rules.items():
            print("----- loded %s rules for %s" % (str(len(val)),key))


                    
    def extract(self, input_str = '', ruleset = ''):
        if input_str == '':
            print ('no input string!')
            return None
        te = []
        ct=0
        vals = self._rules.get(ruleset.upper())
        #print(ruleset.upper())
        #print(self._rules.keys())
        
        for val in vals:
            #print(val, input_str)
            ct=ct+1
            #print  (" rule %s : %s"% (ct,val) )
            
            for m in re.finditer(val[0], input_str):
                tmp_te = TemporalExpression()
                tmp_te.start = m.start()
                tmp_te.end = m.end()
                tmp_te.text = input_str[tmp_te.start:tmp_te.end]
                tmp_te.word_start = len(re.findall(r'\s+', input_str[0: tmp_te.start]))
                tmp_te.word_end = tmp_te.word_start + len(re.findall(r'\s+', tmp_te.text)) + 1
                #print(tmp_te.text)
                if val[1] is not None:
                    # parse 'group' information
                    tmp_te.value = re.sub(r'group\((\d+)\)', lambda n:m.group(int(n.group(1))), val[1])
                    # parse normal patterns in value
                    tmp_te.value = re.sub(r'%(\w+)\((.*?)\)', lambda n: self._normPatterns[n.group(1)][n.group(2).lower()], tmp_te.value)
                if val[2] is not None:
                    # parse 'group' information
                    tmp_te.mod = re.sub(r'group\((\d+)\)', lambda n:m.group(int(n.group(1))), val[2])
                    # parse mod patterns in mod
                    #tmp_te.mod = re.sub(r'%(\w+)\((\w+[\w+-]+\w+|\w+\.?)\)', lambda n: Rule.norm_pattern[n.group(1)][n.group(2).lower()], tmp_te.mod)
                    tmp_te.mod = re.sub(r'%(\w+)\((.*?)\)', lambda n: self._normPatterns[n.group(1)][n.group(2).lower()], tmp_te.mod)
                
                te.append(tmp_te)
                

                if m is None:
                    empty_tmp_te = TemporalExpression()
                    te.append(empty_tmp_te)
        
        return te


if __name__ == "__main__":
    import sys 

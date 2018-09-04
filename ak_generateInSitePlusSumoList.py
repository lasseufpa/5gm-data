'''
Generate a CSV with information from position and also LOS. The basic information
about position is obtained with
python ak_generateInfoList.py > new_simuls_los.csv
which is later complemented wit LOS infomartion with this script.
Important: edit the file new_simuls_los.csv to delete first and last rows
before running this script with
'''
import csv
import os

#from config.py
def base_run_dir_fn(i): #the folders will be run00001, run00002, etc.
    """returns the `run_dir` for run `i`"""
    return "run{:05d}".format(i)

#folder with runs
basePath = 'D:/insitedata/noOverlappingTx5m'
#file generated with ak_generateInfoList.py:
insiteCSVFile = 'D:/github/5gm-data/new_simuls_los.csv'

'''
create dictionary taking the episode, scene and Rx number of file with rows e.g.:
V,0,0,2,flow2.0,753.8198286576,507.38595866735,LOS=1
V,0,0,3,flow2.1,749.7071175056,566.1905128583,LOS=1
I,0,0,4,flow2.2,729.85254058595,670.38607208065,none
'''
with open(insiteCSVFile, 'r') as f:
    insiteReader = csv.reader(f)
    insiteDictionary = {}
    for row in insiteReader:
        #print(row)
        thisKey = str(row[1])+','+str(row[2])+','+str(row[3])
        insiteDictionary[thisKey]=row
#print(insiteDictionary['0,0,0'][4])

for numRun in range(1372):
    thisRunFolder = os.path.join(basePath, base_run_dir_fn(numRun))
    sumoInfoTxtFileName = os.path.join(thisRunFolder, 'sumoOutputInfoFileName.txt')
    #print('Opening', sumoInfoTxtFileName)
    with open(sumoInfoTxtFileName, 'r') as f:
        reader = csv.reader(f)
        allLines = list(reader)
    numLines = len(allLines)
    #print('numLines ', numLines)
    #print(allLines)
    for lineNum in range(1, numLines): #recall that we have a header, so start in 1 instead of 0
        #"episode_i,scene_i,receiverIndex,veh,veh_i,typeID,xinsite,yinsite,x3,y3,z3,lane_id,angle,speed,length, width, height,distance,waitTime"
        thisLine = allLines[lineNum]
        #some lines are []
        if thisLine == []: #this is a bug in reading the text file? CR / LF issue?
            continue
        receiverIndex = int(thisLine[2])
        if receiverIndex == -1:
            continue #skip this line, it's not a receiver
        #now we have a receiver. Let's get information from other line via dictionary
        #print(thisLine)
        thisKey = str(thisLine[0])+','+str(thisLine[1])+','+str(thisLine[2])
        thisInSiteLine = insiteDictionary[thisKey] #recover from dic
        #print(thisInSiteLine)
        isValid = thisInSiteLine[0] #V or I are the first element of the list thisLine
        if isValid == 'I':
            continue #discard invalid vehicles
        #Recall from rwisimulation.placement.py that
        #SUMO reports position of middle of front bumper. We repositioned to the middle of the vehicle
        #So, use coordinates from "InSite" file
        #pick what to include in new list:
        #    0          1          2       3    4      5      6      7      8 9  10    11    12     13    14     15      16     17       18
        #"episode_i,scene_i,receiverIndex,veh,veh_i,typeID,xinsite,yinsite,x3,y3,z3,lane_id,angle,speed,length, width, height,distance,waitTime"
        thisString = thisLine[0] + ',' + thisLine[1] + ',' + thisLine[2] + ',' + thisLine[3] + ',' + thisLine[5] + ',' \
                     + thisInSiteLine[5] + ',' + thisInSiteLine[6] + \
                     ',' + thisLine[16] + ',' + thisRunFolder.replace('\\','/') + ','+ thisInSiteLine[-1]
        print(thisString)

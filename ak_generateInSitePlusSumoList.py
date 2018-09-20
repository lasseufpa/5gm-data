'''
Generate a CSV with information from position and also LOS. The basic information
about position is obtained previously running
python ak_generateInfoList.py > new_simuls_los.csv
and this information now complemented with information obtained, within each
folder, from files sumoOutputInfoFileName.txt.
Important: edit the file new_simuls_los.csv to delete first and last rows
before running this script.
'''
import csv
import os
from sys import argv

#from config.py
def base_run_dir_fn(i): #the folders will be run00001, run00002, etc.
    """returns the `run_dir` for run `i`"""
    return "run{:05d}".format(i)

#folder with runs
#basePath = 'D:/insitedata/results_new_lidar'
#file previously generated with script ak_generateInfoList.py:
insiteCSVFile = 'D:/github/5gm-data/list1_valids_and_invalids.csv'
'''
the list above has information:
V,0,0,2,flow2.0,753.8198286576,507.38595866735,25,LOS=1
V,0,0,3,flow2.1,749.7071175056,566.1905128583,14,LOS=1
I,0,0,4,flow2.2,729.85254058595,670.38607208065,0,none
'''

if len(argv) != 2:
    print('You need to specify the folder that has the output files written by the ray-tracing simulator!')
    print('Usage: python', argv[0], 'input_folder')
    print('PS: Recall that this script assumes you also use the file:',insiteCSVFile,'If this is not the correct one, please edit the script!')
    exit(-1)
# Object which will be modified in the RWI project
#base_insite_project_path = 'D:/insitedata/insite_new_simuls/'
#Folder to store each InSite project and its results (will create subfolders for each "run", run0000, run0001, etc.)
basePath = argv[1] #'D:/owncloud-lasse/5GM_DATA/flat_simulation/results_new_lidar/'


with open(insiteCSVFile, 'r') as f:
    insiteReader = csv.reader(f)
    insiteDictionary = {}
    for row in insiteReader:
        #print(row)
        try:
            thisKey = str(row[1])+','+str(row[2])+','+str(row[3])
        except IndexError as exc:
            print('WARNING:\nHave you deleted the first and last lines of file',insiteCSVFile,'?\n')
            raise exc
        insiteDictionary[thisKey]=row
#print(insiteDictionary['0,0,0'][4])

numRun = 0 #counter
while True:
    thisRunFolder = os.path.join(basePath, base_run_dir_fn(numRun))
    sumoInfoTxtFileName = os.path.join(thisRunFolder, 'sumoOutputInfoFileName.txt')
    #print('Opening', sumoInfoTxtFileName)
    try:
        with open(sumoInfoTxtFileName, 'r') as f:
            reader = csv.reader(f)
            allLines = list(reader)
    except FileNotFoundError:
        break
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
        #    0          1          2       3    4      5      6      7      8 9  10    11    12     13    14     15      16     17       18       19
        #"episode_i,scene_i,receiverIndex,veh,veh_i,typeID,xinsite,yinsite,x3,y3,z3,lane_id,angle,speed,length, width, height,distance,waitTime" numRays
        thisString = thisLine[0] + ',' + thisLine[1] + ',' + thisLine[2] + ',' + thisLine[3] + ',' + thisLine[5] + ',' \
                     + thisInSiteLine[5] + ',' + thisInSiteLine[6] + \
                     ',' + thisLine[16] + ',' + thisInSiteLine[-2] + ',' + thisRunFolder.replace('\\','/') + ','+ thisInSiteLine[-1]
        print(thisString)

    numRun += 1

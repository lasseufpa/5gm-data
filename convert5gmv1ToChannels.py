#Will parse all database and create numpy arrays that represent all channels in the database.
#Specificities: some episodes do not have all scenes. And some scenes do not have all receivers.
#Assuming Ne episodes, with Ns scenes each, and Nr receivers (given there was only one transmitter),
#there are Ne x Ns x Nr channel matrices and each must represent L=25 rays.
#With Ne=119, Ns=50, Nr=10, we have 59500 matrices with 25 rays. It is better to save
#each episode in one file, with the matrix given by
#scene 1:Ns x Tx_index x Rx_index x numberRays and 7 numbers, the following for each ray
        # path_gain
        # timeOfArrival
        # departure_elevation
        # departure_azimuth
        # arrival_elevation
        # arrival_azimuth
        # isLOS
#to simplify I will assume that all episodes have 50 scenes and 10 receivers.
import datetime
import numpy as np
from shapely import geometry
#from matplotlib import pyplot as plt
import h5py

from rwisimulation.positionmatrix import position_matrix_per_object_shape, calc_position_matrix
#from rwisimulation.calcrxpower import calc_rx_power

from rwisimulation.datamodel import save5gmdata as fgdb

#import config as c
class c:
    #analysis_area = (648, 348, 850, 685)
    analysis_area = (744, 429, 767, 679)
    analysis_area_resolution = 0.5
    antenna_number = 4
    frequency = 6e10
analysis_polygon = geometry.Polygon([(c.analysis_area[0], c.analysis_area[1]),
                                     (c.analysis_area[2], c.analysis_area[1]),
                                     (c.analysis_area[2], c.analysis_area[3]),
                                     (c.analysis_area[0], c.analysis_area[3])])
session = fgdb.Session()
totalNumEpisodes = session.query(fgdb.Episode).count()

#pm_per_object_shape = position_matrix_per_object_shape(c.analysis_area, c.analysis_area_resolution)
#print(pm_per_object_shape)

# just to report time
start = datetime.datetime.today()
perc_done = None

#if needed, manually create the output folder
fileNamePrefix = './insitedata/urban_canyon_v2i_5gmv1_rays' #prefix of output files
pythonExtension = '.npz'
matlabExtension = '.hdf5'

# assume 50 scenes per episode, 10 receivers per scene
numScenesPerEpisode = 1 #50
numTxRxPairsPerScene = 10
numRaysPerTxRxPair = 25
numVariablePerRay = 7+1 #has the ray angle now
#plt.ion()
numEpisode = 0
numLOS = 0
numNLOS = 0
for ep in session.query(fgdb.Episode): #go over all episodes
    print('Processing ', ep.number_of_scenes, ' scenes in episode ', ep.insite_pah,)
    print('Start time = ', ep.simulation_time_begin, ' and sampling period = ', ep.sampling_time, ' seconds')
    print('Episode: ' + str(numEpisode+1) + ' out of ' + str(totalNumEpisodes))

    #initialization
    #Ns x [Tx_index x Rx_index x numberRays] and 7 numbers, the following for each ray
    allEpisodeData = np.zeros((numScenesPerEpisode, numTxRxPairsPerScene, numRaysPerTxRxPair,
                               numVariablePerRay), np.float32)
    allEpisodeData.fill(np.nan)
    
    #from the first scene, get all receiver names
    rec_name_to_array_idx_map = [obj.name for obj in ep.scenes[0].objects if len(obj.receivers) > 0]
    print(rec_name_to_array_idx_map)
    
    #process each scene in this episode
    #count # of ep.scenes
    for sc_i, sc in enumerate(ep.scenes):
        #print('Processing scene # ', sc_i)
        polygon_list = []
        polygon_z = []
        polygons_of_interest_idx_list = []
        rec_present = []

        for obj in sc.objects:
            if len(obj.receivers) == 0:
                continue  #do not process objects that are not receivers
            obj_polygon = geometry.asMultiPoint(obj.vertice_array[:,(0,1)]).convex_hull
            # check if object is inside the analysis_area
            if obj_polygon.within(analysis_polygon):
                # if the object is a receiver and is within the analysis area
                if len(obj.receivers) > 0:
                    rec_array_idx = rec_name_to_array_idx_map.index(obj.name)
                    for rec in obj.receivers: #for all receivers
                        ray_i = 0
                        isLOSChannel = 0
                        for ray in rec.rays: #for all rays
                            #gather all info
                            thisRayInfo = np.zeros(8)
                            thisRayInfo[0] = ray.path_gain
                            thisRayInfo[1] = ray.time_of_arrival
                            thisRayInfo[2] = ray.departure_elevation
                            thisRayInfo[3] = ray.departure_azimuth
                            thisRayInfo[4] = ray.arrival_elevation
                            thisRayInfo[5] = ray.arrival_azimuth
                            thisRayInfo[6] = ray.is_los
                            thisRayInfo[7] = ray.phaseInDegrees
                            #allEpisodeData = np.zeros((numScenesPerEpisode, numTxRxPairsPerScene,
                            # numRaysPerTxRxPair, numVariablePerRay), np.float32)
                            allEpisodeData[sc_i][rec_array_idx][ray_i]=thisRayInfo
                            ray_i += 1
                            if ray.is_los == 1:
                                isLOSChannel = True #if one ray is LOS, the channel is
                        if isLOSChannel == True:
                            numLOS += 1
                        else:
                            numNLOS += 1
                        # just for reporting spent time
        perc_done = ((sc_i + 1) / ep.number_of_scenes) * 100
        elapsed_time = datetime.datetime.today() - start
        time_p_perc = elapsed_time / perc_done
        print('\r Done: {:.2f}% Scene: {} time per scene: {} time to finish: {}'.format(
            perc_done,
            sc_i + 1,
            elapsed_time / (sc_i + 1),
            time_p_perc * (100 - perc_done)), end='')

    print()
    outputFileName = fileNamePrefix + '_e' + str(numEpisode+1) + pythonExtension
    np.savez(outputFileName, allEpisodeData=allEpisodeData)
    print('==> Wrote file ' + outputFileName)

    outputFileName = fileNamePrefix + '_e' + str(numEpisode+1) + matlabExtension
    print('==> Wrote file ' + outputFileName)
    f = h5py.File(outputFileName, 'w')
    f['allEpisodeData'] = allEpisodeData
    f.close()

    numEpisode += 1 #increment episode counter

print('numLOS = ', numLOS)
print('numNLOS = ', numNLOS)
print('Sum = ', numLOS + numNLOS)
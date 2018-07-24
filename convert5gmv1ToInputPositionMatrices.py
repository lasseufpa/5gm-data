#Script to generate data for beam selection using only the position of vehicles. 
#The output npz and hdf5 file have the position_matrix_array to be the input of machine
# learning algorithms and informs the vehicles positions

#This code uses a simple sparse representation for the position matrices: only the
#non-zero elements are specified, by three integers indicating the row and column
#indices and the value (as an integer)

import datetime
import numpy as np
import os
from shapely import geometry
from PIL import Image  #used pip install pillow
#from matplotlib import pyplot as plt

from rwisimulation.positionmatrix import position_matrix_per_object_shape, calc_position_matrix, matrix_plot
#from rwisimulation.calcrxpower import calc_rx_power

from rwisimulation.datamodel import save5gmdata as fgdb
import h5py

#import config as c
class c: #this information is obtained from the config.py file used to generate the data
    #analysis_area = (648, 348, 850, 685)
    analysis_area = (744, 429, 767, 679) #coordinates that define the areas the mobile objects should be
    analysis_area_resolution = 0.5 #grid resolution in meters
    #antenna_number = 4 #number of antenna elements in Rx array (not used here)
    #frequency = 6e10 #carrier frequency in Hz (not used here)

analysis_polygon = geometry.Polygon([(c.analysis_area[0], c.analysis_area[1]),
                                     (c.analysis_area[2], c.analysis_area[1]),
                                     (c.analysis_area[2], c.analysis_area[3]),
                                     (c.analysis_area[0], c.analysis_area[3])])

#if needed, create the output folder
outputFolder = './positionMatrices'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
fileNamePrefix = os.path.join(outputFolder,'urban_canyon_v2i_5gmv1_positionMatrix') #prefix of output files
pythonExtension = '.npz'
matlabExtension = '.hdf5'

relevantDistanceFromNeighborCars = 15 #in meters. Use np.inf to disable this
numPixelsRelevantNeighborhood = relevantDistanceFromNeighborCars / c.analysis_area_resolution
receiverValue = 4 #value that represents the receiver of interest in a scene. It depends on analysis_area_resolution

# assume 50 scenes per episode
numScenesPerEpisode = 50
maxNumReceiverPerScene = 10

#this opens the database in a given file. See save5gmdata.py
session = fgdb.Session()

pm_per_object_shape = position_matrix_per_object_shape(c.analysis_area, c.analysis_area_resolution)
print('Will write output matrices of dimension: ', pm_per_object_shape)

# do not look, just to report
start = datetime.datetime.today()
perc_done = None

totalNumEpisodes = session.query(fgdb.Episode).count()
print('Found ', totalNumEpisodes, ' episodes. Processing...')
numEpisode = 0
for ep in session.query(fgdb.Episode):
    # 50 scenes, 10 receivers per scene
    print('Processing ', ep.number_of_scenes, ' scenes. According to the database, ' 
          ' the scenes in this episode were obtained in folder ', ep.insite_pah, ' and consecutive folders.')
    print('Start time = ', ep.simulation_time_begin, ' and sampling period = ', ep.sampling_time, ' seconds')
    #Assumes 50 scenes per episode and 10 Tx/Rx pairs per scene
    position_matrix_array = np.zeros((numScenesPerEpisode, maxNumReceiverPerScene, *pm_per_object_shape), np.int8)
    rec_name_to_array_idx_map = [obj.name for obj in ep.scenes[0].objects if len(obj.receivers) > 0]
    print(rec_name_to_array_idx_map)
    for sc_i, sc in enumerate(ep.scenes):
        polygon_list = []
        polygon_z = []
        polygons_of_interest_idx_list = []
        rec_present = []
        for obj in sc.objects:
            obj_polygon = geometry.asMultiPoint(obj.vertice_array[:,(0,1)]).convex_hull
            # check if object is inside the analysis_area
            if obj_polygon.within(analysis_polygon):
                # if the object is a receiver calc a position_matrix for it
                if len(obj.receivers) > 0:
                    rec_array_idx = rec_name_to_array_idx_map.index(obj.name)
                    for rec in obj.receivers:
                        best_ray = None
                        best_path_gain = - np.inf
                        for ray in rec.rays:
                            if ray.path_gain > best_path_gain:
                                best_path_gain = ray.path_gain
                                best_ray = ray
                    if (best_ray is not None):
                        # the next polygon added will be the receiver
                        polygons_of_interest_idx_list.append(len(polygon_list))
                        rec_present.append(obj.name)
                polygon_list.append(obj_polygon)
                polygon_z.append(-obj.dimension[2])
        if len(polygons_of_interest_idx_list) != 0:
            scene_position_matrix = calc_position_matrix(
                c.analysis_area,
                polygon_list,
                c.analysis_area_resolution,
                polygons_of_interest_idx_list,
                polygon_z=polygon_z,
            )
            #matrix_plot(scene_position_matrix[1])
            #input('Enter')

        for rec_i, rec_name in enumerate(rec_present):
            rec_array_idx = rec_name_to_array_idx_map.index(rec_name)
            if relevantDistanceFromNeighborCars == np.inf:
                #copy the whole matrix, all vehicles are included
                position_matrix_array[sc_i, rec_array_idx, :] = scene_position_matrix[rec_i]
            else:
                #overwrite if 0's the vehicles that are too far apart from receiver of interest
                thisReceiverIndices=np.argwhere(scene_position_matrix[rec_i] == receiverValue)
                if thisReceiverIndices.size == 0:
                    print('############ Error')
                    position_matrix_array[sc_i, rec_array_idx, :] = scene_position_matrix[rec_i]
                    continue
                thisReceiverIndices=thisReceiverIndices[:,1] #take only numbers along length dimension
                minCoordinate = np.min(thisReceiverIndices,0) - numPixelsRelevantNeighborhood
                if minCoordinate < 0:
                    minCoordinate = 0
                maxCoordinate = np.max(thisReceiverIndices,0) + numPixelsRelevantNeighborhood
                if maxCoordinate > pm_per_object_shape[1]-1:
                    maxCoordinate = pm_per_object_shape[1]-1
                #take out vehicles "above" receiver
                scene_position_matrix[rec_i,:,0:np.int(minCoordinate)]=0
                #take out vehicles "below" receiver
                scene_position_matrix[rec_i,:,np.int(maxCoordinate):pm_per_object_shape[1]-1]=0
                #store the modified matrix
                position_matrix_array[sc_i, rec_array_idx, :] = scene_position_matrix[rec_i]

        # just reporting spent time
        perc_done = ((sc_i + 1) / ep.number_of_scenes) * 100
        elapsed_time = datetime.datetime.today() - start
        time_p_perc = elapsed_time / perc_done
        print('\r Done: {:.2f}% Scene: {} time per scene: {} time to finish: {}'.format(
            perc_done,
            sc_i + 1,
            elapsed_time / (sc_i + 1),
            time_p_perc * (100 - perc_done)), end='')
    print()

    for i in range(position_matrix_array.shape[0]):
        for j in range(position_matrix_array.shape[1]):
            outputFileName = fileNamePrefix + '_e' + str(numEpisode+1) + \
                             '_s' + str(i+1) + '_r' + str(j+1) + '.png'
            im = Image.fromarray(position_matrix_array[i,j,:,:])
            im.save(outputFileName)
            print('==> Wrote file ' + outputFileName)

    if 0:
        outputFileName = fileNamePrefix + 'positions_e' + str(numEpisode+1) + pythonExtension
        np.savez(outputFileName, position_matrix_array=position_matrix_array)
        print('==> Wrote file ' + outputFileName)

        outputFileName = fileNamePrefix + 'positions_e' + str(numEpisode+1) + matlabExtension
        print('==> Wrote file ' + outputFileName)
        f = h5py.File(outputFileName, 'w')
        f['position_matrix_array'] = position_matrix_array
        f.close()

    numEpisode += 1

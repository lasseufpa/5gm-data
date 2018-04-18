#Script to generate data for beam selection using only the position of vehicles. 
#The output npz and hdf5 file have the position_matrix_array to be the input of machine
# learning algorithms and informs the vehicles positions

import datetime
import numpy as np
from shapely import geometry
#from matplotlib import pyplot as plt

from rwisimulation.positionmatrix import position_matrix_per_object_shape, calc_position_matrix
from rwisimulation.calcrxpower import calc_rx_power

from rwisimulation.datamodel import save5gmdata as fgdb
import h5py

#import config as c
class c: #this information is obtained from the config.py file used to generate the data
    #analysis_area = (648, 348, 850, 685)
    analysis_area = (744, 429, 767, 679) #coordinates that define the areas the mobile objects should be
    analysis_area_resolution = 0.5 #grid resolution in meters (not used here)
    #antenna_number = 4 #number of antenna elements in Rx array (not used here)
    #frequency = 6e10 #carrier frequency in Hz (not used here)

analysis_polygon = geometry.Polygon([(c.analysis_area[0], c.analysis_area[1]),
                                     (c.analysis_area[2], c.analysis_area[1]),
                                     (c.analysis_area[2], c.analysis_area[3]),
                                     (c.analysis_area[0], c.analysis_area[3])])

#if needed, manually create the output folder
fileNamePrefix = './positionMatrices/urban_canyon_v2i_5gmv1_matrices' #prefix of output files
pythonExtension = '.npz'
matlabExtension = '.hdf5'

# assume 50 scenes per episode
numScenesPerEpisode = 50

session = fgdb.Session()

pm_per_object_shape = position_matrix_per_object_shape(c.analysis_area, c.analysis_area_resolution)
#print(pm_per_object_shape)

# do not look, just to report
start = datetime.datetime.today()
perc_done = None

totalNumEpisodes = session.query(fgdb.Episode).count()
print('Found ', totalNumEpisodes, ' episodes. Processing...')
numEpisode = 0
for ep in session.query(fgdb.Episode):
    # 50 scenes, 10 receivers per scene
    print('Processing ', ep.number_of_scenes, ' scenes in episode ', ep.insite_pah,)
    print('Start time = ', ep.simulation_time_begin, ' and sampling period = ', ep.sampling_time, ' seconds')
    #Assumes 50 scenes per episode and 10 Tx/Rx pairs per scene
    position_matrix_array = np.zeros((50, 10, *pm_per_object_shape), np.int8)
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
        for rec_i, rec_name in enumerate(rec_present):
            rec_array_idx = rec_name_to_array_idx_map.index(rec_name)
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
    outputFileName = fileNamePrefix + 'positions_e' + str(numEpisode+1) + pythonExtension
    np.savez(outputFileName, position_matrix_array=position_matrix_array)
    print('==> Wrote file ' + outputFileName)

    outputFileName = fileNamePrefix + 'positions_e' + str(numEpisode+1) + matlabExtension
    print('==> Wrote file ' + outputFileName)
    f = h5py.File(outputFileName, 'w')
    f['position_matrix_array'] = position_matrix_array
    f.close()

    numEpisode += 1

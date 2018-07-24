# Will parse database to show how information is organized.
# Specificities: some episodes do not have all scenes. And some scenes do not have all receivers.
# The following information can be obtained for each ray
# path_gain
# timeOfArrival
# departure_elevation
# departure_azimuth
# arrival_elevation
# arrival_azimuth
# isLOS
# Look at save5gmdata.py to understand the information stored in the database.

import numpy as np
from shapely import geometry
# from matplotlib import pyplot as plt
import h5py

from rwisimulation.positionmatrix import position_matrix_per_object_shape, calc_position_matrix
# from rwisimulation.calcrxpower import calc_rx_power

from rwisimulation.datamodel import save5gmdata as fgdb


# import config as c  #instead of reading from the config.py file, copy and paste here the used variables:
class c:
    # analysis_area = (648, 348, 850, 685)
    analysis_area = (744, 429, 767, 679)  # coordinates that define the areas the mobile objects should be
    # analysis_area_resolution = 0.5 #grid resolution in meters (not used here)
    # antenna_number = 4 #number of antenna elements in Rx array (not used here)
    # frequency = 6e10 #carrier frequency in Hz (not used here)


# construct an object to help identifying objects within the analysis area
analysis_polygon = geometry.Polygon([(c.analysis_area[0], c.analysis_area[1]),
                                     (c.analysis_area[2], c.analysis_area[1]),
                                     (c.analysis_area[2], c.analysis_area[3]),
                                     (c.analysis_area[0], c.analysis_area[3])])
session = fgdb.Session()
totalNumEpisodes = session.query(fgdb.Episode).count()

numEpisode = 0
numValidChannels = 0
for ep in session.query(fgdb.Episode):  # go over all episodes
    print('Processing ', ep.number_of_scenes, ' scenes in episode ', ep.insite_pah, )
    print('The mentioned file corresponds to the first scene. To find the others, increment the counter.')
    print('For example some_path/run00005/ is the first scene, hence the second correspons to '
          ' some_path/run00006/ and so on')
    print('Start time = ', ep.simulation_time_begin, ' and sampling period = ', ep.sampling_time, ' seconds')
    print('Episode: ' + str(numEpisode + 1) + ' out of ' + str(totalNumEpisodes))

    # from the first scene, get all receiver names
    # this list is important because it allows to converts vehicle names to their indices. From a given episode,
    # a given vehicle name will always have the same index in this list
    rec_name_to_array_idx_map = [obj.name for obj in ep.scenes[0].objects if len(obj.receivers) > 0]
    print('The names of the mobile objects that have at a receiver in this episode are:')
    # See documentation at https://github.com/lasseufpa/5gm-rwi-simulation/wiki
    print(rec_name_to_array_idx_map)
    maxNumOfReceiversInThisEpisode = len(rec_name_to_array_idx_map)

    # process each scene in this episode
    # count # of ep.scenes
    for sc_i, sc in enumerate(ep.scenes):
        # print('Processing scene # ', sc_i, ' with ID ', sc.id, ' from episode ', sc.episode_id,
        #      ' with ', sc.number_of_receivers, ' receivers and ', sc.number_of_mobile_objects, ' mobile objects')
        polygon_list = []
        polygon_z = []
        polygons_of_interest_idx_list = []
        rec_present = []

        validReceiversInThisScene = np.zeros(maxNumOfReceiversInThisEpisode, dtype=np.bool)

        # sc.objects has sc.number_of_mobile_objects, which can be a large number (e.g. 53)
        for obj in sc.objects:  # get object in scene
            if len(obj.receivers) == 0:
                # print('Skipping ', obj.name, ' because it does not have a receiver!')
                continue
            # only check if within area in case it's a receiver to avoid wasting time
            obj_polygon = geometry.asMultiPoint(obj.vertice_array[:, (0, 1)]).convex_hull
            # check if object is inside the analysis_area
            if obj_polygon.within(analysis_polygon):
                # print(obj.name, ' with ID ', obj.id, ' is within analysis area and has a receiver')
                rec_array_idx = rec_name_to_array_idx_map.index(obj.name)
                validReceiversInThisScene[rec_array_idx] = True
                numValidChannels += 1
            else:
                # print(obj.name, ' with ID ', obj.id, ' is not in analysis area')
                continue

                # if the object is a receiver and is within the analysis area
                # in the 5gmv1 data this number is 0 or 1 (there is no more than 1 Rx per object)
                # print(obj.name, ' has ', len(obj.receivers), ' receiver(s)')
                # use the infamouse list to get the index corresponding to the mobile object with name obj.name
            if False:
                print(obj.name, ' is mapped to index ', rec_array_idx)
                for rec in obj.receivers:  # for all potential receivers
                    for ray in rec.rays:  # for all rays
                        # gather all info
                        thisRayInfo = np.zeros(7)
                        thisRayInfo[0] = ray.path_gain
                        thisRayInfo[1] = ray.time_of_arrival
                        thisRayInfo[2] = ray.departure_elevation
                        thisRayInfo[3] = ray.departure_azimuth
                        thisRayInfo[4] = ray.arrival_elevation
                        thisRayInfo[5] = ray.arrival_azimuth
                        thisRayInfo[6] = ray.is_los
                    print('Last ray of this receiver has values: ', thisRayInfo)
        print('Scene ', sc_i, ', valids: ', validReceiversInThisScene)
    numEpisode += 1  # increment episode counter
print('numValidChannels = ', numValidChannels)

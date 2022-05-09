'''
This code gets the p2m files generated by InSite via simulation.py and writes the ray information in files.
Different from todb.py, it does not use a database (no episode.db).
One does not need to specify the number of scenes per episode because this information is obtained from
the JSON file and confirmed (redundancy) with the file 'sumoOutputInfoFileName.txt' at "run_dir".

Wrote by Daniel Suzuki to support pre-processing to v2v data.

We get paired simulations with the very same number of channels. For both the DL and UL we
have X runs. There is a single p2m file in a DL simulations with the same positions of Tx and Rx of
the several (e.g. 10) corresponding files in the UL simulation.

InSite writes a p2m file for each transmitter. Hence, when there are several receivers for a given Tx - Rx pair, one
will obtain the number of the receiver from (inside) the p2m file. When there are several transmitters, in order to
identify the Tx number, one needs to look in a different way. One alternative is to obtain the Tx number from the
file name, given that the file names inform the Tx number.
'''
import os
import json
import numpy as np
import csv
from sys import argv
import h5py


# From https://coderwall.com/p/x6xtxq/convert-bytes-to-int-or-int-to-bytes-in-python
def bytes_to_int(bytes):
    result = 0
    for b in bytes:
        result = result * 256 + int(b)
    return result


def int_to_bytes(value, length):
    result = []
    for i in range(0, length):
        result.append(value >> (i * 8) & 0xff)
    result.reverse()
    return result


# now we don't need to recall config.py. We can simply specify folders below

from rwisimulation.tfrecord import SceneNotInEpisodeSequenceError
from rwiparsing import P2mPaths
from rwiparsing import P2mCir

if len(argv) != 3:
    print(
        'You need to specify the input folder (that has the output files written by the ray-tracing simulator) and output folder!')
    print('Usage: python', argv[0], 'input_folder output_folder')
    exit(-1)

numScenesPerEpisode = 50
# if more than one Tx, should multiply the number of Tx and RX per numTx
numRaysPerTxRxPair = 30
numParametersPerRay = 10
max_num_interactions = 30  # this will be checked in this script, so you can start by guessing and then re-run the script
numTx = 2
numRx = 5
numTxRxPairsPerScene = numTx * numRx
should_write_interactions = True
should_write_npz_file = False

# if needed, manually create the output folder
output_folder = argv[2]

os.makedirs(output_folder, exist_ok=True)

fileNamePrefix = os.path.join(output_folder, 'rosslyn_ray_tracing_60GHz')  # prefix of output files
if should_write_npz_file:
    pythonExtension = '.npz'
matlabExtension = '.hdf5'


def base_run_dir_fn(i):  # the folders will be run00001, run00002, etc.
    """returns the `run_dir` for run `i`"""
    return "run{:05d}".format(i)


def get_paths_tx_file_name(i):  # the files are model.paths.t001_02.r001.p2m, model.paths.t002_02.r001.p2m, etc.
    """return the paths .p2m Tx file name for Tx `i`"""
    return "model.paths.t{:03d}_01.r002.p2m".format(i)

def get_car_with_Tx(results_dir, i):
    run_dir = os.path.join(results_dir, base_run_dir_fn(total_num_scenes))
    #read SUMO information for this scene from text CSV file
    sumoOutputInfoFileName = os.path.join(run_dir,'sumoOutputInfoFileName.txt')
    with open(sumoOutputInfoFileName, 'r') as f:
        cars_with_Tx = []
        for row in sumoReader:
            if len(row) == 1:
                continue
            if row[3] == '-1':
                continue
            cars_with_Tx.append(row[4])
    return cars_with_Tx

last_simulation_info = None
simulation_info = None

# Object which will be modified in the RWI project
# Folder to store each InSite project and its results (will create subfolders for each "run", run0000, run0001, etc.)
results_dir = argv[1]  # 

# The info below typically does not change
# Ray-tracing output folder (where InSite will store the results (Study Area name)).
# They will be later copied to the corresponding output folder specified by results_dir
project_output_dirBaseName = 'study'
# Name (basename) of the paths file generated in the simulation
# paths_file_name = 'model.paths.t00{0}_02.r001.p2m'
# tx_paths_file_name = 'model.paths.t00{0}_02.r001.p2m'
# paths_file_name = 'model.paths.t001_01.r002.p2m'
# Output files, which are written by the Python scripts
# Name (basename) of the JSON output simulation info file
simulation_info_file_name = 'wri-simulation.info'

this_scene_i = 0  # indicates scene number within episode
total_num_scenes = 0  # all processed scenes/
ep_i = -1  # it's summed to 1 and we need to start by 0
actual_max_num_interactions = -np.Infinity  # get the max number of interactions
should_stop = False
while not should_stop:
    # This while will iterate throught all episodes
    at_least_one_valid_scene_in_this_episode = False
    # gains, phases, etc. for gains
    allEpisodeData = np.zeros((numScenesPerEpisode, numTx, numRx, numRaysPerTxRxPair,
                               numParametersPerRay), np.float32)
    allEpisodeData.fill(np.nan)
    # positions (x,y,z) of interactions (see Table 20.1: Propagation Path Interactions) of InSite Reference Manual
    allInteractionsPositions = np.zeros((numScenesPerEpisode, numTx, numRx, numRaysPerTxRxPair,
                                         max_num_interactions, 3), np.float32)  # 3 because (x,y,z)
    allInteractionsPositions.fill(np.nan)
    # Strings
    allInteractionsDescriptions = np.zeros((numScenesPerEpisode, numTx, numRx, numRaysPerTxRxPair,
                                            max_num_interactions, 2),
                                           dtype=int)  # 2 because there are at most 2 letters
    # number of interactions per ray
    allInteractionsNumbers = np.zeros((numScenesPerEpisode, numTx, numRx, numRaysPerTxRxPair), dtype=np.int64)
    allInteractionsNumbers.fill(-1)

    should_reset_episode = True
    for s in range(numScenesPerEpisode):
        
        for transmitter_number in range(numTx):

            run_dir = os.path.join(results_dir, base_run_dir_fn(total_num_scenes))
            # object_file_name = os.path.join(run_dir, dst_object_file_nameBaseName)
            # rays information but phase
            tx_path_file_name = get_paths_tx_file_name(transmitter_number + 1)
            abs_paths_file_name = os.path.join(run_dir, project_output_dirBaseName, tx_path_file_name)
            if os.path.exists(abs_paths_file_name) == False:
                print('\nWarning: could not find file ', abs_paths_file_name, ' Stopping...')
                if transmitter_number == 0: #ugly hack to avoid error with less than one Tx on a scene
                    should_stop = True
                break
            # now we get the phase info from CIR file
            abs_cir_file_name = abs_paths_file_name.replace("paths", "cir")  # name for the impulse response (cir) file
            if os.path.exists(abs_cir_file_name) == False:
                print('ERROR: could not find file ', abs_cir_file_name)
                print('Did you ask InSite to generate the impulse response (cir) file?')
                exit(-1)

            abs_simulation_info_file_name = os.path.join(run_dir, simulation_info_file_name)
            with open(abs_simulation_info_file_name) as infile:
                simulation_info = json.load(infile)

            # start of episode
            if should_reset_episode:
                should_reset_episode = False
            #if simulation_info['scene_i'] == 0 and transmitter_number == numTx-1:

                #print('ak',ep_i,transmitter_number,this_scene_i)

                ep_i += 1
                this_scene_i = 0  # reset counter
                # if episode is not None:
                #    session.add(episode)
                #    session.commit()

                # read SUMO information for this scene from text CSV file
                sumoOutputInfoFileName = os.path.join(run_dir, 'sumoOutputInfoFileName.txt')
                with open(sumoOutputInfoFileName, 'r') as f:
                    sumoReader = csv.reader(
                        f)  # AK-TODO ended up not using the CSV because the string is protected by " " I guess
                    for row in sumoReader:
                        headerItems = row[0].split(',')
                        TsString = headerItems[-1]
                        try:
                            Ts = TsString.split('=')[1]
                            timeString = headerItems[-2]
                            time = timeString.split('=')[1]
                        except IndexError:  # old format
                            Ts = 0.005  # initialize values
                            time = -1
                        break  # process only first 2 rows / line AK-TODO should eliminate the loop
                    for row in sumoReader:
                        thisEpisodeNumber = int(row[0])
                        if thisEpisodeNumber != ep_i:
                            print('ERROR: thisEpisodeNumber != ep_i. They are:', thisEpisodeNumber, 'and', ep_i,
                                  'file: ', sumoOutputInfoFileName, 'read:', row)
                            exit(1)
                        break  # process only first 2 rows / line AK-TODO should eliminate the loop

            if simulation_info['scene_i'] != this_scene_i:
                raise SceneNotInEpisodeSequenceError('Expecting {} found {}'.format(
                    this_scene_i,
                    simulation_info['scene_i'],
                ))

            print(abs_paths_file_name)  # AK TODO take out this comment and use logging
            paths = P2mPaths(abs_paths_file_name)
            cir = P2mCir(abs_cir_file_name)

            for receiver_number in range(paths.n_receivers):
                if paths.get_total_received_power(receiver_number + 1) is not None:
                    total_received_power = paths.get_total_received_power(receiver_number + 1)
                    mean_time_of_arrival = paths.get_mean_time_of_arrival(receiver_number + 1)

                    sixParameters = paths.get_6_parameters_for_all_rays(receiver_number + 1)
                    numRays = sixParameters.shape[0]
                    areLOSChannels = paths.is_los(receiver_number + 1)
                    phases = cir.get_phase_ndarray(receiver_number + 1)  # get phases for all rays in degrees
                    # go from 0:numRays to support a number of valid rays smaller than the maximum
                    allEpisodeData[this_scene_i, transmitter_number, receiver_number, 0:numRays, 0:6] = sixParameters
                    allEpisodeData[this_scene_i, transmitter_number, receiver_number, 0:numRays, 6] = areLOSChannels
                    if numParametersPerRay >= 8:
                        allEpisodeData[this_scene_i, transmitter_number, receiver_number, 0:numRays, 7] = phases
                    
                    #include Rx angle and Tx angle
                    sumoOutputInfoFileName = os.path.join(run_dir, 'sumoOutputInfoFileName.txt')
                    with open(sumoOutputInfoFileName, 'r') as f:
                        reader = f.read().split('\n')
                        for row in reader:
                            if row == reader[0] or len(row)<1:
                                continue #skip the header
                            row_tmp = row.split(',')
                            if int(row_tmp[2]) == receiver_number:
                                allEpisodeData[this_scene_i, transmitter_number, receiver_number, 0:numRays, 9] = float(row_tmp[13])
                            if int(row_tmp[3]) == transmitter_number:
                                allEpisodeData[this_scene_i, transmitter_number, receiver_number, 0:numRays, 8] = float(row_tmp[13])


                    interactions_strings = paths.get_interactions_list(receiver_number + 1)
                    for ray_i in range(numRays):
                        # interactions positions
                        interactions_positions = paths.get_interactions_positions(receiver_number + 1, ray_i + 1)
                        theseInteractions = interactions_strings[ray_i].split('-')
                        num_interactions = len(interactions_positions)
                        allInteractionsNumbers[
                            this_scene_i, transmitter_number,receiver_number, ray_i] = num_interactions  # keep the number of interactions
                        if num_interactions > actual_max_num_interactions:
                            actual_max_num_interactions = num_interactions  # update
                        if num_interactions > max_num_interactions:
                            print('ERROR: Found num of interactions = ', num_interactions,
                                'while you specified the maximum is', max_num_interactions)
                            exit(-1)
                        for interaction_i in range(num_interactions):
                            allInteractionsPositions[this_scene_i, transmitter_number, receiver_number, ray_i, interaction_i] = \
                                interactions_positions[interaction_i]
                            stringAsBytes = theseInteractions[
                                interaction_i].encode()  # https://www.mkyong.com/python/python-3-convert-string-to-bytes/
                            allInteractionsDescriptions[this_scene_i, transmitter_number, receiver_number, ray_i, interaction_i, 0] = int(
                                stringAsBytes[0])
                            if len(stringAsBytes) > 1:
                                allInteractionsDescriptions[this_scene_i, transmitter_number, receiver_number, ray_i, interaction_i, 1] = int(
                                    stringAsBytes[1])

                            # episode.scenes.append(scene)
                    at_least_one_valid_scene_in_this_episode = True  # indicate this episode has at least one valid scene
        #if should_stop: 
        #    break # AKTODO ugly hack just to break extra loop and avoid error message
        print('\rProcessed episode: {} scene: {}, total {} '.format(ep_i, this_scene_i, total_num_scenes), end='')
        this_scene_i += 1
        total_num_scenes += 1  # increment loop counter

    if at_least_one_valid_scene_in_this_episode:
        if should_write_npz_file:
            outputFileName = fileNamePrefix + '_e' + str(ep_i) + pythonExtension
            np.savez(outputFileName, allEpisodeData=allEpisodeData)
            print('==> Wrote file ' + outputFileName)

        if should_write_interactions:
            if should_write_npz_file:
                outputFileName = fileNamePrefix + '_e' + str(ep_i) + '_interactions' + pythonExtension
                np.savez(outputFileName, allInteractionsPositions=allInteractionsPositions, \
                         allInteractionsDescriptions=allInteractionsDescriptions,
                         allInteractionsNumbers=allInteractionsNumbers)
                print('==> Wrote file ' + outputFileName)

        outputFileName = fileNamePrefix + '_e' + str(ep_i) + matlabExtension
        print('==> Wrote file ' + outputFileName)
        f = h5py.File(outputFileName, 'w')
        f['allEpisodeData'] = allEpisodeData
        f.close()

        if should_write_interactions:
            # because they are large arrays, store interactions in another file
            outputFileName = fileNamePrefix + '_e' + str(ep_i) + '_interactions' + matlabExtension
            print('==> Wrote file ' + outputFileName)
            f = h5py.File(outputFileName, 'w')
            f['allInteractionsPositions'] = allInteractionsPositions
            f['allInteractionsDescriptions'] = allInteractionsDescriptions
            f['allInteractionsNumbers'] = allInteractionsNumbers
            f.close()

print()
print('Processed ', total_num_scenes, ' scenes (RT simulations)')
if max_num_interactions != actual_max_num_interactions:
    print('Found a max num of interactions = ', actual_max_num_interactions, 'while you specified = ',
          max_num_interactions)
    print(
        'Maybe you can consider re-running in case you do not want to waste some space in the array that stores interaction strings')
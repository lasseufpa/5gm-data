'''
Read channel information (rays organized as npz files) and output
indices of best pair of beams.
'''
import numpy as np
import os
# import rwisimulation as rwi #.calcrxpower import getNarrowBandULAMIMOChannel
from rwisimulation.calcrxpower import getNarrowBandULAMIMOChannel


def main():
    number_Rx_antennas = 8
    number_Tx_antennas = 32
    normalizedAntDistance = 0.5
    angleWithArrayNormal = 0  # use 0 when the angles are provided by InSite
    numberEpisodes = 119  # total number of episodes
    outputFolder = './outputnn/'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if os.name == 'nt':
        #116 episodes
        #inputPath = 'D:/github/5gm-data/insitedata/urban_canyon_v2i_5gmv1_rays_e'
        #119 episodes
        inputPath = 'D:/ak/Works/2018-proj-beam-sense-ml-lidar/lidar_and_insite/e119/insitedata/urban_canyon_v2i_5gmv1_positionMatrix_e'
    else:
        #inputPath = '/mnt/d/github/5gm-data/insitedata/urban_canyon_v2i_5gmv1_rays_e'
        inputPath = '/mnt/d/ak/Works/2018-proj-beam-sense-ml-lidar/lidar_and_insite/e119/insitedata/urban_canyon_v2i_5gmv1_positionMatrix_e'

    # initialize variables
    numOfValidChannels = 0
    numOfInvalidChannels = 0
    numLOS = 0
    numNLOS = 0
    numOccurrencesTxIndices = np.zeros((number_Tx_antennas,), dtype=np.int)
    numOccurrencesRxIndices = np.zeros((number_Rx_antennas,), dtype=np.int)
    for e in range(numberEpisodes):
        print("Episode # ", (e + 1))
        # if using owncloud files
        # b = np.load('d:/github/5gm-data/insitedata/urban_canyon_v2i_5gmv1_rays_e'+str(e+1)+'.npz')
        b = np.load(inputPath + str(e + 1) + '.npz')
        # b = np.load('./insitedata/urban_canyon_v2i_5gmv1_rays_e'+str(n+1)+'.npz')
        allEpisodeData = b['allEpisodeData']
        numScenes = allEpisodeData.shape[0]
        numReceivers = allEpisodeData.shape[1]
        #store two integers converted to 1
        episodeOutputs = np.nan * np.ones((numScenes, numReceivers),
                                    np.float32)
        for s in range(numScenes):  # 50
            for r in range(numReceivers):  # 10
                insiteData = allEpisodeData[s, r, :, :]
                if sum(np.isnan(insiteData.flatten())) > 0:
                    numOfInvalidChannels += 1
                    continue  # next Tx / Rx pair
                numOfValidChannels += 1
                gain_in_dB = insiteData[:, 0]
                timeOfArrival = insiteData[:, 1]
                # InSite provides angles in degrees. Convert to radians
                # This conversion is being done within the channel function
                if True:  # use angles in degrees (convert later)
                    AoD_el = insiteData[:, 2]
                    AoD_az = insiteData[:, 3]
                    AoA_el = insiteData[:, 4]
                    AoA_az = insiteData[:, 5]
                else:  # convert now
                    AoD_el = np.deg2rad(insiteData[:, 2])
                    AoD_az = np.deg2rad(insiteData[:, 3])
                    AoA_el = np.deg2rad(insiteData[:, 4])
                    AoA_az = np.deg2rad(insiteData[:, 5])

                isLOSperRay = insiteData[:, 6]

                if False:  # enable for debugging with fixed angles
                    ad = -(np.pi / 4) * 180 / np.pi  # in degrees, as InSite provides
                    aa = (3 * np.pi / 2) * 180 / np.pi
                    g = 10
                    AoD_az = ad * np.ones(AoD_az.shape, AoD_az.dtype)
                    AoA_az = ad * np.ones(AoA_az.shape, AoA_az.dtype)
                    gain_in_dB = g * np.ones(gain_in_dB.shape, gain_in_dB.dtype)

                # order the rays to have the shortest path first
                # [timeOfArrival,sortedIndices] = sort(timeOfArrival);
                # theseRays=[];
                # theseRays.gainMagnitude = gainMagnitude(sortedIndices);
                # theseRays.timeOfArrival = timeOfArrival;
                # theseRays.AoA_el = AoA_el(sortedIndices); %not currently used
                # theseRays.AoD_el = AoD_el(sortedIndices); %not currently used
                # theseRays.AoA_az = AoA_az(sortedIndices);
                # theseRays.AoD_az = AoD_az(sortedIndices);
                # theseRays.isLOS = isLOS(sortedIndices);

                # in case any of the rays in LOS, then indicate that the output is 1
                isLOS = 0  # for the channel
                if np.sum(isLOSperRay) > 0:
                    isLOS = 1
                    numLOS += 1
                else:
                    numNLOS += 1

                equivalentChannel = getNarrowBandULAMIMOChannel(AoD_az, AoA_az, gain_in_dB, number_Tx_antennas,
                                                                number_Rx_antennas, normalizedAntDistance,
                                                                angleWithArrayNormal)
                equivalentChannel = np.abs(equivalentChannel)
                # print(equivalentChannel)
                (bestRxIndex, bestTxIndex) = np.unravel_index(np.argmax(equivalentChannel, axis=None),
                                                              equivalentChannel.shape)
                numOccurrencesTxIndices[bestTxIndex] += 1  # increment counters
                numOccurrencesRxIndices[bestRxIndex] += 1
                # if bestRxIndex + bestTxIndex != 0:
                # print('bestRxIndex: ', bestRxIndex, ' and bestTxIndex: ', bestTxIndex)
                #    exit(1)
                outputLabel = bestTxIndex * number_Rx_antennas + bestRxIndex
                # when one needs to recover the labels:
                #recoverLabelTx = np.floor(outputLabel/number_Rx_antennas)
                #recoverLabelRx = outputLabel - recoverLabelTx*number_Rx_antennas
                episodeOutputs[s,r]=outputLabel

            #finished processing this episode
        npz_name = outputFolder + 'output_e_' +str(e+1)+'.npz'
        np.savez(npz_name, output=episodeOutputs)
        print('Saved file ', npz_name)


    print('total numOfInvalidChannels = ', numOfInvalidChannels)
    print('total numOfValidChannels = ', numOfValidChannels)
    print('Sum = ', numOfValidChannels + numOfInvalidChannels)

    print('total numNLOS = ', numNLOS)
    print('total numLOS = ', numLOS)
    print('Sum = ', numLOS + numNLOS)

    print('Tx indices histogram:')
    print(numOccurrencesTxIndices)
    print('Rx indices histogram:')
    print(numOccurrencesRxIndices)


if __name__ == '__main__':
    main()

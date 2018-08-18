'''
Read channel information (rays organized as npz files) and output the complex-valued
equivalent channels (not anymore only the indices of best pair of beams, which can
be calculated with the generated data).
'''
import numpy as np
import os
from mimo_channels import getCodebookOperatedChannel, readUPASteeringCodebooks, getNarrowBandULAMIMOChannel, getNarrowBandUPAMIMOChannel, getDFTOperatedChannel

def main():
    useUPA = True
    if useUPA == True:
        if False:
            number_Tx_antennasX = 16
            number_Tx_antennasY = 2
            number_Rx_antennasX = 4
            number_Rx_antennasY = 2
        else:
            #to get statistics:
            #txCodebookInputFileName = 'D:/gits/lasse/software/mimo-matlab/upa_codebook_12x12.mat'
            #rxCodebookInputFileName = 'D:/gits/lasse/software/mimo-matlab/upa_codebook_12x12.mat'
            txCodebookInputFileName = 'D:/gits/lasse/software/mimo-matlab/tx_upa_codebook_12x12_valid.mat'
            rxCodebookInputFileName = 'D:/gits/lasse/software/mimo-matlab/rx_upa_codebook_12x12_valid.mat'
            Wt, number_Tx_antennasX, number_Tx_antennasY = readUPASteeringCodebooks(txCodebookInputFileName)
            Wr, number_Rx_antennasX, number_Rx_antennasY = readUPASteeringCodebooks(rxCodebookInputFileName)
            number_Tx_vectors = Wt.shape[1]
            number_Rx_vectors = Wr.shape[1]
            print(number_Tx_antennasX, number_Tx_antennasY, number_Rx_antennasX, number_Rx_antennasY, number_Tx_vectors, number_Rx_vectors)
        number_Tx_antennas = number_Tx_antennasX * number_Tx_antennasY
        number_Rx_antennas = number_Rx_antennasX * number_Rx_antennasY
    else:
        number_Tx_antennas = 32
        number_Rx_antennas = 8
    normalizedAntDistance = 0.5
    angleWithArrayNormal = 0  # use 0 when the angles are provided by InSite
    numberEpisodes = 1372 #119  # total number of episodes
    #outputFolder = './outputnn/'
    outputFolder = 'D:/github/5gm-data/outputnn/'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if os.name == 'nt':
        #116 episodes
        #inputPath = 'D:/github/5gm-data/insitedata/urban_canyon_v2i_5gmv1_rays_e'
        #119 episodes
        #inputPath = 'D:/ak/Works/2018-proj-beam-sense-ml-lidar/lidar_and_insite/e119/insitedata/urban_canyon_v2i_5gmv1_positionMatrix_e'
        inputPath = 'D:/github/5gm-data/insitedata/urban_canyon_v2i_5gmv1_rays_e'
    else:
        #inputPath = '/mnt/d/github/5gm-data/insitedata/urban_canyon_v2i_5gmv1_rays_e'
        #inputPath = '/mnt/d/ak/Works/2018-proj-beam-sense-ml-lidar/lidar_and_insite/e119/insitedata/urban_canyon_v2i_5gmv1_positionMatrix_e'
        inputPath = '/mnt/d/github/5gm-data/insitedata/urban_canyon_v2i_5gmv1_rays_e'

    # initialize variables
    numOfValidChannels = 0
    numOfInvalidChannels = 0
    numLOS = 0
    numNLOS = 0
    numOccurrencesTxIndices = np.zeros((number_Tx_vectors,), dtype=np.int)
    numOccurrencesRxIndices = np.zeros((number_Rx_vectors,), dtype=np.int)
    numOccurrencesBeamPairIndices = np.zeros((np.maximum(number_Tx_vectors,number_Rx_vectors)**2,), dtype=np.int)
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
        episodeOutputs = np.nan * np.ones((numScenes, numReceivers, number_Rx_vectors, number_Tx_vectors),
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
                pathPhases = insiteData[:, 7]

                #AK TODO check
                AoA_az = AoA_az - 90
                AoD_az = AoD_az - 90

                if False:  # enable for debugging with fixed angles
                    ad = -(np.pi / 4) * 180 / np.pi  # in degrees, as InSite provides
                    aa = (3 * np.pi / 2) * 180 / np.pi
                    g = 10
                    AoD_az = ad * np.ones(AoD_az.shape, AoD_az.dtype)
                    AoA_az = ad * np.ones(AoA_az.shape, AoA_az.dtype)
                    gain_in_dB = g * np.ones(gain_in_dB.shape, gain_in_dB.dtype)
                    pathPhases = np.zeros(pathPhases.shape)

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

                if useUPA == True:
                    #departure_angles = np.array((AoD_el,AoD_az)).T
                    #arrival_angles = np.array((AoA_el,AoA_az)).T
                    #calc_rx_power(departure_angles, arrival_angles, gain_in_dB, number_Tx_antennas, frequency=6e10)

                    mimoChannel = getNarrowBandUPAMIMOChannel(AoD_el,AoD_az,AoA_el,AoA_az, gain_in_dB,pathPhases,
                                                              number_Tx_antennasX, number_Tx_antennasY, number_Rx_antennasX,
                                                              number_Rx_antennasY,
                                                              normalizedAntDistance)
                    equivalentChannel = getCodebookOperatedChannel(mimoChannel, Wt, Wr)
                else:
                    mimoChannel = getNarrowBandULAMIMOChannel(AoD_az, AoA_az, gain_in_dB, number_Tx_antennas,
                                                                number_Rx_antennas, normalizedAntDistance,
                                                                angleWithArrayNormal)
                    equivalentChannel = getDFTOperatedChannel(mimoChannel, number_Tx_antennas, number_Rx_antennas)
                equivalentChannelMagnitude = np.abs(equivalentChannel)
                # print(equivalentChannel)
                bestBeamPairIndex = np.argmax(equivalentChannelMagnitude, axis=None)
                numOccurrencesBeamPairIndices[bestBeamPairIndex] += 1
                (bestRxIndex, bestTxIndex) = np.unravel_index(bestBeamPairIndex,
                                                              equivalentChannelMagnitude.shape)
                numOccurrencesTxIndices[bestTxIndex] += 1  # increment counters
                numOccurrencesRxIndices[bestRxIndex] += 1
                # if bestRxIndex + bestTxIndex != 0:
                # print('bestRxIndex: ', bestRxIndex, ' and bestTxIndex: ', bestTxIndex)
                #    exit(1)
                outputLabel = bestTxIndex * number_Rx_antennas + bestRxIndex
                # when one needs to recover the labels:
                #recoverLabelTx = np.floor(outputLabel/number_Rx_antennas)
                #recoverLabelRx = outputLabel - recoverLabelTx*number_Rx_antennas
                episodeOutputs[s,r]=np.abs(equivalentChannel)

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

    print('tx_indices_histogram = [', end=" ")
    for i in range(len(numOccurrencesTxIndices)):
        print(numOccurrencesTxIndices[i], end=" ")
    print('];')
    print('rx_indices_histogram = [', end=" ")
    for i in range(len(numOccurrencesRxIndices)):
        print(numOccurrencesRxIndices[i], end=" ")
    print('];')
    if False:
        print('Maximum among beam pair indices histogram:')
        print(np.amax(numOccurrencesBeamPairIndices))
        print('Beam pair indices histogram:')
        for i in range(len(numOccurrencesBeamPairIndices)):
            print(numOccurrencesBeamPairIndices[i], ' ')


if __name__ == '__main__':
    main()

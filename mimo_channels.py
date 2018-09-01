import os
import datetime

import numpy as np
from rwiparsing import P2mPaths, P2mCir
from akpy.matlab_tofrom_python import read_matlab_array_from_mat

def calc_omega(elevationAngles, azimuthAngles, normalizedAntDistance = 0.5):
    sinElevations = np.sin(elevationAngles)
    omegax = 2 * np.pi * normalizedAntDistance * sinElevations * np.cos(azimuthAngles)
    omegay = 2 * np.pi * normalizedAntDistance * sinElevations * np.sin(azimuthAngles)
    return np.matrix((omegax, omegay))

def calc_vec_i(i, omega, antenna_range):
    print('a ', omega[:, i])
    print('b ', omega[:, i].shape)
    vec = np.exp(1j * omega[:, i] * antenna_range)
    print('c ', np.matrix(np.kron(vec[1], vec[0])).shape)
    return np.matrix(np.kron(vec[1], vec[0]))

def dft_codebook(dim):
    seq = np.matrix(np.arange(dim))
    mat = seq.conj().T * seq
    w = np.exp(-1j * 2 * np.pi * mat / dim)
    return w

def getNarrowBandUPAMIMOChannel(departureElevation,departureAzimuth,arrivalElevation,arrivalAzimuth, p_gainsdB,
                                pathPhases, number_Tx_antennasX, number_Tx_antennasY, number_Rx_antennasX,
                                number_Rx_antennasY, normalizedAntDistance=0.5):
    """Uses UPAs at both TX and RX.
    Will assume that all 4 normalized distances (Tx and Rx, x and y) are the same.
    """
    number_Tx_antennas = number_Tx_antennasX * number_Tx_antennasY
    number_Rx_antennas = number_Rx_antennasX * number_Rx_antennasY
    departureElevation = np.deg2rad(departureElevation)
    departureAzimuth = np.deg2rad(departureAzimuth)
    arrivalElevation = np.deg2rad(arrivalElevation)
    arrivalAzimuth = np.deg2rad(arrivalAzimuth)

    numRays = np.shape(departureElevation)[0]
    #number_Rx_antennas is the total number of antenna elements of the array, idem Tx
    H = np.matrix(np.zeros((number_Rx_antennas, number_Tx_antennas)))

    path_gain = np.power(10, p_gainsdB / 10)

    #generate uniformly distributed random phase in radians
    if pathPhases is None:
        pathPhases = 2*np.pi * np.random.rand(len(path_gain))
    else:
        #convert from degrees to radians the phase obtained with simulator (InSite)
        pathPhases = np.deg2rad(pathPhases)

    #include phase information, converting gains in complex-values
    path_complexGains = path_gain * np.exp(1j * pathPhases)

    # recall that in the narrowband case, the time-domain H is the same as the
    # frequency-domain H
    # Each vector below has the x and y values for each ray. Example 2 x 25 dimension
    departure_omega = calc_omega(departureElevation, departureAzimuth, normalizedAntDistance)
    arrival_omega = calc_omega(arrivalElevation, arrivalAzimuth, normalizedAntDistance)

    rangeTx_x = np.arange(number_Tx_antennasX)
    rangeTx_y = np.arange(number_Tx_antennasY)
    rangeRx_x = np.arange(number_Rx_antennasX)
    rangeRx_y = np.arange(number_Rx_antennasY)
    for ray_i in range(numRays):
        #departure
        vecx = np.exp(1j * departure_omega[0,ray_i] * rangeTx_x)
        vecy = np.exp(1j * departure_omega[1,ray_i] * rangeTx_y)
        departure_vec = np.matrix(np.kron(vecy, vecx))
        #arrival
        vecx = np.exp(1j * arrival_omega[0,ray_i] * rangeRx_x)
        vecy = np.exp(1j * arrival_omega[1,ray_i] * rangeRx_y)
        arrival_vec = np.matrix(np.kron(vecy, vecx))

        H = H + path_complexGains[ray_i] * arrival_vec.conj().T * departure_vec
    return H

def getNarrowBandULAMIMOChannel(azimuths_tx, azimuths_rx, p_gainsdB, number_Tx_antennas, number_Rx_antennas,
                                normalizedAntDistance=0.5, angleWithArrayNormal=0, pathPhases=None):
    """This .m file uses ULAs at both TX and RX.

    - assumes one beam per antenna element

    the first column will be the elevation angle, and the second column is the azimuth angle correspondingly.
    p_gain will be a matrix size of (L, 1)
    departure angle/arrival angle will be a matrix as size of (L, 2), where L is the number of paths

    t1 will be a matrix of size (nt, nr), each
    element of index (i,j) will be the received
    power with the i-th precoder and the j-th
    combiner in the departing and arrival codebooks
    respectively

    :param departure_angles: ((elevation angle, azimuth angle),) (L, 2) where L is the number of paths
    :param arrival_angles: ((elevation angle, azimuth angle),) (L, 2) where L is the number of paths
    :param p_gaindB: path gain (L, 1) in dB where L is the number of paths
    :param number_Rx_antennas, number_Tx_antennas: number of antennas at Rx and Tx, respectively
    :param pathPhases: in degrees, same dimension as p_gaindB
    :return:
    """
    azimuths_tx = np.deg2rad(azimuths_tx)
    azimuths_rx = np.deg2rad(azimuths_rx)
    # nt = number_Rx_antennas * number_Tx_antennas #np.power(antenna_number, 2)
    m = np.shape(azimuths_tx)[0]  # number of rays
    H = np.matrix(np.zeros((number_Rx_antennas, number_Tx_antennas)))

    gain_dB = p_gainsdB
    path_gain = np.power(10, gain_dB / 10)
    path_gain = np.sqrt(path_gain)

    #generate uniformly distributed random phase in radians
    if pathPhases is None:
        pathPhases = 2*np.pi * np.random.rand(len(path_gain))
    else:
        #convert from degrees to radians
        pathPhases = np.deg2rad(pathPhases)

    #include phase information, converting gains in complex-values
    path_complexGains = path_gain * np.exp(1j * pathPhases)

    # recall that in the narrowband case, the time-domain H is the same as the
    # frequency-domain H
    for i in range(m):
        # at and ar are row vectors (using Python's matrix)
        at = np.matrix(arrayFactorGivenAngleForULA(number_Tx_antennas, azimuths_tx[i], normalizedAntDistance,
                                                   angleWithArrayNormal))
        ar = np.matrix(arrayFactorGivenAngleForULA(number_Rx_antennas, azimuths_rx[i], normalizedAntDistance,
                                                   angleWithArrayNormal))
        H = H + path_complexGains[i] * ar.conj().T * at  # outer product of ar Hermitian and at
    factor = (np.linalg.norm(path_complexGains) / np.sum(path_complexGains)) * np.sqrt(
        number_Rx_antennas * number_Tx_antennas)  # scale channel matrix
    H *= factor  # normalize for compatibility with Anum's Matlab code

    return H

def arrayFactorGivenAngleForULA(numAntennaElements, theta, normalizedAntDistance=0.5, angleWithArrayNormal=0):
    '''
    Calculate array factor for ULA for angle theta. If angleWithArrayNormal=0
    (default),the angle is between the input signal and the array axis. In
    this case when theta=0, the signal direction is parallel to the array
    axis and there is no energy. The maximum values are for directions 90
        and -90 degrees, which are orthogonal with array axis.
    If angleWithArrayNormal=1, angle is with the array normal, which uses
    sine instead of cosine. In this case, the maxima are for
        thetas = 0 and 180 degrees.
    References:
    http://www.waves.utoronto.ca/prof/svhum/ece422/notes/15-arrays2.pdf
    Book by Balanis, book by Tse.
    '''
    indices = np.arange(numAntennaElements)
    if (angleWithArrayNormal == 1):
        arrayFactor = np.exp(-1j * 2 * np.pi * normalizedAntDistance * indices * np.sin(theta))
    else:  # default
        arrayFactor = np.exp(-1j * 2 * np.pi * normalizedAntDistance * indices * np.cos(theta))
    return arrayFactor / np.sqrt(numAntennaElements)  # normalize to have unitary norm

def calc_rx_power(departure_angle, arrival_angle, p_gain, antenna_number, frequency=6e10):
    """This .m file uses a m*m SQUARE UPA, so the antenna number at TX, RX will be antenna_number^2.

    - antenna_number^2 number of element arrays in TX, same in RX
    - assumes one beam per antenna element

    the first column will be the elevation angle, and the second column is the azimuth angle correspondingly.
    p_gain will be a matrix size of (L, 1)
    departure angle/arrival angle will be a matrix as size of (L, 2), where L is the number of paths

    t1 will be a matrix of size (nt, nr), each
    element of index (i,j) will be the received
    power with the i-th precoder and the j-th
    combiner in the departing and arrival codebooks
    respectively

    :param departure_angle: ((elevation angle, azimuth angle),) (L, 2) where L is the number of paths
    :param arrival_angle: ((elevation angle, azimuth angle),) (L, 2) where L is the number of paths
    :param p_gain: path gain (L, 1) where L is the number of paths
    :param antenna_number: antenna number at TX, RX is antenna_number**2
    :param frequency: default
    :return:
    """
    departure_angle = np.deg2rad(departure_angle)
    arrival_angle = np.deg2rad(arrival_angle)
    c = 3e8
    mlambda = c / frequency
    k = 2 * np.pi / mlambda
    d = mlambda / 2
    nt = np.power(antenna_number, 2)
    m = np.shape(departure_angle)[0]
    nr = nt
    wt = dft_codebook(nt)
    wr = dft_codebook(nr)
    H = np.matrix(np.zeros((nt, nr)))

    # TO DO: need to generate random phase and convert gains in complex-values
    gain_dB = p_gain
    path_gain = np.power(10, gain_dB / 10)
    antenna_range = np.arange(antenna_number)

    def calc_omega(angle):
        sin = np.sin(angle)
        omegay = k * d * sin[:, 1] * sin[:, 0]
        omegax = k * d * sin[:, 0] * np.cos(angle[:, 1])
        return np.matrix((omegax, omegay))

    departure_omega = calc_omega(departure_angle)
    arrival_omega = calc_omega(arrival_angle)

    def calc_vec_i(i, omega, antenna_range):
        vec = np.exp(1j * omega[:, i] * antenna_range)
        return np.matrix(np.kron(vec[1], vec[0]))

    for i in range(m):
        departure_vec = calc_vec_i(i, departure_omega, antenna_range)
        arrival_vec = calc_vec_i(i, arrival_omega, antenna_range)
        H = H + path_gain[i] * departure_vec.conj().T * arrival_vec
    t1 = wt.conj().T * H * wr
    return t1

def getDFTOperatedChannel(H, number_Tx_antennas, number_Rx_antennas):
    wt = dft_codebook(number_Tx_antennas)
    wr = dft_codebook(number_Rx_antennas)
    dictionaryOperatedChannel = wr.conj().T * H * wt
    # dictionaryOperatedChannel2 = wr.T * H * wt.conj()
    return dictionaryOperatedChannel  # return equivalent channel after precoding and combining

def getCodebookOperatedChannel(H, Wt, Wr):
    if Wr is None: #only 1 antenna at Rx, and Wr was passed as None
        return H * Wt
    if Wt is None: #only 1 antenna at Tx
        return Wr.conj().T * H
    return Wr.conj().T * H * Wt # return equivalent channel after precoding and combining

def readUPASteeringCodebooks(inputFileName):
    '''Read data created by
    D:/gits/lasse/software/mimo-matlab/upa_codebook_creation.m
    Used the Kronecker to represent the matrix for a pair of wx and wy as a single array
    %See  John Brady, Akbar Sayeed, Millimeter-Wave MIMO Transceivers - Chap 10
    %Section 10.5
    %http://dune.ece.wisc.edu/wp-uploads/2015/11/main_sayeed_brady.pdf
    '''
    #saved redundant information on Wx and Wy in case they help later
    #Used on Matlab:
    #save(outputFileName,'W','codebook','Wx','Wy','Nax','Nay','-v6')
    arrayName = 'W'
    codevectors = read_matlab_array_from_mat(inputFileName, arrayName)
    arrayName = 'Nax'
    Nx = read_matlab_array_from_mat(inputFileName, arrayName)
    arrayName = 'Nay'
    Ny = read_matlab_array_from_mat(inputFileName, arrayName)
    return codevectors, int(Nx), int(Ny)

def test_channel():
    # RESULTS_DIR='/Users/psb/ownCloud/Projects/DNN Wireless/rwi-3d-modeling/restuls/run00000'
    # RESULTS_DIR = 'D:/insitedata/pc128/example_working/restuls/run00000'
    RESULTS_DIR='D:/github/5gm-rwi-simulation/example/results_new_simuls/run00003/'
    antenna_number = 2
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    P2MPATHS_FILE = os.path.join(RESULTS_DIR, 'study', 'model.paths.t001_01.r002.p2m')
    print('Reading file', P2MPATHS_FILE)
    with open(P2MPATHS_FILE, 'rb') as infile:
        paths = P2mPaths(P2MPATHS_FILE)
        #get info for a given receiver
        rec_i = 0 #first receiver is 0 here and 1 in the database
        departure_angles = paths.get_departure_angle_ndarray(rec_i+1)
        arrival_angles = paths.get_arrival_angle_ndarray(rec_i+1)
        p_gainsdB = paths.get_p_gain_ndarray(rec_i+1)
        abs_cir_file_name = P2MPATHS_FILE.replace("paths", "cir")  # name for the impulse response (cir) file
        if os.path.exists(abs_cir_file_name) == False:
            print('ERROR: could not find file ', abs_cir_file_name)
            print('Did you ask InSite to generate the impulse response (cir) file?')
            exit(-1)
        cir = P2mCir(abs_cir_file_name) #read impulse response with phases
        pathPhasesInDegrees = cir.get_phase_ndarray(rec_i+1)

    if True:  # enable for debugging with fixed angles
        ad = (np.pi / 4) * 180 / np.pi  # in degrees, as InSite provides
        aa = (3 * np.pi / 2) * 180 / np.pi
        g = 10
        departure_angles = ad * np.ones(departure_angles.shape, departure_angles.dtype)
        arrival_angles = aa * np.ones(arrival_angles.shape, arrival_angles.dtype)
        p_gainsdB = g * np.ones(p_gainsdB.shape, p_gainsdB.dtype)
        pathPhasesInDegrees = np.zeros(len(p_gainsdB))

    print(arrival_angles.shape)
    azimuths_tx = departure_angles[:, 1]  # azimuth is 2nd column
    azimuths_rx = arrival_angles[:, 1]

    start = datetime.datetime.today()
    t1_py = calc_rx_power(departure_angles, arrival_angles, p_gainsdB, antenna_number)

    number_Rx_antennas = antenna_number * antenna_number
    number_Tx_antennas = antenna_number * antenna_number
    normalizedAntDistance = 0.5

    t1 = getNarrowBandUPAMIMOChannel(departure_angles, arrival_angles, p_gainsdB, number_Tx_antennas,
                                     number_Rx_antennas, normalizedAntDistance=0.5)
    t1 = np.abs(t1)
    t2 = getNarrowBandULAMIMOChannel(azimuths_tx, azimuths_rx, p_gainsdB, number_Tx_antennas, number_Rx_antennas,
                                     normalizedAntDistance=0.5, angleWithArrayNormal=1, pathPhases=pathPhasesInDegrees)
    print('MSE 1 = ', np.mean(np.power(np.abs(t1 - t1_py), 2)))
    print('MSE 2 = ', np.mean(np.power(np.abs(t1 - t2), 2)))
    # print(t2)
    t2 = np.abs(t2)
    (bestRxIndex, bestTxIndex) = np.unravel_index(np.argmax(t2, axis=None), t2.shape)
    print('bestRxIndex: ', bestRxIndex, ' and bestTxIndex: ', bestTxIndex)

    stop = datetime.datetime.today()
    print(stop - start)
    # print(t1_py)

def test_readUPASteeringCodebooks():
    #inputFileName = 'D:/gits/lasse/software/mimo-matlab/upa_codebook_12x12.mat'
    inputFileName = 'D:/gits/lasse/software/mimo-matlab/upa_codebook_2x3.mat'
    codevectors, Nx, Ny = readUPASteeringCodebooks(inputFileName)
    print('#1 = ', codevectors[:,0])
    print('#4 = ', codevectors[:,3])
    print(Nx, Ny)

if __name__ == '__main__':
    #test_channel()
    test_readUPASteeringCodebooks()

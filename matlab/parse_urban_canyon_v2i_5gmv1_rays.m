clear *
%close all
rng(1)

shouldShowDebugPlots = 0; %use 1 to show plots

%compose as fileName = [insitePath filePrefix num2str(episode) extension]
insitePath='D:\github\5gm-data\insitedata_canyon\';
filePrefix='urban_canyon_v2i_5gmv1_rays_e';
extension='.hdf5';

powerThreshold=0.99; %threshold, say 99% of total power
maxChannelLength = -Inf;

numEpisodes = 116;
numOfValidChannels=0;
for e=1:numEpisodes
    fileName = [insitePath filePrefix num2str(e) extension];
    disp(['Processing ' fileName])
    %Read hdf5 file in Matlab
    %https://stackoverflow.com/questions/21624653/python-created-hdf5-dataset-transposed-in-matlab
    %do not forget the / as prefix onf the second argument below:
    allEpisodeData=h5read(fileName,'/allEpisodeData');
    %need to permute dimensions of 4-d tensor
    allEpisodeData = permute(allEpisodeData,ndims(allEpisodeData):-1:1);
    
    [numScenesPerEpisode, numTxRxPairsPerScene, numRaysPerTxRxPair, ...
        numVariablePerRay] = size(allEpisodeData);
    for s=1:numScenesPerEpisode
        for r=1:numTxRxPairsPerScene
            insiteData = squeeze(allEpisodeData(s,r,:,:));
            if sum(isnan(insiteData(:))) > 0
                disp([' ** Has NaN. Skipping episode ' num2str(e) ...
                    ', scene ' num2str(s), ', Tx/Rx pair (channel) ' ...
                    num2str(r)])
                continue %next Tx / Rx pair
            else
                disp(['Processing episode ' num2str(e) ', scene ' ...
                    num2str(s), ', Tx/Rx pair (channel) ' num2str(r)])
                numOfValidChannels = numOfValidChannels+1;
            end
            %organized as follows:
            %path_gain, timeOfArrival, departure_elevation,
            %departure_azimuth, arrival_elevation, arrival_azimuth, isLOS
            
            %received power
            powerLinearScale = (10.^(insiteData(:,1)/10)); %dBm to mWatts
            %assume the transmitter used power of 0 dBm
            gainMagnitude = sqrt(powerLinearScale);
            
            timeOfArrival = insiteData(:,2);
            
            %InSite provides angles in degrees. Convert to radians
            AoD_el = degtorad(insiteData(:,3));
            AoD_az = degtorad(insiteData(:,4));
            AoA_el = degtorad(insiteData(:,5));
            AoA_az = degtorad(insiteData(:,6));
            isLOS = insiteData(:,7);
            
            %order the rays to have the shortest path first
            %one could also order based on the channel gain
            [timeOfArrival,sortedIndices] = sort(timeOfArrival);
            AoA_el = AoA_el(sortedIndices); %not currently used
            AoD_el = AoD_el(sortedIndices); %not currently used
            AoA_az = AoA_az(sortedIndices);
            AoD_az = AoD_az(sortedIndices);
            gainMagnitude = gainMagnitude(sortedIndices);
            
            %want to find the percentage of energy/power in N first taps of
            %impulse response
            powerAccumulative=cumsum(gainMagnitude.^2);
            totalPower=powerAccumulative(end);
            thIndex = find(powerAccumulative >= ...
                powerThreshold*totalPower,1,'first');
            
            channelLength = timeOfArrival(end); %in seconds
            disp(['total channelLength = ' num2str(channelLength )])
            thresholdedChannelLength = timeOfArrival(thIndex); %up to x% of energy
            disp(['thresholded channelLengthInTaps = ' ...
                num2str(thresholdedChannelLength)])
            if channelLength > maxChannelLength
                maxChannelLength = channelLength;
            end
            
            if shouldShowDebugPlots == 1
                figure(1)
                clf
                subplot(211)
                plot(powerAccumulative/totalPower);
                hold on,
                plot(thIndex,1,'x');
                hold off
                
                Hk=[]; Hv=[];
                subplot(212)
                gainsdB=20*log10(gainMagnitude);
                plot(gainsdB - max(gainsdB));
            end
        end
    end
end
disp('######## Final statistics: ###########')
disp(['maxChannelLength = ' num2str(maxChannelLength) ...
    ' seconds for power threshold = ' num2str(100*powerThreshold) '%'])
Fs=1e9; %sampling frequency
Ts=1/Fs; %sampling interval
disp(['maxChannelLength = ' num2str(round(maxChannelLength/Ts)) ...
    ' taps for power threshold above and sampling frequency = ' num2str(Fs/1e6) ' MHz'])
disp(['numOfValidChannels = ' num2str(numOfValidChannels)])

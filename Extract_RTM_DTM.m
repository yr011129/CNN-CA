%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: raw I-Q radar data
% Output: RTM+DTM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clearvars
close all

addpath(genpath('./.idea'));
addpath(genpath('./classifier'));
addpath(genpath('./config'));
addpath(genpath('./logs'));
addpath(genpath('./modules'));
addpath(genpath('./results'));
addpath(genpath('./template data'));
addpath(genpath('./utils'));



% parameter setting
params = get_params_value_version2();
% constant parameters
c = params.c; % Speed of light in air (m/s)
fc = params.fc; % Center frequency (Hz)
lambda = params.lambda;
Rx = params.Rx;
Tx = params.Tx;

% configuration parameters
Fs = params.Fs;
sweepSlope = params.sweepSlope;
samples = params.samples;
loop = params.loop;

Tc = params.Tc; % us 
fft_Rang = params.fft_Rang;
fft_Vel = params.fft_Vel;
fft_Ang = params.fft_Ang;
num_crop = params.num_crop;
max_value = params.max_value; % normalization the maximum of data WITH 1843

% Creat grid table
rng_grid = params.rng_grid;
agl_grid = params.agl_grid;
vel_grid = params.vel_grid;

% Algorithm parameters
data_each_frame = samples*loop*Tx;
set_frame_number = 30;
frame_start = 1;
frame_end = set_frame_number;
Is_Windowed = 1;% 1==> Windowing before doing range and angle fft
Is_plot_rangeDop = 1;

% specify data name and load data as variable data_frames
for q = 1:100
    filename = sprintf('%d.bin', q) ;
    data_frames = readDCA1000(filename,256);

% figure('visible','on')
for i = 1:32
    % read the data of each frame, and then arrange for each chirps
    data_frame = data_frames(:, (i-1)*data_each_frame+1:i*data_each_frame);
    data_chirp = [];
    for cj = 1:Tx*loop
        temp_data = data_frame(:, (cj-1)*samples+1:cj*samples);
        data_chirp(:,:,cj) = temp_data;
    end
    
    % separate the odd-index chirps and even-index chirps for TDM-MIMO with 2 TXs
    chirp_odd = data_chirp(:,:,1:2:end);
    chirp_even = data_chirp(:,:,2:2:end);
    
    % permutation with the format [samples, Rx, chirp]
    chirp_odd2 = permute(chirp_odd, [2,1,3]);
    chirp_even2 = permute(chirp_even, [2,1,3]);
 

    % Range FFT for odd chirps
    [Rangedata_odd] = fft_range(chirp_odd2,fft_Rang,1);

        fft1d=squeeze(sum(Rangedata_odd,2));
        avg = sum(fft1d,2)/loop;%/Doppler_Number
        for chirp=1:loop
            fft1d_avg(:,chirp) = fft1d(:,chirp)-avg;
        end
        
        for ii =1:loop-1
            fft1d_MTI (:,ii) = fft1d(:,ii+1)-fft1d(:,ii);
        end
               
        for ii =1:loop-2
            fft1d_MTI2 (:,ii) = fft1d_MTI(:,ii+1)-fft1d_MTI(:,ii);
        end
       
    % Range FFT for even chirps
    [Rangedata_even] = fft_range(chirp_even2,fft_Rang,Is_Windowed);
    fft1d_even=squeeze(sum(Rangedata_even,2));
    avg_even = sum(fft1d_even,2)/loop;%/Doppler_Number
    for chirp=1:loop
        fft1d_avg_even(:,chirp) = fft1d_even(:,chirp)-avg_even;
    end
    
    for ii =1:loop-1
        fft1d_MTI_even (:,ii) = fft1d_even(:,ii+1)-fft1d_even(:,ii);
    end
    
    for ii =1:loop-2
        fft1d_MTI2_even (:,ii) = fft1d_MTI_even(:,ii+1)-fft1d_MTI_even(:,ii);
    end
    
      gesture_data_even(:,i)=fft1d_MTI_even (:,1); 
      gesture_data2_even(:,(i-1)*128+1:i*128)=Rangedata_even (:,1,1:128);   
      gesture_data3_even(:,(i-1)*127+1:i*127)=fft1d_MTI_even (:,1:127);  
      gesture_data4_even(:,(i-1)*126+1:i*126)=fft1d_MTI2_even (:,1:126);  
   
      
      fft1d_MTI_sum_even(:,i) = sum(fft1d_MTI_even,2);
      fft1d_MTI_sum2_even(:,i) = sum(fft1d_MTI2_even,2);
    
%     gesture_data(:,(i-1)*128+1:i*128)=Rangedata_odd(:,1,1:128);    
      gesture_data(:,i)=fft1d_MTI (:,1); 
      gesture_data2(:,(i-1)*128+1:i*128)=Rangedata_odd (:,1,1:128);   
      gesture_data3(:,(i-1)*127+1:i*127)=fft1d_MTI (:,1:127);  
      gesture_data4(:,(i-1)*126+1:i*126)=fft1d_MTI2 (:,1:126);  
   
      
      fft1d_MTI_sum(:,i) = sum(fft1d_MTI,2);
      fft1d_MTI_sum2(:,i) = sum(fft1d_MTI2,2);

    % Doppler FFT
    Dopplerdata_odd = fft_doppler(Rangedata_odd, fft_Vel, 1);
    Dopplerdata_even = fft_doppler(Rangedata_even, fft_Vel, 1);
    Dopdata_sum = squeeze(sum(abs(Dopplerdata_odd), 2));%Nr*Nd
    for kk=1:512
        fft2d_MTI(kk,:) = fftshift(fft(fft1d_MTI(kk,:),fft_Vel));
    end
    fft2d_MTI_data(:,:,i) = fft2d_MTI;
    

   Rangedata_merge = [Rangedata_odd, Rangedata_even];
   for kk=1:8
       middle_fft = squeeze(Rangedata_merge(:,kk,:));
       avg_middle = sum(middle_fft,2)/loop;%/Doppler_Number
        for chirp=1:loop
            FFT1d(:,chirp) = middle_fft(:,chirp)-avg_middle;
        end
        Rangedata_merge_MTI(:,kk,:) = FFT1d;
   end
   searchAngleRange=60;
   agl_grid2=-60:60;
   range_profile = permute(Rangedata_merge_MTI,[3,1,2]); %frame_count*Nr*TxRx
   [~,azimuSpectrogram,~] = IWR1642ODS_DOA(range_profile,2,128,searchAngleRange);
   azimuSpectrogram(:,17:end)=0;
   
   [~,angle_index(i)]=max(max(azimuSpectrogram.'));
   Range_angle_data(:,:,i) = (azimuSpectrogram).';
    
end

figure
plot(agl_grid2(angle_index))
ylim([-60 60])
% axis off; % 隐藏坐标轴
savePath_angle = sprintf('E:\\手势数据2.0\\推\\角度\\image%d.png',q); 
exportgraphics(gcf, savePath_angle, 'BackgroundColor', 'white', 'ContentType', 'image', 'Resolution', 300);
close(gcf);

figure
imagesc(1:size(gesture_data3,2),rng_grid,abs(gesture_data3))
set(gca,'YDir','normal');
xlabel('帧数')
ylabel('距离（m）')
% colormap(jet)
ylim([0 4])
axis off; 

savePath_range = sprintf('E:\\手势数据2.0\\推\\距离-时间\\image%d.png',q); 
exportgraphics(gcf, savePath_range, 'BackgroundColor', 'white', 'ContentType', 'image', 'Resolution', 300);
close(gcf);
 
B=gesture_data3(4:20,:);
A=max(B);
% STFT 
nfft=256;
window=hamming(128);
overlap=127;
[s_breath]=fftshift(spectrogram(A,window,overlap,nfft,20),1);

figure
imagesc(1:size(s_breath,2),vel_grid,abs(s_breath))
set(gca,'YDir','normal');
colormap(jet) 
xlabel('时间')
ylabel('频率')
axis off; % 隐藏坐标轴
savePath_micro = sprintf('E:\\手势数据2.0\\推\\微多普勒\\image%d.png',q); 
exportgraphics(gcf, savePath_micro, 'BackgroundColor', 'white', 'ContentType', 'image', 'Resolution', 300);
close(gcf);

end

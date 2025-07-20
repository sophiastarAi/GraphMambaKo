%% Generate Koopman PM2.5 Modes

function [Psi]=RunDraweival(data,mode1,mode2,month)
%% Load Data
clc; close all;
disp('Loading Data Set...')
tic
if strcmp(data,'Day_mean')
%Data=dlmread('2018-01.txt');
Data=dlmread('month\2020-01-48.txt');
delay=24; dtype='Mean'; delt=1; delx=1;
hwy='day'; hwylength=731; xpath='x121.txt'; ypath='y121.txt';

elseif strcmp(data,'Monthly_2018') 
    Data1 = dlmread(strcat('month\',num2str(month),'-01.txt')); 
    Data2 = dlmread(strcat('month\',num2str(month),'-02.txt')); 
    Data3 = dlmread(strcat('month\',num2str(month),'-03.txt'));
    Data4 = dlmread(strcat('month\',num2str(month),'-04.txt'));
    Data5 = dlmread(strcat('month\',num2str(month),'-05.txt'));
    Data6 = dlmread(strcat('month\',num2str(month),'-06.txt'));
    Data7 = dlmread(strcat('month\',num2str(month),'-07.txt'));
    Data8 = dlmread(strcat('month\',num2str(month),'-08.txt'));
    Data9 = dlmread(strcat('month\',num2str(month),'-09.txt'));
    Data10 = dlmread(strcat('month\',num2str(month),'-10.txt'));
    Data11 = dlmread(strcat('month\',num2str(month),'-11.txt'));
    Data12 = dlmread(strcat('month\',num2str(month),'-12.txt'));
    Data000 = [Data1,Data2,Data3,Data4,Data5,Data6,Data7,Data8,Data9,Data10,Data11,Data12];
    Data = transpose(Data000)
    Data = Data000
    delay=24; dtype='hour'; delt=1; hwy='2019hour'; hwylength=8928; xpath='x121.txt'; ypath='y121.txt'; 
end
toc
%% Compute KMD and Sort Modes
disp('Computing KMD via Hankel-DMD...')
tic  %start timing
Avg=mean(Data,2);
[eigval,Modes1,bo] = H_DMD(Data-repmat(Avg,1,size(Data,2)),delay);   
toc
disp('Sorting Modes...')
tic

%
aa=real(diag(eigval));  
bb=imag(diag(eigval));   
% xlabel('Re(\lambda_i)'); % 
% ylabel('Im(\lambda_i)'); % 
%legend({'aa','bb'},'Location','northeast','NumColumns',10)
cc = real(log(diag(eigval)))   %logarithm
dd = imag(log(diag(eigval)))   %logarithm
omega=log(diag(eigval))./delt;   % Compute Cont. Time Eigenvalues
Freal=imag(omega)./(2*pi);    % Compute Frequency
[T,Im]=sort((1./Freal),'descend');    % Sort Frequencies
omega=omega(Im); Modes1=Modes1(:,Im); bo=bo(Im);    % Sort Modes



num_points = size(eigval, 1);  


lambda = diag(eigval);  
real_lambda = real(lambda);
imag_lambda = imag(lambda);

log_lambda = log(lambda);
real_log = real(log_lambda);
imag_log = imag(log_lambda);

omega = log_lambda ./ delt;     
growth_rates = real(omega);      

stable_threshold = -0.01;       
unstable_threshold = 0.01;       


idx_stable = growth_rates < stable_threshold;
idx_neutral = (growth_rates >= stable_threshold) & (growth_rates <= unstable_threshold);
idx_unstable = growth_rates > unstable_threshold;



omega = log_lambda ./ delt;              
frequencies = imag(omega)./(2*pi);        
periods = abs(1./frequencies);           
periods_days = periods/(24);        

amplitudes = abs(bo); 


[T_sorted, idx_sort] = sort(periods_days, 'descend');
amplitudes_sorted = amplitudes(idx_sort);
growth_rates_sorted = growth_rates(idx_sort);


%subplot(2,2,3)
%scatter(T_sorted, amplitudes_sorted, 40, [0.9 0.6 0], 'filled');
%xlabel('Period (day)');
%ylabel('Amplitude');
%title('(c) Mode Periods');
%grid on;
%set(gca, 'XScale', 'log'); 


subplot(1,1,1)
scatter(amplitudes_sorted, growth_rates_sorted, 40, [0.9 0.6 0], 'filled');
xlabel('Amplitude','FontSize', 16);
ylabel('Growth Rate (Re(s))','FontSize', 16);
title('(c) Mode Stability','FontSize', 16);
grid on;

hold on;
plot(xlim(), [stable_threshold, stable_threshold], 'r--', 'LineWidth', 1.5);
%plot(xlim(), [0, 0], 'g--', 'LineWidth', 1.5); % Neutral line at zero
text(0.8*max(amplitudes_sorted), stable_threshold-0.009, 'Stable Boundary', 'Color', 'r','FontSize', 14);
%text(0.8*max(amplitudes_sorted), 0.03, 'Neutral Line', 'Color', 'g');
toc
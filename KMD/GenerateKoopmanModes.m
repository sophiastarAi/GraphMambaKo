%% Generate Koopman PM2.5 Modes
% Dynamic Mode Decomposition of PM2.5

function [Psi]=GenerateKoopmanModes(data,~,~,month)
%function [Psi]=GenerateKoopmanModes(data,mode1,mode2,month)
%% Load Data
clc; close all;
disp('Loading Data Set...')
tic
if strcmp(data,'Day_mean')
Data=dlmread('2020-01-48.txt');
%Data=dlmread('renji_miaosuan_movestate.txt');
delay=1; dtype='Mean'; delt=1; delx=1;
hwy='day'; hwylength=731; xpath='x121.txt'; ypath='y121.txt';

elseif strcmp(data,'Monthly_2019') 
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
    delay=400; dtype='hour'; delt=1; hwy='2019hour'; hwylength=8928; xpath='x121.txt'; ypath='y121.txt'; 
end
toc
%% Compute KMD and Sort Modes
disp('Computing KMD via Hankel-DMD...')
tic  %start timing
Avg=mean(Data,2);% Compute and Store Time Average
[eigval,Modes1,bo] = H_DMD(Data-repmat(Avg,1,size(Data,2)),delay);   %
toc
disp('Sorting Modes...')
tic

%
scatter(real(diag(eigval)),imag(diag(eigval)))   %scatter 
aa=real(diag(eigval));   %%eigenvalues of real  
bb=imag(diag(eigval));   %imaginary of eigenvalue
xlabel('Re(\lambda_i)'); 
ylabel('Im(\lambda_i)'); 
%legend({'aa','bb'},'Location','northeast','NumColumns',10)
cc = real(log(diag(eigval)))   %logarithm
dd = imag(log(diag(eigval)))   %logarithm
omega=log(diag(eigval))./delt;   % Compute Cont. Time Eigenvalues
Freal=imag(omega)./(2*pi);    % Compute Frequency
[T,Im]=sort((1./Freal),'descend');    % Sort Frequencies
omega=omega(Im); Modes1=Modes1(:,Im); bo=bo(Im);    % Sort Modes
toc

%% ======= MODIFIED START ======= %%
% 只保留前25个长周期模态
nmodes = min(25, length(T));  % 取前25或更少
T = T(1:nmodes);
%T = T / 24;
omega = omega(1:nmodes);
Modes1 = Modes1(:, 1:nmodes);
bo = bo(1:nmodes);

[nbx,nbt]=size(Data); % Get Data Size 
fprintf('nbt 的值是：%d\n', nbt); 
time=(0:nbt-1)*delt;% Specify Time Interval delt 


modeIndices = zeros(nmodes, 1);
periods = zeros(nmodes, 1);
amplitudes = zeros(nmodes, 1);
decayRates = zeros(nmodes, 1);

mode1 = 1;
mode2 = nmodes;
%% ======= MODIFIED END ======== %%


%% Compute and Plot Modes 
disp('Computing and Plotting Modes...')
tic


Psi=zeros(nbx,nbt,mode2-mode1+1);  %
res=[]  %
for i=mode1:mode2 % Loop Through all Modes to Plot.
    psi=zeros(1,nbt);% Preallocate Time Evolution of Mode.
    omeganow=omega(i);% Get Current Eigenvalue.
    bnow=bo(i);% Get Current Amplitude Coefficient.
    parfor t=1:length(time) 
        psi(:,t)=exp(omeganow*time(t))*bnow; % Evolve for Time Length.
    end
    psi=Modes1(1:nbx,i)*psi;    % Compute Mode.
    Psi(:,:,i)=psi;    % Store & Output Modes
    m=abs(psi) %
    mag=mean(m(:)) %average amplitude
    res=[res mag]  %save

% ======= MODIFIED ======= %
    modeIndices(i) = i;
    periods(i) = T(i);
    amplitudes(i) = mag;
    decayRates(i) = real(omega(i));
    % ======================================== %
    % ======= MODIFIED:====== %
    h=figure;  
    warning('off','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');	
    jFrame = get(h,'JavaFrame');	
    pause(0.3);					
    set(jFrame,'Maximized',1);	
    pause(0.5);					
    warning('on','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');
    
    if strcmp(hwy,'day')
        [X,Y]=meshgrid(time./1,linspace(0,150,nbx)); 
    elseif strcmp(hwy,'2019hour') 
        [X,Y]=meshgrid(time./24,linspace(0,150,nbx));
    end
    

% -------------------------------------------------------------------------%
% -------------------------------------------------------------------------%
    FONTSIZE = 35;  %字体
    TICKSIZE = 28;  %刻度


    s1=surfc(X,Y,real(psi));% Generate Surface Plot 
    set(s1,'LineStyle','none')% No Lines  

%'position', [0.1, 0.15, 0.60, 0.78] 
%'TickLabelInterpreter', 'latex' 
%'linewidth', 2.5 
%'FontSize', 30 
    set(gca,'position',[0.12,0.15,0.70,0.78],'TickLabelInterpreter','latex','linewidth',2.5,'FontSize',30)
%title(strcat('Mode #',num2str(i)),... 
%                     'Interpreter','Latex','FontSize',30)
%title({['Mode #' num2str(i)],[ 'Period=' num2str(T(i),'%.2f')...

    info_str = {sprintf('Mode %d', i),...
                sprintf('Period: %.2f hours', T(i)),...
                sprintf('Decay: %.4f', real(omega(i))),...
                sprintf('Amplitude: %.4f', mag)};
            
    annotation('textbox', [0.6 0.75 0.2 0.18],...
        'String', info_str,...
        'FitBoxToText','on',...
        'FontSize',15,...
        'BackgroundColor','w',...
        'EdgeColor','k');
        
    %title(['Mode #' num2str(i) newline 'Period=' num2str(T(i),'%.2f') ' frames    Growth/Decay Rate=' num2str(real(omega(i)),'%.4f')],'FontSize',15);
    xlabel('Time','Interpreter','tex','FontSize',25,'rotation',13); 
    h=colorbar;
    yh=ylabel('Monitoring station (μg/m^{3})','rotation',-20,'FontSize',25);

    yh = get(gca, 'YLabel');
    set(yh, 'Units', 'normalized');
    set(yh, 'Position', [0.35, -0.1, -0.3], ... % 
             'HorizontalAlignment', 'right', ...
             'VerticalAlignment', 'middle');

    %pos = get(yh, 'Position');
    

    %set(yh, 'Position', [pos(1)*0.65, pos(2)*1.5, pos(3)]);  % 左上微调

    if strcmp(dtype,'Mean')
        set(get(h,'title'),'FontSize',30);
    elseif strcmp(dtype,'Mean')
        set(get(h,'title'),'string', {'μg/m^{3} per hour'});
    end
end %End Modes to Plot Loop

%% ======= MODIFIED: Excel ======= %
T_table = table(modeIndices, periods, amplitudes, decayRates, ...
    'VariableNames', {'ModeIndex', 'Period', 'Amplitude', 'DecayRate'});
writetable(T_table, 'ModeSummary.xlsx');
disp('Saved mode summary to ModeSummary.xlsx');
%% =========================================== %

csvwrite('period.csv',T)
csvwrite('psi.csv',res) 

toc
disp('All Done')

end %End function

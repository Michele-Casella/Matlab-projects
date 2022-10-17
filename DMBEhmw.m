%% 1


%%(a)

%ITALY
tableIT = readtable('IT');
years = table2array(tableIT(:,1));
gdpIT = table2array(tableIT(:,2));

changeIT = diff(gdpIT); %GDP variation for ITALY
rocIT = changeIT./gdpIT(1:end-1)*100;

%GERMANY
tableGER = readtable('GER');
years = table2array(tableGER(:,1));
gdpGER = table2array(tableGER(:,2));

changeGER = diff(gdpGER);
rocGER = changeGER./gdpGER(1:end-1)*100;

%SPAIN
tableES = readtable('ES');
years = table2array(tableES(:,1));
gdpES = table2array(tableES(:,2));
changeES = diff(gdpES);
rocES = changeES./gdpES(1:end-1)*100;

%FRANCE
tableFR = readtable('FR');
years = table2array(tableFR(:,1));
gdpFR = table2array(tableFR(:,2));

changeFR = diff(gdpFR);
rocFR = changeFR./gdpFR(1:end-1)*100;

%to compare gdp change for each country
figure
subplot(2,2,1)
plot(years(1:end-1),changeIT)
title('gdp change (millions of euros)- IT')
xlabel ('year')
ylabel ('change')
subplot(2,2,2)
plot(years(1:end-1),changeGER)
title('gdp change (millions of euros)- GER')
xlabel('year')
ylabel('change')
subplot(2,2,3)
plot(years(1:end-1),10*changeES)
title('gdp change (millions of euros)- ES')
xlabel ('year')
ylabel ('change')
subplot(2,2,4)
plot(years(1:end-1),changeFR)
title('gdp change (millions of euros)- FR')
xlabel ('year')
ylabel ('change')

%to compare gpd change rate for each country
figure
subplot(2,2,1)
plot(years(1:end-1),rocIT)
title('gdp roc (millions of euros)- IT')
xlabel ('year')
ylabel ('%')
subplot(2,2,2)
plot(years(1:end-1),rocGER)
title('gdp roc (millions of euros)- GER')
xlabel('year')
ylabel('%')
subplot(2,2,3)
plot(years(1:end-1),rocES)  % Multiplication by 10 is simply a normalization to make the two prices comparable
title('gdp roc (millions of euros)- ES') 
xlabel ('year')
ylabel ('%')
subplot(2,2,4)
plot(years(1:end-1),rocFR)
title('gdp roc (millions of euros)- FR')
xlabel ('year')
ylabel ('%')


%% (b)
fprintf('(b)\n\n\n') 

%ITALY

meanIT = mean(changeIT);
meanrocIT = mean(rocIT);

medianIT = median(changeIT);
medianrocIT = median(rocIT);

modeIT = mode(changeIT);
moderocIT = mode(rocIT);

rangeIT = max(changeIT)-min(changeIT);
rangerocIT = max(rocIT)-min(rocIT);

iqIT = quantile(changeIT,0.75)-quantile(changeIT,0.25);
iqrIT = quantile(rocIT,0.75)-quantile(rocIT,0.25);

stdIT = std(changeIT);
stdrocIT = std(rocIT);

varIT = var(changeIT);
varocIT = var(rocIT);

fprintf('The italian gdp change  MEAN is: %f\n', meanIT)
fprintf('The italian gdp change MEDIAN is: %f\n', medianIT)
fprintf('The italian gdp change INTERQUANTILE RANGE is: %f\n', iqIT)
fprintf('The italian gdp change DEVIANCE is: %f\n',stdIT)
fprintf('The italian gdp change VARIANCE id: %f\n', varIT)
fprintf('The italian gdp change rate MEAN is: %f%%\n', meanrocIT)
fprintf('The italian gdp change rate MEDIAN is: %f%%\n', medianrocIT)
fprintf('The italian gdp change rate INTERQUANTILE RANGE is: %f%%\n', iqrIT)
fprintf('The italian gdp change rate DEVIANCE is: %f%%\n',stdrocIT)
fprintf('The italian gdp change rate VARIANCE id: %f%%\n\n', varocIT)

%GERMANY

meanGER = mean(changeGER);
meanrocGER = mean(rocGER);

meadianGER = median(changeGER);
meadianrocGER = median(rocGER);

modeGER = mode(changeGER);
moderocGER = mode(rocGER);

rangeGER = max(changeGER)-min(changeGER);
rangerocGER = max(changeGER)-min(changeGER);

iqGER = quantile(changeGER,0.75)-quantile(changeGER,0.25);
iqrGER = quantile(rocGER,0.75)-quantile(rocGER,0.25);

stdGER = std(changeGER);
stdrocGER = std(rocGER);

varGER = var(changeGER);
varocGER = var(rocGER);

fprintf('The german gdp change MEAN is: %f\n', meanGER)
fprintf('The german gdp change MEDIAN is: %f\n', meadianGER)
fprintf('The german gdp change INTERQUANTILE RANGE is: %f\n', iqGER)
fprintf('The german gdp change DEVIANCE is: %f\n',stdGER)
fprintf('The german gdp change VARIANCE id: %f\n', varGER)
fprintf('The german gdp change rate MEAN is: %f%%\n', meanrocGER)
fprintf('The german gdp change rate MEDIAN is: %f%%\n', meadianrocGER)
fprintf('The german gdp change rate INTERQUANTILE RANGE is: %f%%\n', iqrGER)
fprintf('The german gdp change rate DEVIANCE is: %f%\n',stdrocGER)
fprintf('The german gdp change rate VARIANCE id: %f%\n\n', varocGER)

%SPAIN
meanES = mean(changeES);
meanrocES = mean(rocES);

medianES = median(changeES);
medianrocES = median(rocES);

modeES = mode(changeES);
moderocES = mode(rocES);

rangeES = max(changeES)-min(changeES);
rangerocES = max(rocES)-min(rocES);

iqES = quantile(changeES,0.75)-quantile(changeES,0.25);
iqrES = quantile(rocES,0.75)-quantile(rocES,0.25);

stdES = std(changeES);
stdrocES = std(rocES);

varES = var(changeES);
varocES = var(rocES);

fprintf('The spanish gdp change  MEAN is: %f\n', meanES)
fprintf('The spanish gdp change MEDIAN is: %f\n', medianES)
fprintf('The spanish gdp change INTERQUANTILE RANGE is: %f\n', iqES)
fprintf('The spanish gdp change DEVIANCE is: %f\n',stdES)
fprintf('The spanish gdp change VARIANCE id: %f\n', varES)
fprintf('The spanish gdp change rate MEAN is: %f%%\n', meanrocES)
fprintf('The spanish gdp change rate MEDIAN is: %f%%\n', medianrocES)
fprintf('The spanish gdp change rate INTERQUANTILE RANGE is: %f%%\n', iqrES)
fprintf('The spanish gdp change rate DEVIANCE is: %f%%\n',stdrocES)
fprintf('The spanish gdp change rate VARIANCE id: %f%%\n\n', varocES)


%FRANCE
meanFR = mean(changeFR);
meanrocFR = mean(rocFR);

medianFR = median(changeFR);
medianrocFR = median(rocFR);

modeFR = mode(changeFR);
moderocFR = mode(rocFR);

rangeFR = max(changeFR)-min(changeFR);
rangerocFR = max(rocFR)-min(rocFR);

iqFR = quantile(changeFR,0.75)-quantile(changeFR,0.25);
iqrFR = quantile(rocFR,0.75)-quantile(rocFR,0.25);

stdFR = std(changeFR);
stdrocFR = std(rocFR);

varFR = var(changeFR);
varocFR = var(rocFR);
fprintf('The french gdp change MEAN is: %f\n', meanFR)
fprintf('The french gdp change MEDIAN is: %f\n', medianFR)
fprintf('The french gdp change INTERQUANTILE RANGE is: %f\n', iqFR)
fprintf('The french gdp change DEVIANCE is: %f\n',stdFR)
fprintf('The french gdp change VARIANCE id: %f\n', varFR)
fprintf('The french gdp change rate MEAN is: %f%%\n', meanrocFR)
fprintf('The french gdp change rate MEDIAN is: %f%%\n', medianrocFR)
fprintf('The french gdp change rate INTERQUANTILE RANGE is: %f%%\n', iqrFR)
fprintf('The french gdp change rate DEVIANCE is: %f%%\n',stdrocFR)
fprintf('The french gdp change rate VARIANCE id: %f%%\n\n', varocFR)

% Dispersion comparison among ITALY,GERMANY, SPAIN and FRANCE
figure
boxplot([changeIT, changeGER, changeES, changeFR], 'labels',{'ITALY','GERMANY', 'SPAIN','FRANCE'})
xlabel ('country');
ylabel ('gdp change');
figure
boxplot([rocIT, rocGER, rocES, rocFR], 'labels',{'ITALY','GERMANY', 'SPAIN','FRANCE'})
ylabel ('gdp roc change')
xlabel('country')
% Dependence estimators among ITALY, GERMANY, SPAIN and FRANCE
%the dependence among these countries can be computed by covariance, 
% whose command is cov, over variance product but the fastest way to get the same result is 
% the 'corr' command 

rhoITGER = corr(changeIT, changeGER);                
rhoITFR = corr(changeIT, changeFR);
rhoITES = corr(changeIT, changeES);
rhoGERFR = corr(changeGER, changeFR);
rhoGERES = corr(changeGER, changeES);
rhoFRES = corr(changeFR, changeES);
fprintf('\ncorrelation according to changes\n')
fprintf('The correlation between ITALY and GERMANY is %f \n',rhoITGER);
fprintf('The correlation between ITALY and FRANCE is %f \n',rhoITFR);
fprintf('The correlation between ITALY and SPAIN is %f \n',rhoITES);
fprintf('The correlation between GERMANY and SPAIN is %f \n',rhoGERES);
fprintf('The correlation between GERMANY and FRANCE is %f\n', rhoGERFR);
fprintf('The correlation between FRANCE and SPAIN is %f \n\n',rhoFRES);
fprintf('correlation according to rates of change\n')
rhorocITGER = corr(rocIT, rocGER);                
rhorocITFR = corr(rocIT, rocFR);
rhorocITES = corr(rocIT, rocES);
rhorocGERFR = corr(rocGER, rocFR);
rhorocGERES = corr(rocGER, rocES);
rhorocFRES = corr(rocFR, rocES);
fprintf('\ncorrelation according to changes\n')
fprintf('The correlation between ITALY and GERMANY is %f \n',rhorocITGER);
fprintf('The correlation between ITALY and FRANCE is %f \n',rhorocITFR);
fprintf('The correlation between ITALY and SPAIN is %f \n',rhorocITES);
fprintf('The correlation between GERMANY and SPAIN is %f \n',rhorocGERES);
fprintf('The correlation between GERMANY and FRANCE is %f\n', rhorocGERFR);
fprintf('The correlation between FRANCE and SPAIN is %f \n\n',rhorocFRES);
figure
subplot(2,3,1)
scatter(changeIT, changeGER);
ylabel('ITALY');
xlabel('GERMANY');
subplot(2,3,2)
scatter(changeIT, changeES);
ylabel('ITALY');
xlabel('SPAIN');
subplot(2,3,3)
scatter(changeIT, changeFR);
ylabel('ITALY');
xlabel('FRANCE');
subplot(2,3,4)
scatter(changeGER, changeES);
ylabel('GERMANY');
xlabel('SPAIN');
subplot(2,3,5)
scatter(changeGER, changeFR);
ylabel('GERMANY');
xlabel('FRANCE');
subplot(2,3,6)
scatter(changeES, changeFR);
ylabel('SPAIN');
xlabel('FRANCE');
%% (c)
fprintf('(c)\n\n\n')
alpha1 = 0.10/2;
alpha2 = 0.05/2;
alpha3 = 0.01/2;
t1 = tinv(0.90+alpha1 , 24);
t2 = tinv(0.95+alpha2 , 24);
t3 = tinv(0.99+alpha3 , 24);
fprintf('Quantiles of Student-T distribution are:\nt(0.95,24)=%f\nt(0.975,24)=%f\nt(0.995,24)=%f\n\n\n\n',t1,t2,t3)
%ITALY
tIT = meanrocIT/(stdrocIT/sqrt(25));
pIT = 1-tcdf(tIT, 24);
fprintf('ITALY\n')
% 90 percentage of confidence
uclt = tIT + stdrocIT/sqrt(25)*t1;
lclt = tIT - stdrocIT/sqrt(25)*t1;
fprintf(' C.I. given a 90 percentage of confidence  = [%f, %f]\n',lclt,uclt);
fprintf(' zIT= %f', tIT);
if tIT > t1     
    fprintf('>%f and p-value=%f<%f , H_0 can be rejected given a 90 percentage of confidence', t1,pIT,alpha1);
else tIT < t1;
    fprintf('<%f and p-value=%f>%f, H_0 cannot be rejected given a 90 percentage of confidence ',t1,pIT,alpha1);
end
% 95 percentage of confidence
uclt = tIT + stdrocIT/sqrt(25)*t2;
lclt = tIT - stdrocIT/sqrt(25)*t2;
fprintf('\n C.I. given a 95 percentage of confidence  = [%f, %f]',lclt,uclt);
fprintf('\n zIT= %f', tIT);
if tIT > t2     
    fprintf('>%f and p-value=%f<%f , H_0 can be rejected given a 95 percentage of confidence', t2,pIT,alpha2);
else tIT < t2;
    fprintf('<%f and p-value=%f>%f, H_0 cannot be rejected given a 95 percentage of confidence ',t2,pIT,alpha2);
end
% 99 percentage of confidence
uclt = tIT + stdrocIT/sqrt(25)*t3;
lclt = tIT - stdrocIT/sqrt(25)*t3;
fprintf('\n C.I. given a 99 percentage of confidence  = [%f, %f]',lclt,uclt);
fprintf('\n zIT= %f', tIT);
if tIT > t3
    fprintf('>%f and p-value=%f<%f , H_0 can be rejected given a 99 percentage of confidence', t3,pIT,alpha3);
else tIT < t3;
    fprintf('<%f and p-value=%f>%f, H_0 cannot be rejected given a 99 percentage of confidence ',t3,pIT,alpha3);
end

%GERMANY
tGER = meanrocGER/(stdrocGER/sqrt(25));
pGER = 1-tcdf(tGER,24);
fprintf('\n\n GERMANY\n')
% 90 percentage of confidence
uclt = tGER + stdrocGER/sqrt(25)*t1;
lclt = tGER - stdrocGER/sqrt(25)*t1;
fprintf(' C.I. given a 90 percentage of confidence  = [%f, %f]\n',lclt,uclt);
fprintf(' zGER= %f', tGER);
if tGER > t1     
    fprintf('>%f and p-value=%f<%f , H_0 can be rejected given a 90 percentage of confidence', t1,pGER,alpha1);
else tGER < t1;
    fprintf('<%f and p-value=%f>%f, H_0 cannot be rejected given a 90 percentage of confidence ',t1,pGER,alpha1);
end
% 95 percentage of confidence
uclt = tGER + stdrocGER/sqrt(25)*t2;
lclt = tGER - stdrocGER/sqrt(25)*t2;
fprintf('\n C.I. given a 95 percentage of confidence  = [%f, %f]',lclt,uclt);
fprintf('\n zGER= %f', tGER);
if tGER > t2     
    fprintf('>%f and p-value=%f<%f , H_0 can be rejected given a 95 percentage of confidence', t2,pGER,alpha2);
else tGER < t3;
    fprintf('<%f and p-value=%f>%f, H_0 cannot be rejected given a 95 percentage of confidence ',t2,pGER,alpha2);
end
% 99 percentage of confidence
uclt = tGER + stdrocGER/sqrt(25)*t3;
lclt = tGER - stdrocGER/sqrt(25)*t3;
fprintf('\n C.I. given a 99 percentage of confidence  = [%f, %f]',lclt,uclt);
fprintf('\n zGER= %f', tGER);
if tGER > t3     
    fprintf('>%f and p-value=%f<%f , H_0 can be rejected given a 99 percentage of confidence', t3,pGER,alpha3);
else tGER < t3;
    fprintf('<%f and p-value=%f>%f, H_0 cannot be rejected given a 99 percentage of confidence ',t3,pGER,alpha3);
end

%SPAIN
tES = meanrocES/(stdrocES/sqrt(25));
pES = 1-tcdf(tES,24);
% 90 percentage of confidence
uclt = tES + stdrocES/sqrt(25)*t1;
lclt = tES - stdrocES/sqrt(25)*t1;
fprintf('\n\n SPAIN')
fprintf('\n C.I. given a 90 percentage of confidence  = [%f, %f]',lclt,uclt);
fprintf('\n zES= %f', tES);
if tES > t1
    fprintf('>%f and p-value=%f<%f, H_0 can be rejected given a 90 percentage of confidence', t1,pES,alpha1);
else tES < t1;
    fprintf('<%f and p-value=%f>%f, H_0 cannot be rejected given a 90 percentage of confidence ',t1,pES,alpha1);
end
% 95 percentage of confidence
uclt = tES + stdrocES/sqrt(25)*t2;
lclt = tES - stdrocES/sqrt(25)*t2;
fprintf('\n C.I. given a 95 percentage of confidence  = [%f, %f]',lclt,uclt);
fprintf('\n zES= %f', tES);
if tES > t2
    fprintf('>%f and p-value=%f<%f, H_0 can be rejected given a 95 percentage of confidence', t2,pES,alpha2);
else tES < t2;
    fprintf('<%f and p-value=%f>%f, H_0 cannot be rejected given a 95 percentage of confidence ',t2,pES,alpha2);
end
% 99 percentage of confidence
uclt = tES + stdrocES/sqrt(25)*t3;
lclt = tES - stdrocES/sqrt(25)*t3;
fprintf('\n C.I. given a 99 percentage of confidence  = [%f, %f]',lclt,uclt);
fprintf('\n zES= %f', tES);
if tES > t3
    fprintf('>%f and p-value=%f<%f, H_0 can be rejected given a 99 percentage of confidence', t3,pES,alpha3);
else tES < t3;
    fprintf('<%f and p-value=%f>%f, H_0 cannot be rejected given a 99 percentage of confidence ',t3,pES,alpha3);
end
%FRANCE
tFR = meanrocFR/(stdrocFR/sqrt(25));
pFR = 1-tcdf(tFR,24);
% 90 percentage of confidence
uclt = tFR + stdrocFR/sqrt(25)*t1;
lclt = tFR - stdrocFR/sqrt(25)*t1;
fprintf('\n\n FRANCE')
fprintf('\n C.I. given a 90 percentage of confidence  = [%f, %f]',lclt,uclt);
fprintf('\n zFR= %f', tFR);
if tFR > t1
    fprintf('>%f and p-value=%f<%f, H_0 can be rejected given a 90 percentage of confidence', t1,pFR,alpha1);
else tFR < t1;
    fprintf('<%f and p-value=%f>%f, H_0 cannot be rejected given a 90 percentage of confidence ',t1,pFR,alpha1);
end
% 95 percentage of confidence
uclt = tFR + stdrocFR/sqrt(25)*t2;
lclt = tFR - stdrocFR/sqrt(25)*t2;
fprintf('\n C.I. given a 95 percentage of confidence  = [%f, %f]',lclt,uclt);
fprintf('\n zFR= %f', tFR);
if tFR > t2
    fprintf('>%f and p-value=%f<%f, H_0 can be rejected given a 95 percentage of confidence', t2,pFR,alpha2);
else tFR < t2;
    fprintf('<%f and p-value=%f>%f, H_0 cannot be rejected given a 95 percentage of confidence ',t2,pFR,alpha2);
end
% 99 percentage of confidence
uclt = tFR + stdrocFR/sqrt(25)*t3;
lclt = tFR - stdrocFR/sqrt(25)*t3;
fprintf('\n C.I. given a 99 percentage of confidence  = [%f, %f]',lclt,uclt);
fprintf('\n zFR= %f', tFR);
if tFR > t3
     fprintf('>%f and p-value=%f<%f, H_0 can be rejected given a 99 percentage of confidence', t3,pFR,alpha3);
else tFR < t3;
    fprintf('<%f and p-value=%f>%f, H_0 cannot be rejected given a 99 percentage of confidence ',t3,pFR,alpha3);
end

%%


%% 2


x = [1, 2, 3, 4, 5, 6];
p = [2/15, 1/3, 2/15, 2/15, 2/15, 2/15];
%% (a)
popMean = sum(x.*p);
popVar = sum(((x-popMean).^2).*p);

%% (b) (c)
yax = [0];
populationmean = [0];
xax = [0];
popmean = [0];
partition = [0, cumsum(p)];

for i = 1:250
    u = rand;
    if u>partition(1) && u<partition(2)
        dice = 1;
    elseif u>partition(2) && u<partition(3)
        dice = 2;
    elseif u>partition(3) && u<partition(4)
        dice = 3;
    elseif u>partition(4) && u<partition(5)
        dice = 4;
    elseif u>partition(5) && u<partition(6)
        dice = 5;
    elseif u>partition(6) && u<partition(7)
        dice = 6;
    end
    
yax(i) = dice;
populationmean(i) = mean(yax);
xax(i) = i;    
popmean(i) = 3.2;
plot(xax, populationmean, xax, popmean)
ylabel('roll of dice result')
ylim([0,6])
xlim([1,250])
legend('Sample mean','Population mean')
titlePic = sprintf('Sample size = %d',i); %tite changes while sapling size increases
title(titlePic);
drawnow;
end 

%%


%% 3



%% data
P = 1/6; %probabily for each dice number


%% (a) compute population mean and variance
x = [1, 2, 3, 4, 5, 6];
p = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];

popMean = sum(x.*p);
popVar = sum(((x-popMean).^2).*p);
%% (b) one roll of fair dice simulation

dice = 6*rand;    % Gives a uniform distribution in [0,6]
dice = ceil(dice); % Gives a discrete distribution with population space {1,2,3,4,5,6}
    
fprintf('The roll result is %d \n',dice)
    

%% (c) verify pivotal statistic PDF approaches to Gaussian's one

%Z = (sampleMean - popMean)/(sampleSD/(n^0.5)) %pivotal statistic formula

N = 250;               
n = 700; 

% Compute the PDF of the standard normal
xrange = -5:0.01:5;
pdf = 1/(sqrt(2*pi))*exp(-(xrange.^2)/2);
    
r = 6*rand(N,n);
r = ceil(r);
for i = 1:n
    Xi = mean(r(:,1:n),2);
    Zi = (Xi-popMean)/((popVar/i)^0.5);
    
    histogram(Zi,(-5:0.25:5), 'Normalization','pdf');   
%set the bins for better stability. Otherwise use ksdensity
hold on
plot(xrange,pdf,'LineWidth',2);
hold off
ylim([0,1])     % To fix the range of the vertical axis
xlim([-5,5])      % To fix the range of the horizontal axis
legend('Frequency distribution of $Z_n$','PDF of standard Normal','interpreter','latex')
titlePic = sprintf('Sample size = %d',i);
title(titlePic);
drawnow;
end
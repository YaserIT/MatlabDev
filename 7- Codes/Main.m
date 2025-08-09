clear
clc
close all
load total
%%
ND=1000;
K=randperm(length(D),ND);
X=D(K,:);
Y=C(K,:);
%% ABC based Feature selection
ABCopts.lb        = 0;
ABCopts.ub        = 1;
ABCopts.thres     = 0.5;
ABCopts.max_limit = 5;
ABCopts.N        = 50;
ABCopts.T        = 50;
ABCopts.k        = 5;
% Ratio of validation data
ho            = 0.2;
HO            = cvpartition(C,'HoldOut',ho);
ABCopts.Model    = HO;
ABC=jArtificialBeeColony(X,Y,ABCopts);
disp(['Selecteg features by ABC : ', num2str(ABC.sf)])

figure
plot(ABC.c,'*-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',10)
hold on
title('Convergence curve of MOABC')
xlabel('Number of Iteration')
ylabel('Fitness value')
ylim([0,1])
%% Genetic based feature
GAopts.CR = 0.8;    % crossover rate
GAopts.MR = 0.01;   % mutation rate
GAopts.N        = 50;
GAopts.T        = 50;
GAopts.k        = 5;
% Ratio of validation data
ho            = 0.2;
HO            = cvpartition(C,'HoldOut',ho);
GAopts.Model    = HO;
GA=jGeneticAlgorithm(X,Y,GAopts);
disp(['Selecteg features by GA : ', num2str(GA.sf)])

figure
plot(GA.c,'*-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',10)
hold on
title('Convergence curve of MOGA')
xlabel('Number of Iteration')
ylabel('Fitness value')
ylim([0,1])
%% PSO based Feature selection
PSOopts.lb    = 0;
PSOopts.ub    = 1;
PSOopts.thres = 0.5;
PSOopts.c1    = 2;              % cognitive factor
PSOopts.c2    = 2;              % social factor
PSOopts.w     = 0.9;            % inertia weight
PSOopts.Vmax  = (PSOopts.ub - PSOopts.lb) / 2;  % Maximum velocity
PSOopts.N        = 50;
PSOopts.T        = 50;
PSOopts.k        = 5;
% Ratio of validation data
ho            = 0.2;
HO            = cvpartition(C,'HoldOut',ho);
PSOopts.Model    = HO;
PSO=jParticleSwarmOptimization(X,Y,PSOopts);
disp(['Selecteg features by PSO : ', num2str(PSO.sf)])

figure
plot(PSO.c,'*-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',10)
hold on
title('Convergence curve of MOPSO')
xlabel('Number of Iteration')
ylabel('Fitness value')
ylim([0,1])
%% GWO based Feature selection
GWOopts.lb    = 0;
GWOopts.ub    = 1;
GWOopts.thres = 0.5;
GWOopts.N        = 50;
GWOopts.T        = 50;
GWOopts.k        = 5;
% Ratio of validation data
ho            = 0.2;
HO            = cvpartition(C,'HoldOut',ho);
GWOopts.Model    = HO;
GWO=jGreyWolfOptimizer(X,Y,GWOopts);
disp(['Selecteg features by GWO : ', num2str(GWO.sf)])

figure
plot(GWO.c,'*-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',10)
hold on
title('Convergence curve of MOGWO')
xlabel('Number of Iteration')
ylabel('Fitness value')
ylim([0,1])
%% result
ACC=[GWO.ACC; GA.ACC; PSO.ACC; ABC.ACC];
figure
bar(ACC)
title('Comparison of metaheuristic based feature selection methods')
xticklabels({'MOGWO','MOGA','MOPSO','MOABC'})
% legend('MOGWO','MOGA','MOPSO','MOABC','Location','northoutside','Orientation','horizontal')
%%
NT=0.3;%test number
TK=randperm(length(D),round(NT*length(D)));
TEST=D(TK,:);
TClass=C(TK,1);
%% ABC based test
ABCTrain=ABC.ff;
ABCTEST=TEST(:,ABC.sf);
%% GA based test
GATrain =GA.ff;
GATEST=TEST(:,GA.sf);
%% PSO based test
PSOTrain=PSO.ff;
PSOTEST=TEST(:,PSO.sf);
%% GWO based test
GWOTrain=GWO.ff;
GWOTEST=TEST(:,GWO.sf);
%% ABCKNN
ABCKNN=KNN(ABCTrain,Y,ABCTEST);
[ABCKNNACC,ABCKNNRecall,ABCKNNPrecision,ABCKNNFmeasure]=EvaluateCLF(ABCKNN,TClass);
%% ABCNN
ABCNN=NN(ABCTrain,Y,ABCTEST);
[ABCNNACC,ABCNNRecall,ABCNNPrecision,ABCNNFmeasure]=EvaluateCLF(ABCNN,TClass);
%% ABCSVM
ABCSVM=SVM(ABCTrain,Y,ABCTEST);
[ABCSVMACC,ABCSVMRecall,ABCSVMPrecision,ABCSVMFmeasure]=EvaluateCLF(ABCSVM,TClass);
%% ABCNB
ABCNB=NB(ABCTrain,Y,ABCTEST);
[ABCNBACC,ABCNBRecall,ABCNBPrecision,ABCNBFmeasure]=EvaluateCLF(ABCNB,TClass);
%% GWOKNN
GWOKNN=KNN(GWOTrain,Y,GWOTEST);
[GWOKNNACC,GWOKNNRecall,GWOKNNPrecision,GWOKNNFmeasure]=EvaluateCLF(GWOKNN,TClass);
%% GWONN
GWONN=NN(GWOTrain,Y,GWOTEST);
[GWONNACC,GWONNRecall,GWONNPrecision,GWONNFmeasure]=EvaluateCLF(GWONN,TClass);
%% GWOSVM
GWOSVM=SVM(GWOTrain,Y,GWOTEST);
[GWOSVMACC,GWOSVMRecall,GWOSVMPrecision,GWOSVMFmeasure]=EvaluateCLF(GWOSVM,TClass);
%% GWONB
GWONB=NB(GWOTrain,Y,GWOTEST);
[GWONBACC,GWONBRecall,GWONBPrecision,GWONBFmeasure]=EvaluateCLF(GWONB,TClass);
%% GAKNN
GAKNN=KNN(GATrain,Y,GATEST);
[GAKNNACC,GAKNNRecall,GAKNNPrecision,GAKNNFmeasure]=EvaluateCLF(GAKNN,TClass);
%% GANN
GANN=NN(GATrain,Y,GATEST);
[GANNACC,GANNRecall,GANNPrecision,GANNFmeasure]=EvaluateCLF(GANN,TClass);
%% GASVM
GASVM=SVM(GATrain,Y,GATEST);
[GASVMACC,GASVMRecall,GASVMPrecision,GASVMFmeasure]=EvaluateCLF(GASVM,TClass);
%% GANB
GANB=NB(GATrain,Y,GATEST);
[GANBACC,GANBRecall,GANBPrecision,GANBFmeasure]=EvaluateCLF(GANB,TClass);
%% PSOKNN
PSOKNN=KNN(PSOTrain,Y,PSOTEST);
[PSOKNNACC,PSOKNNRecall,PSOKNNPrecision,PSOKNNFmeasure]=EvaluateCLF(PSOKNN,TClass);
%% PSONN
PSONN=NN(PSOTrain,Y,PSOTEST);
[PSONNACC,PSONNRecall,PSONNPrecision,PSONNFmeasure]=EvaluateCLF(PSONN,TClass);
%% PSOSVM
PSOSVM=SVM(PSOTrain,Y,PSOTEST);
[PSOSVMACC,PSOSVMRecall,PSOSVMPrecision,PSOSVMFmeasure]=EvaluateCLF(PSOSVM,TClass);
%% PSONB
PSONB=NB(PSOTrain,Y,PSOTEST);
[PSONBACC,PSONBRecall,PSONBPrecision,PSONBFmeasure]=EvaluateCLF(PSONB,TClass);

%% Test Evaluating
ST=length(GWOKNNACC)/10;
e=1:ST:length(GWOKNNACC);
figure
plot(e,sort(GWOKNNACC(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GAKNNACC(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSOKNNACC(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCKNNACC(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Accuracy comparison in test prediction by feature selection methods')
legend('GWO-KNN','GA-KNN','PSO-KNN','ABC-KNN','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Accurace rate')
axis tight
%%
figure
plot(e,sort(GWOKNNRecall(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GAKNNRecall(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSOKNNRecall(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCKNNRecall(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Recall comparison in test prediction by feature selection methods')
legend('GWO-KNN','GA-KNN','PSO-KNN','ABC-KNN','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Recall rate')
axis tight
%%
figure
plot(e,sort(GWOKNNPrecision(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GAKNNPrecision(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSOKNNPrecision(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCKNNPrecision(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Precision comparison in test prediction by feature selection methods')
legend('GWO-KNN','GA-KNN','PSO-KNN','ABC-KNN','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Precision rate')
axis tight
%%
figure
plot(e,sort(GWOKNNFmeasure(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GAKNNFmeasure(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSOKNNFmeasure(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCKNNFmeasure(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Fmeasure comparison in test prediction by feature selection methods')
legend('GWO-KNN','GA-KNN','PSO-KNN','ABC-KNN','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Fmeasure rate')
axis tight
%% NN
ST=length(GWONNACC)/10;
e=1:ST:length(GWONNACC);
figure
plot(e,sort(GWONNACC(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GANNACC(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSONNACC(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCNNACC(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Accuracy comparison in test prediction by feature selection methods')
legend('GWO-NN','GA-NN','PSO-NN','ABC-NN','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Accurace rate')
axis tight
%%
figure
plot(e,sort(GWONNRecall(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GANNRecall(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSONNRecall(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCNNRecall(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Recall comparison in test prediction by feature selection methods')
legend('GWO-NN','GA-NN','PSO-NN','ABC-NN','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Recall rate')
axis tight
%%
figure
plot(e,sort(GWONNPrecision(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GANNPrecision(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSONNPrecision(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCNNPrecision(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Precision comparison in test prediction by feature selection methods')
legend('GWO-NN','GA-NN','PSO-NN','ABC-NN','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Precision rate')
axis tight
%%
figure
plot(e,sort(GWONNFmeasure(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GANNFmeasure(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSONNFmeasure(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCNNFmeasure(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Fmeasure comparison in test prediction by feature selection methods')
legend('GWO-NN','GA-NN','PSO-NN','ABC-NN','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Fmeasure rate')
axis tight
%% SVM
ST=length(GWOSVMACC)/10;
e=1:ST:length(GWOSVMACC);
figure
plot(e,sort(GWOSVMACC(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GASVMACC(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSOSVMACC(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCSVMACC(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Accuracy comparison in test prediction by feature selection methods')
legend('GWO-SVM','GA-SVM','PSO-SVM','ABC-SVM','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Accurace rate')
axis tight
%%
figure
plot(e,sort(GWOSVMRecall(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GASVMRecall(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSOSVMRecall(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCSVMRecall(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Recall comparison in test prediction by feature selection methods')
legend('GWO-SVM','GA-SVM','PSO-SVM','ABC-SVM','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Recall rate')
axis tight
%%
figure
plot(e,sort(GWOSVMPrecision(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GASVMPrecision(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSOSVMPrecision(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCSVMPrecision(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Precision comparison in test prediction by feature selection methods')
legend('GWO-SVM','GA-SVM','PSO-SVM','ABC-SVM','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Precision rate')
axis tight
%%
figure
plot(e,sort(GWOSVMFmeasure(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GASVMFmeasure(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSOSVMFmeasure(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCSVMFmeasure(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Fmeasure comparison in test prediction by feature selection methods')
legend('GWO-SVM','GA-SVM','PSO-SVM','ABC-SVM','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Fmeasure rate')
axis tight
%% NB
ST=length(GWONBACC)/10;
e=1:ST:length(GWONBACC);
figure
plot(e,sort(GWONBACC(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GANBACC(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSONBACC(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCNBACC(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Accuracy comparison in test prediction by feature selection methods')
legend('GWO-NB','GA-NB','PSO-NB','ABC-NB','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Accurace rate')
axis tight
%%
figure
plot(e,sort(GWONBRecall(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GANBRecall(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSONBRecall(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCNBRecall(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Recall comparison in test prediction by feature selection methods')
legend('GWO-NB','GA-NB','PSO-NB','ABC-NB','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Recall rate')
axis tight
%%
figure
plot(e,sort(GWONBPrecision(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GANBPrecision(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSONBPrecision(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCNBPrecision(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Precision comparison in test prediction by feature selection methods')
legend('GWO-NB','GA-NB','PSO-NB','ABC-NB','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Precision rate')
axis tight
%%
figure
plot(e,sort(GWONBFmeasure(1:ST:end)),'.-r','LineWidth',1.5,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',20)
hold on
plot(e,sort(GANBFmeasure(1:ST:end)),'.-g','LineWidth',1.5,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g',...
    'MarkerSize',20)
hold on
plot(e,sort(PSONBFmeasure(1:ST:end)),'.-m','LineWidth',1.5,...
    'MarkerEdgeColor','m',...
    'MarkerFaceColor','m',...
    'MarkerSize',20)
hold on
plot(e,sort(ABCNBFmeasure(1:ST:end)),'.-b','LineWidth',1.5,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b',...
    'MarkerSize',20)
hold off
title('Fmeasure comparison in test prediction by feature selection methods')
legend('GWO-NB','GA-NB','PSO-NB','ABC-NB','Location','northoutside','Orientation','horizontal')
xlabel('test(*500)')
ylabel('Fmeasure rate')
axis tight
%%
KNNABC_Accuracy = mean(ABCKNNACC)
KNNABC_Recall = mean(ABCKNNRecall)
KNNABC_Precision = mean(ABCKNNPrecision)
KNNABC_Fmeasure = mean(ABCKNNFmeasure)
%%
KNNGA_Accuracy = mean(GAKNNACC)
KNNGA_Recall = mean(GAKNNRecall)
KNNGA_Precision = mean(GAKNNPrecision)
KNNGA_Fmeasure = mean(GAKNNFmeasure)
%%
KNNGWO_Accuracy = mean(GWOKNNACC)
KNNGWO_Recall = mean(GWOKNNRecall)
KNNGWO_Precision = mean(GWOKNNPrecision)
KNNGWO_Fmeasure = mean(GWOKNNFmeasure)
%%
KNNPSO_Accuracy = mean(PSOKNNACC)
KNNPSO_Recall = mean(PSOKNNRecall)
KNNPSO_Precision = mean(PSOKNNPrecision)
KNNPSO_Fmeasure = mean(PSOKNNFmeasure)
%%
NNABC_Accuracy = mean(ABCNNACC)
NNABC_Recall = mean(ABCNNRecall)
NNABC_Precision = mean(ABCNNPrecision)
NNABC_Fmeasure = mean(ABCNNFmeasure)
%%
NNGA_Accuracy = mean(GANNACC)
NNGA_Recall = mean(GANNRecall)
NNGA_Precision = mean(GANNPrecision)
NNGA_Fmeasure = mean(GANNFmeasure)
%%
NNGWO_Accuracy = mean(GWONNACC)
NNGWO_Recall = mean(GWONNRecall)
NNGWO_Precision = mean(GWONNPrecision)
NNGWO_Fmeasure = mean(GWONNFmeasure)
%%
NNPSO_Accuracy = mean(PSONNACC)
NNPSO_Recall = mean(PSONNRecall)
NNPSO_Precision = mean(PSONNPrecision)
NNPSO_Fmeasure = mean(PSONNFmeasure)

%%
%%
SVMABC_Accuracy = mean(ABCSVMACC)
SVMABC_Recall = mean(ABCSVMRecall)
SVMABC_Precision = mean(ABCSVMPrecision)
SVMABC_Fmeasure = mean(ABCSVMFmeasure)
%%
SVMGA_Accuracy = mean(GASVMACC)
SVMGA_Recall = mean(GASVMRecall)
SVMGA_Precision = mean(GASVMPrecision)
SVMGA_Fmeasure = mean(GASVMFmeasure)
%%
SVMGWO_Accuracy = mean(GWOSVMACC)
SVMGWO_Recall = mean(GWOSVMRecall)
SVMGWO_Precision = mean(GWOSVMPrecision)
SVMGWO_Fmeasure = mean(GWOSVMFmeasure)
%%
SVMPSO_Accuracy = mean(PSOSVMACC)
SVMPSO_Recall = mean(PSOSVMRecall)
SVMPSO_Precision = mean(PSOSVMPrecision)
SVMPSO_Fmeasure = mean(PSOSVMFmeasure)
%%
NBABC_Accuracy = mean(ABCNBACC)
NBABC_Recall = mean(ABCNBRecall)
NBABC_Precision = mean(ABCNBPrecision)
NBABC_Fmeasure = mean(ABCNBFmeasure)
%%
NBGA_Accuracy = mean(GANBACC)
NBGA_Recall = mean(GANBRecall)
NBGA_Precision = mean(GANBPrecision)
NBGA_Fmeasure = mean(GANBFmeasure)
%%
NBGWO_Accuracy = mean(GWONBACC)
NBGWO_Recall = mean(GWONBRecall)
NBGWO_Precision = mean(GWONBPrecision)
NBGWO_Fmeasure = mean(GWONBFmeasure)
%%
NBPSO_Accuracy = mean(PSONBACC)
NBPSO_Recall = mean(PSONBRecall)
NBPSO_Precision = mean(PSONBPrecision)
NBPSO_Fmeasure = mean(PSONBFmeasure)
%% END
%% define input/output and parameters

input_fname = '../data/sila/in/bf2_tau_asymmetry_ad_long_SILA_in.csv';
output_dir = '../data/sila/out';
plots_dir = '../data/sila/out/_plots';

ROI = 'fnc_late_amyloid_right';
cutoff = 1.033;
ylimit = 3;

addpath(genpath('../../../src/software/SILA_m'));

%% read in BF2 data

disp('reading data...')
t_all = readtable(input_fname);

t = t_all(:,{'sid' 'age' ROI});
t.Properties.VariableNames = ["subid","age", "val"];
t = rmmissing(t);

toDelete = ismember(t.subid,[2204 2485]);
t(toDelete,:) = [];

if ~exist(plots_dir, 'dir')
    mkdir(plots_dir);
end

%% train the SILA model

disp('training the SILA model...')
[tsila,tdrs] = SILA(t.age,t.val,t.subid,0.25,cutoff,200);

%% estimate time to threshold and age at threshold for each subject

disp('generating subject-level estimates...')
sila_out = SILA_estimate(tsila,t.age,t.val,t.subid);

file_path = sprintf('%s/sila_out_%s.csv', output_dir, ROI);
writetable(sila_out, file_path);

%% plots to show the raw data and SILA outputs

disp('generating plots...')

% spaghetti plot of value vs. age for simulated data
figure('Units','centimeters','Position',[2,2,12,8])
spaghetti_plot(t.age,t.val,t.subid)
hold on, plot(xlim,ylimit*[1,1],'--k')
title('Raw data')
xlabel('Age (years)');
ylabel(ROI, 'Interpreter', 'none');

% plots showing the output from descrete rate sampling (i.e., rate vs. value) 
% and modeled value vs. time data.
figure('Units','centimeters','Position',[2,2,12,12])
subplot(2,1,1)
plot(tdrs.val,tdrs.rate,'-'),hold on
plot(tdrs.val,tdrs.rate + tdrs.ci,'--r')
plot(tdrs.val,tdrs.rate - tdrs.ci,'--r')
title('Discrete Rate Sampling Curve')
xlabel('Value');
ylabel('\DeltaValue per Year');

subplot(2,1,2)
plot(tsila.adtime,tsila.val,'-'),hold on
plot(xlim,ylimit*[1,1],'--k')
title('SILA Modeled{\it Value vs. Time} Curve')
xlabel('Time from Threshold');
ylabel(ROI, 'Interpreter', 'none');
legend({'Modeled curve','threshold'},'Location','northwest')

% value vs. time for all subjects
figure('Units','centimeters','Position',[2,2,12,8])
spaghetti_plot(sila_out.estdtt0,sila_out.val,sila_out.subid)
plot(tsila.adtime,tsila.val,'-k'), hold on
hold on, plot(xlim,ylimit*[1,1],'--k')
title('Data Aligned by Estimated Time to Threshold')
xlabel('Estimated time to threshold (years)');
ylabel(ROI, 'Interpreter', 'none');

plot_path = sprintf('%s/threshold_aligned_%s.pdf', plots_dir, ROI);
exportgraphics(gcf, plot_path, 'ContentType', 'vector', 'Resolution', 300)

% value vs. time for an indivdual case
sub = find(sila_out.estdtt0>1 & sila_out.estdtt0<10,1);
ids = sila_out.subid==sila_out.subid(sub);

figure('Units','centimeters','Position',[2,2,9,12])
subplot(2,1,1)
spaghetti_plot(sila_out.age(ids),sila_out.val(ids),sila_out.subid(ids))
hold on, plot([min(t.age),max(t.age)],ylimit*[1,1],'--k')
title('Observations by Age')
xlabel('Age (years)');
ylabel(ROI, 'Interpreter', 'none');
legend({'Individual Case Observations'},'Location','northwest')
ylim([min(tsila.val),max(tsila.val)])
xlim([min(t.age),max(t.age)])

subplot(2,1,2)
spaghetti_plot(sila_out.estdtt0(ids),sila_out.val(ids),sila_out.subid(ids))
plot(tsila.adtime,tsila.val,'-k'),hold on
hold on, plot(xlim,ylimit*[1,1],'--k')
xlim([min(tsila.adtime),max(tsila.adtime)])
title('Observations by Estimated Time to Threshold')
xlabel('Estimated time to threshold (years)');
ylabel(ROI, 'Interpreter', 'none');
legend({'Individual Case Observations','SILA Modeled Values'},'Location','northwest')
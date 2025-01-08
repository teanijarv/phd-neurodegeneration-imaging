%%% DEFINE

atlas = "dk";
modality = "fc";
mask = "normal_mask_90"; % normal_mask_90 or no_mask
groups = {'S', 'LA'};
use_covars = true;
contrast = '[-1, 1, 0, 0, 0]';
threshold = '3.0';
permutations = '5000'; 
alpha = '0.05';

%%% NBS ANALYSIS

% Add the NBS functions to path
% addpath(genpath('interface/NBS_m'));
addpath(genpath('../../../src/software/NBS_m'));

% Input file directories
% data_dir = "data";
data_dir = "../data";
input_dir = sprintf("%s/nbs/in", data_dir);
corr_fname = sprintf("%s_%s_file", atlas, modality);

% Temporary file directories
mkdir ../data/nbs/temp
temp_dir = "../data/nbs/temp";

% Output file directory
output_dir = sprintf("%s/nbs/out", data_dir);

% Read and merge the correlation matrices for two groups
corr_1 = load(sprintf("%s/%s_%s_%s.mat", input_dir, ...
              corr_fname, groups{1}, mask));
corr_1 = cell2mat(struct2cell(corr_1));
corr_2 = load(sprintf("%s/%s_%s_%s.mat", input_dir, ...
              corr_fname, groups{2}, mask));
corr_2 = cell2mat(struct2cell(corr_2));
corr = cat(3, corr_1, corr_2);

% Create a design matrix
len_group_1 = size(corr_1, 3);
len_group_2 = size(corr_2, 3);
design_mat = zeros(len_group_1+len_group_2, 2);
design_mat(1:len_group_1, 1) = 1;
design_mat(len_group_1+1:end, 2) = 1;

% Add covariates
if use_covars
    covars_1 = load(sprintf("%s/covars_%s_%s.mat", input_dir, modality, groups{1}));
    covars_1 = cell2mat(struct2cell(covars_1));
    covars_2 = load(sprintf("%s/covars_%s_%s.mat", input_dir, modality, groups{2}));
    covars_2 = cell2mat(struct2cell(covars_2));
    covars = [covars_1; covars_2];
    design_mat = [design_mat, covars];
end

% Export the correlation matrix and design matrix to temp directory
groups_str = strjoin(groups, '-');
design_mat_file = sprintf("%s/design_%s_%s.mat", temp_dir, corr_fname, groups_str);
corr_mat_file = sprintf("%s/corrs_%s_%s_%s.mat", temp_dir, corr_fname, groups_str, mask);
save(design_mat_file, 'design_mat');
save(corr_mat_file, 'corr');

% Insert NBS model parameters & run
UI.design.ui = design_mat_file;
UI.matrices.ui = corr_mat_file;
UI.node_coor.ui = sprintf("%s/%s_coords.txt", input_dir, atlas);                        
UI.node_label.ui = sprintf("%s/%s_labels.txt", input_dir, atlas); 
UI.method.ui='Run NBS'; 
UI.test.ui='t-test';
UI.size.ui='Extent';
UI.thresh.ui=threshold;
UI.perms.ui=permutations; 
UI.alpha.ui=alpha;
UI.exchange.ui=''; 
UI.contrast.ui=contrast;
NBSrun(UI,[])

% Extract NBS results and export
global nbs
n_components = nbs.NBS.n;
adj_mat = nbs.NBS.con_mat;
pval = nbs.NBS.pval;
t_stat_mat = nbs.NBS.test_stat;
node_coords = nbs.NBS.node_coor;
node_labels = nbs.NBS.node_label;

groups_str = strjoin(groups, '-');
output_fname = sprintf("%s_%s_%s_%s_%s_%s.mat",corr_fname, ...
                      mask, groups_str, ...
                      contrast, threshold, permutations);
export_dir = sprintf("%s/%s", output_dir, output_fname);

save(export_dir, "contrast","threshold","n_components","adj_mat","pval", ...
    "t_stat_mat","node_coords","node_labels")
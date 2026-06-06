% Mandani FIS for WMH segmentation
%
% Inputs, assumed to be already normalized in [0, 1], are:
% meanNorm
% localZNorm
% topHatNorm
% priorNorm
% regionalPriorNorm
% distanceNorm
%
% Output:
% WMHscore in [0, 1]

clear;
clc;
close all;

%% CREATE MAMDANI FIS 


fis = mamfis("Name", "Compact_WMH_Mamdani_FIS");

% Mamdani inference settings
fis.AndMethod = "min";
fis.OrMethod = "max";
fis.ImplicationMethod = "min";
fis.AggregationMethod = "max";
fis.DefuzzificationMethod = "centroid";

%% MEAN 

fis = addInput(fis, [0 1], "Name", "meanNorm");

fis = addMF(fis, "meanNorm", "trapmf", [0 0 0.20 0.42], ...
    "Name", "Low");

fis = addMF(fis, "meanNorm", "trimf", [0.30 0.52 0.72], ...
    "Name", "Medium");

fis = addMF(fis, "meanNorm", "trapmf", [0.60 0.78 1 1], ...
    "Name", "High");

%% LOCAL Z 

fis = addInput(fis, [0 1], "Name", "localZNorm");

fis = addMF(fis, "localZNorm", "trapmf", [0 0 0.22 0.45], ...
    "Name", "Low");

fis = addMF(fis, "localZNorm", "trimf", [0.32 0.52 0.72], ...
    "Name", "Medium");

fis = addMF(fis, "localZNorm", "trapmf", [0.62 0.78 1 1], ...
    "Name", "High");

%% TOP-HAT 

fis = addInput(fis, [0 1], "Name", "topHatNorm");

fis = addMF(fis, "topHatNorm", "trapmf", [0 0 0.20 0.42], ...
    "Name", "Low");

fis = addMF(fis, "topHatNorm", "trimf", [0.30 0.52 0.72], ...
    "Name", "Medium");

fis = addMF(fis, "topHatNorm", "trapmf", [0.60 0.78 1 1], ...
    "Name", "High");

%% SPATIAL PRIOR 

fis = addInput(fis, [0 1], "Name", "priorNorm");

fis = addMF(fis, "priorNorm", "trapmf", [0 0 0.08 0.28], ...
    "Name", "Low");

fis = addMF(fis, "priorNorm", "trimf", [0.20 0.48 0.76], ...
    "Name", "Medium");

fis = addMF(fis, "priorNorm", "trapmf", [0.58 0.78 1 1], ...
    "Name", "High");

%% REGIONAL PRIOR 

fis = addInput(fis, [0 1], "Name", "regionalPriorNorm");

fis = addMF(fis, "regionalPriorNorm", "trapmf", [0 0 0.12 0.34], ...
    "Name", "Low");

fis = addMF(fis, "regionalPriorNorm", "trimf", [0.24 0.52 0.78], ...
    "Name", "Medium");

fis = addMF(fis, "regionalPriorNorm", "trapmf", [0.60 0.80 1 1], ...
    "Name", "High");

%% DISTANCE FROM CENTER 
% Low distance = central
% High distance = peripheral

fis = addInput(fis, [0 1], "Name", "distanceNorm");

fis = addMF(fis, "distanceNorm", "trapmf", [0 0 0.25 0.45], ...
    "Name", "Central");

fis = addMF(fis, "distanceNorm", "trimf", [0.30 0.55 0.78], ...
    "Name", "Intermediate");

fis = addMF(fis, "distanceNorm", "trapmf", [0.62 0.80 1 1], ...
    "Name", "Peripheral");

%% ANY MEMBERSHIP FUNCTIONS 
% Used only for the weak fallback rule.

fis = addMF(fis, "meanNorm", "trapmf", [-0.10 -0.05 1.05 1.10], ...
    "Name", "Any");

fis = addMF(fis, "localZNorm", "trapmf", [-0.10 -0.05 1.05 1.10], ...
    "Name", "Any");

fis = addMF(fis, "topHatNorm", "trapmf", [-0.10 -0.05 1.05 1.10], ...
    "Name", "Any");

fis = addMF(fis, "priorNorm", "trapmf", [-0.10 -0.05 1.05 1.10], ...
    "Name", "Any");

fis = addMF(fis, "regionalPriorNorm", "trapmf", [-0.10 -0.05 1.05 1.10], ...
    "Name", "Any");

fis = addMF(fis, "distanceNorm", "trapmf", [-0.10 -0.05 1.05 1.10], ...
    "Name", "Any");

%% OUTPUT: WMH SCORE 

fis = addOutput(fis, [0 1], "Name", "WMHscore");

fis = addMF(fis, "WMHscore", "trapmf", [0.00 0.00 0.08 0.22], ...
    "Name", "VeryLow");

fis = addMF(fis, "WMHscore", "trimf", [0.15 0.28 0.42], ...
    "Name", "Low");

fis = addMF(fis, "WMHscore", "trimf", [0.38 0.50 0.62], ...
    "Name", "Medium");

fis = addMF(fis, "WMHscore", "trimf", [0.58 0.72 0.86], ...
    "Name", "High");

fis = addMF(fis, "WMHscore", "trapmf", [0.78 0.90 1.00 1.00], ...
    "Name", "VeryHigh");

%% RULES
% Rule format:

% Inputs:
% 0 = don't care
% 1 = Low / Central
% 2 = Medium / Intermediate
% 3 = High / Peripheral
% 4 = Any
%
% Outputs:
% 1 = VeryLow
% 2 = Low
% 3 = Medium
% 4 = High
% 5 = VeryHigh
%
% op:
% 1 = AND
% 2 = OR

ruleList = [

  3  3  3  0  0  0   5   1.00   1;
  % R1: mean High AND localZ High AND topHat High -> VeryHigh
  % Strong image evidence alone can detect lesions even outside prior zones.

  3  3  0  0  3  0   5   1.00   1;
  % R2: mean High AND localZ High AND regional prior High -> VeryHigh

  3  0  3  0  3  0   5   1.00   1;
  % R3: mean High AND topHat High AND regional prior High -> VeryHigh

  3  3  0  3  0  0   5   1.00   1;
  % R4: mean High AND localZ High AND spatial prior High -> VeryHigh

  3  0  3  3  0  0   5   1.00   1;
  % R5: mean High AND topHat High AND spatial prior High -> VeryHigh

  3  2  3  0  3  0   4   1.00   1;
  % R6: mean High AND localZ Medium AND topHat High AND regional prior High -> High

  2  3  3  0  3  0   4   1.00   1;
  % R7: mean Medium AND localZ High AND topHat High AND regional prior High -> High

  3  3  2  0  2  0   4   1.00   1;
  % R8: mean High AND localZ High AND topHat Medium AND regional prior Medium -> High

  3  2  2  3  3  0   4   1.00   1;
  % R9: mean High AND localZ Medium AND topHat Medium AND both priors High -> High

  3  0  0  3  3  1   4   1.00   1;
  % R10: mean High AND both priors High AND Central -> High

  2  3  0  3  3  0   4   1.00   1;
  % R11: mean Medium AND localZ High AND both priors High -> High

  0  3  3  1  1  0   3   1.00   1;
  % R12: localZ High AND topHat High BUT both priors Low -> Medium
  

  % FALSE-POSITIVE SUPPRESSION RULES

  1  0  0  0  0  0   1   1.00   1;
  % R13: mean Low -> VeryLow

  0  1  1  0  0  0   1   1.00   1;
  % R14: localZ Low AND topHat Low -> VeryLow

  2  1  1  0  0  0   1   1.00   1;
  % R15: mean Medium AND localZ Low AND topHat Low -> VeryLow

  3  1  1  0  0  0   2   1.00   1;
  % R16: mean High but no local abnormality and no topHat response -> Low

  0  2  1  1  1  3   1   1.00   1;
  % R17: weak image evidence, low priors, peripheral -> VeryLow

% FALLBACK

  4  4  4  4  4  4   1   0.02   1;
  % R18: fallback background rule
];

fis = addRule(fis, ruleList);


fis = addRule(fis, ruleList);

%% OPTIMIZE RULE WEIGHTS WITH GA 
opts = struct();

opts.preprocessedFile = fullfile("Data", "preprocessed_features_for_fis.mat");

% number of patients to generalize over
% one can use all those patients used for the spatial prior computation
opts.nPatients = 1;

opts.populationSize = 30;
opts.maxGenerations = 30;
opts.eliteCount = 2;

% Initial threshold. The GA also optimizes it.
opts.threshold = 0.50;

% Loss weights
opts.lambdaDice = 1.00;
opts.lambdaSens = 0.35;
opts.lambdaFPR  = 1.20;
opts.lambdaFNR  = 0.55;
opts.lambdaPrec = 0.85;

opts.ruleWeightMin = 0.00;
opts.ruleWeightMax = 1.00;

opts.weightMutationSigma = 0.12;

opts.saveOptimizedFis = true;
opts.optimizedFisName = "optimized_WMH_FIS_GA_mamdani.fis";

[fis, bestWeights, gaHistory] = optimize_fis_rule_weights_ga(fis, opts);

%% SAVE OPTIMIZED FIS 

outputFisName = "optimized_WMH_FIS_GA_mamdani.fis";

try
    writeFIS(fis, outputFisName);
catch
    writefis(fis, outputFisName);
end

fprintf("Optimized Mamdani FIS saved as: %s\n", outputFisName);
fprintf("Number of inputs: %d\n", numel(fis.Inputs));
fprintf("Number of outputs: %d\n", numel(fis.Outputs));
fprintf("Number of rules: %d\n", numel(fis.Rules));



%% plots: input membership functions
figure("Name", "Input membership functions");

for i = 1:numel(fis.Inputs)
    subplot(2,3,i);

    plotmf(fis, "input", i);
    grid on;

    title(fis.Inputs(i).Name, ...
        "Interpreter", "none", ...
        "Units", "normalized", ...
        "Position", [0.5 1.12 0]);

    xlabel(fis.Inputs(i).Name, "Interpreter", "none");
    ylabel("Degree of membership");
    ylim([-0.05 1.05]);

    % Removes the overlapping MF labels printed inside the plot
    textObjects = findall(gca, "Type", "Text");
    delete(textObjects);

    % Add clean legend 
    mfNames = string({fis.Inputs(i).MembershipFunctions.Name});
    legend(mfNames, ...
        "Location", "southoutside", ...
        "Interpreter", "none");
end
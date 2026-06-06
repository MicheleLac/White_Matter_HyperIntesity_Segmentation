% Preprocessing of the feature maps for the Fuzzy Inference System.

% loads features from:
% - mean_dataset.npy
% - std_dataset.npy
% - skew_dataset.npy
% - kurtosis_dataset.npy
% - cached_extra_features.mat
%
% Apply:
% 1. brain mask filtering
% 2. NaN/Inf removing
% 3. clipping to 1-99 percentiles
% 4. normalization in [0,1]
% 5. saves in .mat format
%
% Output:
% Data/preprocessed_features_for_fis.mat

clear;
clc;
close all;

%% SETTINGS 

dataDir = "Data";

outputFile = fullfile(dataDir, "preprocessed_features_for_fis.mat");

% set the values that will be later used for the feature clipping
% all the pixels values lower the first percentile will be clipped to the
% first percentile value, same for the 99 one. This is to make the
% normalization more stable.
lowPercentile  = 1;
highPercentile = 99;

% activate the visualization of some preprocessed features to understand if
% everything behaved as expected 
showPreview = true;

%% LOAD DATA 

fprintf("Loading datasets...\n");

flair = readNPY_matlab(fullfile(dataDir, "FLAIR_dataset.npy"));
wmh   = readNPY_matlab(fullfile(dataDir, "WMH_masks.npy"));

meanFeat = readNPY_matlab(fullfile(dataDir, "mean_dataset.npy"));
stdFeat  = readNPY_matlab(fullfile(dataDir, "std_dataset.npy"));
skewFeat = readNPY_matlab(fullfile(dataDir, "skew_dataset.npy"));
kurtFeat = readNPY_matlab(fullfile(dataDir, "kurtosis_dataset.npy"));

extra = load(fullfile(dataDir, "cached_extra_features.mat"));

localZFeat = extra.localZFeat;
topHatFeat = extra.topHatFeat;
wmhProbMap = extra.wmhProbMap;
distanceFeat   = extra.distanceFeat;
centralityFeat = extra.centralityFeat;

% load data that have been used to compute the spatial prior
if isfield(extra, "wmhRegionalPriorMap")
    wmhRegionalPriorMap = extra.wmhRegionalPriorMap;
else
    error("cached_extra_features.mat does not contain wmhRegionalPriorMap. Run compute_features.m first.");
end

if isfield(extra, "wmhPriorPeakCenters")
    wmhPriorPeakCenters = extra.wmhPriorPeakCenters;
else
    wmhPriorPeakCenters = [];
end

if isfield(extra, "regionalPriorPercentile")
    regionalPriorPercentile = extra.regionalPriorPercentile;
else
    regionalPriorPercentile = NaN;
end

if isfield(extra, "regionalPriorRadius")
    regionalPriorRadius = extra.regionalPriorRadius;
else
    regionalPriorRadius = NaN;
end

if isfield(extra, "regionalPriorSigma")
    regionalPriorSigma = extra.regionalPriorSigma;
else
    regionalPriorSigma = NaN;
end

% data regarding the patients used to build the prior (here 80% of the dataset)
priorTrainIdx = extra.priorTrainIdx;
priorTestIdx  = extra.priorTestIdx;
usedForPrior  = extra.usedForPrior;
diagnosis     = extra.diagnosis;

%%  FIX DIMENSIONS (if needed) 
%  makes sure that everything is in the format H x W x N.

[H, W, nImgs] = size(flair);

meanFeat   = ensureHwn(meanFeat, H, W, nImgs, "meanFeat");
stdFeat    = ensureHwn(stdFeat, H, W, nImgs, "stdFeat");
skewFeat   = ensureHwn(skewFeat, H, W, nImgs, "skewFeat");
kurtFeat   = ensureHwn(kurtFeat, H, W, nImgs, "kurtFeat");
localZFeat = ensureHwn(localZFeat, H, W, nImgs, "localZFeat");
topHatFeat = ensureHwn(topHatFeat, H, W, nImgs, "topHatFeat");
wmh        = ensureHwn(wmh, H, W, nImgs, "wmh");
distanceFeat  = ensureHwn(distanceFeat, H, W, nImgs, "distanceFeat");
centralityFeat = ensureHwn(centralityFeat, H, W, nImgs, "centralityFeat");

fprintf("Dataset size: H=%d, W=%d, N=%d\n", H, W, nImgs);

%%  BUILD BRAIN MASKS 
% The brain mask is created starting from the FLAIR image, so that not
% to evaluate pixels outside the brain. The subsequent features are
% computed only inside the masks computed here.

fprintf("Creating brain masks...\n");

brainMasks = false(H, W, nImgs);

for idx = 1:nImgs

    img = flair(:,:,idx);

    brainMask = img > 0;
    % fill possible holes in the mask
    brainMask = imfill(brainMask, "holes");
    % removes small objects (with less than 50 pixels) so that remove
    % possible small noise
    brainMask = bwareaopen(brainMask, 50);
    
    % find connected components and keep the biggest
    % again to check that small bright dots that could be found outside the
    % brain can be considered part of the mask 
    CC = bwconncomp(brainMask);
    numPixels = cellfun(@numel, CC.PixelIdxList);

    cleanBrainMask = false(size(brainMask));

    if ~isempty(numPixels)
        [~, largestIdx] = max(numPixels);
        cleanBrainMask(CC.PixelIdxList{largestIdx}) = true;
    else
        cleanBrainMask = brainMask;
    end

    brainMasks(:,:,idx) = cleanBrainMask;
end

%%  PREPROCESS FEATURE DATASETS 

fprintf("Preprocessing feature maps...\n");

meanNorm   = preprocessFeatureDataset(meanFeat,   brainMasks, lowPercentile, highPercentile, "mean");
stdNorm    = preprocessFeatureDataset(stdFeat,    brainMasks, lowPercentile, highPercentile, "std");
skewNorm   = preprocessFeatureDataset(skewFeat,   brainMasks, lowPercentile, highPercentile, "skew");
kurtNorm   = preprocessFeatureDataset(kurtFeat,   brainMasks, lowPercentile, highPercentile, "kurtosis");
localZNorm = preprocessFeatureDataset(localZFeat, brainMasks, lowPercentile, highPercentile, "localZ");
topHatNorm = preprocessFeatureDataset(topHatFeat, brainMasks, lowPercentile, highPercentile, "topHat");
distanceNorm = preprocessFeatureDataset(distanceFeat, brainMasks, 0, 100, "distance");
centralityNorm = preprocessFeatureDataset(centralityFeat, brainMasks, 0, 100, "centrality");
%% SPATIAL PRIORS 
% wmhProbMap and wmhRegionalPriorMap are 2D maps, common to all training patients.
%
% wmhProbMap:
%   exact pixel-wise prior.
%   High value means: this exact pixel was often WMH in the prior training set.
%
% wmhRegionalPriorMap:
%   regional prior.
%   High value means: this pixel is close to one of the frequent WMH poles.
%
% Since both maps are sparse, I normalize them using only positive values
% inside the global brain mask. This avoids the zeros dominating the
% percentile computation.

fprintf("Preprocessing spatial priors...\n");

globalBrainMask = any(brainMasks, 3);

% Check dimensions
if ~isequal(size(wmhProbMap), [H W])
    error("wmhProbMap has wrong size. Expected [%d %d], found [%s].", ...
        H, W, num2str(size(wmhProbMap)));
end

if ~isequal(size(wmhRegionalPriorMap), [H W])
    error("wmhRegionalPriorMap has wrong size. Expected [%d %d], found [%s].", ...
        H, W, num2str(size(wmhRegionalPriorMap)));
end

% Exact pixel-wise prior
priorNorm2D = robustNormalizePositiveInsideMask( ...
    wmhProbMap, ...
    globalBrainMask, ...
    lowPercentile, ...
    highPercentile ...
    );

% Regional spatial prior
regionalPriorNorm2D = robustNormalizePositiveInsideMask( ...
    wmhRegionalPriorMap, ...
    globalBrainMask, ...
    lowPercentile, ...
    highPercentile ...
    );

% Replicate the 2D priors for all patients.
% This makes later FIS code easier because every feature has size H x W x N.
priorNorm = repmat(priorNorm2D, 1, 1, nImgs);
regionalPriorNorm = repmat(regionalPriorNorm2D, 1, 1, nImgs);

% Remove values outside each subject-specific brain mask
for idx = 1:nImgs

    currentMask = brainMasks(:,:,idx);

    tmp = priorNorm(:,:,idx);
    tmp(~currentMask) = 0;
    priorNorm(:,:,idx) = tmp;

    tmp = regionalPriorNorm(:,:,idx);
    tmp(~currentMask) = 0;
    regionalPriorNorm(:,:,idx) = tmp;
end

%%  SAVE 

metadata = struct();
metadata.description = "Preprocessed feature maps for Mamdani FIS.";
metadata.lowPercentile = lowPercentile;
metadata.highPercentile = highPercentile;
metadata.normalization = "Robust percentile clipping inside each subject brain mask, then min-max to [0,1].";
metadata.dimensionOrder = "H x W x N";
metadata.createdFrom = {
    "mean_dataset.npy"
    "std_dataset.npy"
    "skew_dataset.npy"
    "kurtosis_dataset.npy"
    "cached_extra_features.mat:localZFeat, topHatFeat, distanceFeat, centralityFeat, spatial priors"
};
metadata.notes = "priorNorm is replicated from wmhProbMap for each subject; centralityNorm = 1 - normalized distance from brain center.";

fprintf("Saving preprocessed features...\n");

save(outputFile, ...
    "meanNorm", ...
    "stdNorm", ...
    "skewNorm", ...
    "kurtNorm", ...
    "localZNorm", ...
    "topHatNorm", ...
    "priorNorm", ...
    "priorNorm2D", ...
    "regionalPriorNorm", ...
    "regionalPriorNorm2D", ...
    "wmhRegionalPriorMap", ...
    "wmhPriorPeakCenters", ...
    "regionalPriorPercentile", ...
    "regionalPriorRadius", ...
    "regionalPriorSigma", ...
    "distanceNorm", ...
    "centralityNorm", ...
    "brainMasks", ...
    "globalBrainMask", ...
    "wmh", ...
    "priorTrainIdx", ...
    "priorTestIdx", ...
    "usedForPrior", ...
    "diagnosis", ...
    "metadata", ...
    "-v7.3");

fprintf("\nDone.\n");
fprintf("Saved file:\n%s\n", outputFile);

%% ===================== PREVIEW =====================

if showPreview

    idx = 1;

    figure("Name", "Preprocessed features preview");

    subplot(3,4,1);
    imshow(flair(:,:,idx), []);
    title("FLAIR");

    subplot(3,4,2);
    imshow(meanNorm(:,:,idx), []);
    title("meanNorm");
    colorbar;

    subplot(3,4,3);
    imshow(stdNorm(:,:,idx), []);
    title("stdNorm");
    colorbar;

    subplot(3,4,4);
    imshow(localZNorm(:,:,idx), []);
    title("localZNorm");
    colorbar;

    subplot(3,4,5);
    imshow(topHatNorm(:,:,idx), []);
    title("topHatNorm");
    colorbar;

    subplot(3,4,6);
    imshow(priorNorm(:,:,idx), []);
    title("priorNorm");
    colorbar;

    subplot(3,4,7);
    imshow(centralityNorm(:,:,idx), []);
    title("centralityNorm");
    colorbar;

    subplot(3,4,8);
    imshow(wmh(:,:,idx), []);
    title("WMH mask");

    subplot(3,4,9);
    imshow(regionalPriorNorm(:,:,idx), []);
    title("regionalPriorNorm");
    colorbar;
end

%% ===================== LOCAL FUNCTIONS =====================

function dataOut = ensureHwn(dataIn, H, W, nImgs, varName)
% ensureHwn
% Converte array in formato H x W x N se è salvato come N x H x W.

    sz = size(dataIn);

    if isequal(sz, [H, W, nImgs])
        dataOut = dataIn;
        return;
    end

    if isequal(sz, [nImgs, H, W])
        fprintf("Permuting %s from N x H x W to H x W x N\n", varName);
        dataOut = permute(dataIn, [2 3 1]);
        return;
    end

    if numel(sz) == 2 && nImgs == 1 && isequal(sz, [H, W])
        dataOut = reshape(dataIn, H, W, 1);
        return;
    end

    error("Unexpected size for %s. Found [%s], expected [%d %d %d] or [%d %d %d].", ...
        varName, num2str(sz), H, W, nImgs, nImgs, H, W);
end

function featureNormDataset = preprocessFeatureDataset(featureDataset, brainMasks, lowP, highP, featureName)
% Preprocesses a feature 3D H x W x N for each patient.

    [H, W, nImgs] = size(featureDataset);

    featureNormDataset = zeros(H, W, nImgs, "single");

    for idx = 1:nImgs

        if mod(idx, 50) == 0 || idx == 1
            fprintf("  %s: subject %d / %d\n", featureName, idx, nImgs);
        end

        feature = featureDataset(:,:,idx);
        mask = brainMasks(:,:,idx);

        featureNorm = robustNormalizeInsideMask(feature, mask, lowP, highP);

        featureNormDataset(:,:,idx) = single(featureNorm);
    end
end

function featureNorm = robustNormalizeInsideMask(feature, brainMask, lowP, highP)
% robustNormalizeInsideMask
%
% Normalizes one 2D feature map inside the subject brain mask.
% Percentiles are computed using all finite values inside the mask.
% Output is clipped to [0, 1] and set to 0 outside the brain.

feature = double(feature);

if ~isequal(size(feature), size(brainMask))
    error("The feature and the brainMask must have the same size.");
end

feature(~isfinite(feature)) = NaN;

vals = feature(brainMask);
vals = vals(isfinite(vals));

if isempty(vals)
    featureNorm = zeros(size(feature));
    return;
end

lowVal = prctile(vals, lowP);
highVal = prctile(vals, highP);

if abs(highVal - lowVal) < eps
    featureNorm = zeros(size(feature));
    return;
end

featureClipped = feature;
featureClipped(featureClipped < lowVal) = lowVal;
featureClipped(featureClipped > highVal) = highVal;

featureNorm = zeros(size(feature));

validMask = brainMask & isfinite(featureClipped);

featureNorm(validMask) = ...
    (featureClipped(validMask) - lowVal) ./ (highVal - lowVal + eps);

featureNorm(featureNorm < 0) = 0;
featureNorm(featureNorm > 1) = 1;

featureNorm(~brainMask) = 0;
featureNorm(~isfinite(featureNorm)) = 0;
end

function featureNorm = robustNormalizePositiveInsideMask(feature, brainMask, lowP, highP)
% robustNormalizePositiveInsideMask
%
% Similar to robustNormalizeInsideMask, but it computes the percentiles
% only on positive values inside the brain mask.
%
% This is useful for sparse spatial priors, because most pixels are zero.
% If zeros are included in the percentile computation, the normalization
% can collapse the map and make the prior almost invisible.

    feature = double(feature);

    if ~isequal(size(feature), size(brainMask))
        error("The feature and the brainMask must have the same size.");
    end

    feature(~isfinite(feature)) = NaN;

    vals = feature(brainMask);
    vals = vals(isfinite(vals));
    vals = vals(vals > 0);

    if isempty(vals)
        featureNorm = zeros(size(feature));
        return;
    end

    lowVal = prctile(vals, lowP);
    highVal = prctile(vals, highP);

    if abs(highVal - lowVal) < eps
        % Fallback: simple max normalization on positive values
        maxVal = max(vals);

        if maxVal <= 0
            featureNorm = zeros(size(feature));
            return;
        end

        featureNorm = zeros(size(feature));
        featureNorm(brainMask) = feature(brainMask) ./ maxVal;
        featureNorm(featureNorm < 0) = 0;
        featureNorm(featureNorm > 1) = 1;
        featureNorm(~brainMask) = 0;
        featureNorm(~isfinite(featureNorm)) = 0;
        return;
    end

    featureClipped = feature;

    featureClipped(featureClipped < lowVal) = lowVal;
    featureClipped(featureClipped > highVal) = highVal;

    featureNorm = zeros(size(feature));

    positiveMask = brainMask & isfinite(featureClipped) & feature > 0;

    featureNorm(positiveMask) = ...
        (featureClipped(positiveMask) - lowVal) ./ (highVal - lowVal + eps);

    featureNorm(featureNorm < 0) = 0;
    featureNorm(featureNorm > 1) = 1;

    featureNorm(~brainMask) = 0;
    featureNorm(~isfinite(featureNorm)) = 0;
end
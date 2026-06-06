clear; close all; clc;

dataDir = "Data";

flair = readNPY_matlab(fullfile(dataDir, "FLAIR_dataset.npy"));
wmh   = readNPY_matlab(fullfile(dataDir, "WMH_masks.npy"));

% Diagnosis: vector with one label per patient/image
diagnosis = readNPY_matlab(fullfile(dataDir, "diagnosis.npy"));
diagnosis = squeeze(diagnosis); %squeeze removes singleton dimensions

[H, W, nImgs] = size(flair);

if numel(diagnosis) ~= nImgs
    error("diagnosis.npy must contain one diagnosis label for each image. Expected %d, found %d.", ...
        nImgs, numel(diagnosis));
end

fprintf("Computing cached features for %d images...\n", nImgs);

% intialize the features, use single here to initially save space 
localZFeat  = nan(H, W, nImgs, "single");
topHatFeat  = nan(H, W, nImgs, "single");
distanceFeat   = nan(H, W, nImgs, "single");
centralityFeat = nan(H, W, nImgs, "single");

%% 
%  Stratified 80% split for spatial prior computation
%  The prior is computed only using this subset, not the whole dataset.
%  Stratification preserves the disease/non-disease proportions.


rng(1);  % fixed seed for reproducibility

priorFraction = 0.80;
nPrior = round(priorFraction * nImgs);

% find all unique diagnosis labels
classes = unique(diagnosis);
nClasses = numel(classes);

% initialize quota variables 
priorTrainIdx = [];  %indices of patients used for the prior
classCounts = zeros(nClasses,1);
rawQuotas = zeros(nClasses,1);  % the ideal number of patients in the split
baseQuotas = zeros(nClasses,1); % store the integer number of patient to actually use per quota

% First compute proportional quotas for each class
for c = 1:nClasses
    classIdx = find(diagnosis == classes(c));
    classCounts(c) = numel(classIdx);

    rawQuotas(c) = priorFraction * classCounts(c);
    baseQuotas(c) = floor(rawQuotas(c));
end

% Adjust quotas so that total selected patients is exactly nPrior
remaining = nPrior - sum(baseQuotas);

%sort classes form the one with more remainder of patients to the one with less
[~, order] = sort(rawQuotas - baseQuotas, "descend");

% the classes with the biggest remained get an additional patient (those are the proportionally bigger)
for r = 1:remaining
    baseQuotas(order(r)) = baseQuotas(order(r)) + 1;
end

% Randomly select patients inside each diagnosis class

% For each diagnosis class: 
% find all patients in that class
% randomly shuffle them
% select the required number
% append them to priorTrainIdx
for c = 1:nClasses
    classIdx = find(diagnosis == classes(c));
    classIdx = classIdx(randperm(numel(classIdx)));

    nSelect = baseQuotas(c);
    selectedIdx = classIdx(1:nSelect);

    priorTrainIdx = [priorTrainIdx; selectedIdx(:)];
end


priorTrainIdx = sort(priorTrainIdx);
priorTestIdx = setdiff((1:nImgs)', priorTrainIdx);

fprintf("\nSpatial prior will be computed using %d / %d images (%.1f%%).\n", ...
    numel(priorTrainIdx), nImgs, 100*numel(priorTrainIdx)/nImgs);

fprintf("\nDiagnosis distribution in full dataset:\n");
for c = 1:nClasses
    fprintf("  Diagnosis %g: %d / %d images (%.2f%%)\n", ...
        classes(c), ...
        sum(diagnosis == classes(c)), ...
        nImgs, ...
        100 * sum(diagnosis == classes(c)) / nImgs);
end

fprintf("\nDiagnosis distribution in prior subset:\n");
for c = 1:nClasses
    fprintf("  Diagnosis %g: %d / %d images (%.2f%%)\n", ...
        classes(c), ...
        sum(diagnosis(priorTrainIdx) == classes(c)), ...
        numel(priorTrainIdx), ...
        100 * sum(diagnosis(priorTrainIdx) == classes(c)) / numel(priorTrainIdx));
end

%% Spatial prior computed only on stratified 80%
% pixel/area has WMH prior if at least one WMH pixel appears inside a small
% 5x5 neighbourhood

windowSize = 5;

wmhPriorStack = false(H, W, numel(priorTrainIdx));

for k = 1:numel(priorTrainIdx)

    idx = priorTrainIdx(k);

    mask = wmh(:,:,idx) > 0;

    % For each pixel, check if there is at least one WMH pixel
    % in a local window around it.
    localWMH = conv2(double(mask), ones(windowSize, windowSize), "same") > 0;

    wmhPriorStack(:,:,k) = localWMH;
end

% Probability that each local region contains at least one WMH pixel
wmhProbMap = single(mean(wmhPriorStack, 3));
%% ============================================================
%  Regional spatial prior
%  This feature generalizes the 5x5 window prior.
%  Instead of only rewarding small regions that were frequently WMH pixels,
%  it creates wider high-probability regions only around the brightest
%  peaks of the original prior.
%
%  
%  The map is still computed only from priorTrainIdx.
%  Therefore, test patients are not used.


% Parameters that you can tune
regionalPriorPercentile = 90;   % selects only the brightest prior areas
regionalPriorRadius     = 25;   % radius of the red-like regions, in pixels
regionalPriorSigma      = 11;    % smoothness of the regional heat around peaks

% Smooth the original prior to make peak detection more stable
smoothedPrior = imgaussfilt(double(wmhProbMap), 2);

% Use only non-zero prior values to avoid the background dominating percentiles
positivePriorValues = smoothedPrior(smoothedPrior > 0);

if isempty(positivePriorValues)
    warning("The WMH prior map contains no positive values. Regional prior will be empty.");
    wmhRegionalPriorMap = zeros(H, W, "single");
    wmhPriorPeakCenters = [];
else
    % Select the brightest regions of the prior map
    brightThreshold = prctile(positivePriorValues, regionalPriorPercentile);
    brightPriorMask = smoothedPrior >= brightThreshold;

    % Clean tiny isolated points
    brightPriorMask = bwareaopen(brightPriorMask, 3);

    % Connected components represent the main high-prior poles
    CC_prior = bwconncomp(brightPriorMask);

    wmhRegionalPriorMap = zeros(H, W);
    wmhPriorPeakCenters = zeros(CC_prior.NumObjects, 2);

    [Xgrid, Ygrid] = meshgrid(1:W, 1:H);

    for k = 1:CC_prior.NumObjects

        pixIdx = CC_prior.PixelIdxList{k};

        % Coordinates of this bright prior component
        [rows, cols] = ind2sub([H W], pixIdx);

        % Use prior intensity as weight, so the center is pulled toward
        % the brightest part of the component
        weights = smoothedPrior(pixIdx);
        weights = weights + eps;

        centerY = sum(rows .* weights) / sum(weights);
        centerX = sum(cols .* weights) / sum(weights);

        wmhPriorPeakCenters(k,:) = [centerY, centerX];

        % Distance of every pixel from this peak center
        distFromCenter = sqrt((Xgrid - centerX).^2 + (Ygrid - centerY).^2);

        % Circular support: the red area around the peak
        circularRegion = distFromCenter <= regionalPriorRadius;

        % Soft Gaussian heat centered on the peak
        gaussianRegion = exp(-(distFromCenter.^2) / (2 * regionalPriorSigma^2));

        % Keep only the selected circular neighborhood
        gaussianRegion(~circularRegion) = 0;

        % Weight the region using the peak intensity
        peakWeight = max(weights);

        % Combine multiple regions using max, not sum, to keep range stable
        wmhRegionalPriorMap = max(wmhRegionalPriorMap, peakWeight * gaussianRegion);
    end

    % Normalize to [0, 1]
    if max(wmhRegionalPriorMap(:)) > 0
        wmhRegionalPriorMap = wmhRegionalPriorMap ./ max(wmhRegionalPriorMap(:));
    end

    wmhRegionalPriorMap = single(wmhRegionalPriorMap);
end
%% Save file indicating which patients were used for the prior

usedForPrior = false(nImgs,1);
usedForPrior(priorTrainIdx) = true;

priorTable = table( ...
    (1:nImgs)', ...
    diagnosis(:), ...
    usedForPrior, ...
    'VariableNames', {'patient_index', 'diagnosis', 'used_for_prior'} ...
);

writetable(priorTable, fullfile(dataDir, "prior_patients_used.csv"));

save(fullfile(dataDir, "prior_patients_used.mat"), ...
    "priorTrainIdx", "priorTestIdx", "diagnosis", "usedForPrior", ...
    "priorFraction", "classes");

fprintf("\nPrior patient list saved in:\n");
fprintf("  Data/prior_patients_used.csv\n");
fprintf("  Data/prior_patients_used.mat\n");

%% Compute image-specific features


for idx = 1:nImgs

    fprintf("Image %d / %d\n", idx, nImgs);

    img = flair(:,:,idx);

    %% Brain mask
    brainMask = img > 0;
    brainMask = imfill(brainMask, "holes");
    brainMask = bwareaopen(brainMask, 50);

    CC = bwconncomp(brainMask);
    numPixels = cellfun(@numel, CC.PixelIdxList);

    cleanBrainMask = false(size(brainMask));

    if ~isempty(numPixels)
        [~, largestIdx] = max(numPixels);
        cleanBrainMask(CC.PixelIdxList{largestIdx}) = true;
    else
        cleanBrainMask = brainMask;
    end
    
    %% Distance from brain center / centrality

    [y, x] = find(cleanBrainMask);

    if ~isempty(x)
        cx = mean(x);
        cy = mean(y);

        [X, Y] = meshgrid(1:W, 1:H);

        dist = sqrt((X - cx).^2 + (Y - cy).^2);
        dist(~cleanBrainMask) = NaN;

        % Raw centrality: high near center, low near border.
        % Normalization will be done later in preprocessing.m.
        maxDist = max(dist(cleanBrainMask), [], "omitnan");

        cent = nan(H, W);
        cent(cleanBrainMask) = 1 - dist(cleanBrainMask) ./ (maxDist + eps);

        distanceFeat(:,:,idx)   = single(dist);
        centralityFeat(:,:,idx) = single(cent);
    end

    %% Local z-score
    localMean = colfilt(img, [5 5], "sliding", @mean);
    localStd  = colfilt(img, [5 5], "sliding", @std);

    localZ = (img - localMean) ./ (localStd + eps);

    % Keep meaningful values only inside the brain
    localZ(~cleanBrainMask) = NaN;

    %% White top-hat
    se = strel("disk", 3);
    topHat = imtophat(img, se);

    % Keep meaningful values only inside the brain
    topHat(~cleanBrainMask) = NaN;

    %% Store
    localZFeat(:,:,idx) = single(localZ);
    topHatFeat(:,:,idx) = single(topHat);
end

%% Save cached features

save(fullfile(dataDir, "cached_extra_features.mat"), ...
    "localZFeat", ...
    "topHatFeat", ...
    "distanceFeat", ...
    "centralityFeat", ...
    "wmhProbMap", ...
    "wmhRegionalPriorMap", ...  
    "wmhPriorPeakCenters", ...
    "regionalPriorPercentile", ...
    "regionalPriorRadius", ...
    "regionalPriorSigma", ...
    "priorTrainIdx", ...
    "priorTestIdx", ...
    "diagnosis", ...
    "usedForPrior", ...
    "-v7.3");

fprintf("\nDone. Features saved in Data/cached_extra_features.mat\n");
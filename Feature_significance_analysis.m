clear; close all; clc;

dataDir = "Data";

flair = readNPY_matlab(fullfile(dataDir, "FLAIR_dataset.npy"));
wmh   = readNPY_matlab(fullfile(dataDir, "WMH_masks.npy"));

diagnosis    = readNPY_matlab(fullfile(dataDir, "diagnosis.npy"));
kurtosisFeat = readNPY_matlab(fullfile(dataDir, "kurtosis_dataset.npy"));
meanFeat     = readNPY_matlab(fullfile(dataDir, "mean_dataset.npy"));
skewFeat     = readNPY_matlab(fullfile(dataDir, "skew_dataset.npy"));
stdFeat      = readNPY_matlab(fullfile(dataDir, "std_dataset.npy"));

disp("FLAIR size:");    disp(size(flair))
disp("WMH size:");      disp(size(wmh))
disp("Diagnosis size:");disp(size(diagnosis))
disp("Kurtosis size:"); disp(size(kurtosisFeat))
disp("Mean size:");     disp(size(meanFeat))
disp("Skew size:");     disp(size(skewFeat))
disp("Std size:");      disp(size(stdFeat))

%% Feature significance analysis over the whole dataset

nImgs = size(flair, 3);

featureNames = ["Mean", "Std", "Skew", "Kurtosis", ...
                "LocalZ", "TopHat", ...
                "X position", "Y position", "Distance center", ...
                "Spatial prior", "Regional spatial prior"];

% Store metrics: rows = images, columns = features
muWMH_all   = nan(nImgs, numel(featureNames));
muNon_all   = nan(nImgs, numel(featureNames));
cohenD_all  = nan(nImgs, numel(featureNames));
auc_all     = nan(nImgs, numel(featureNames));
aucAbs_all  = nan(nImgs, numel(featureNames));

[H, W, nImgs] = size(flair);

cachedFile = fullfile(dataDir, "cached_extra_features.mat");

if ~isfile(cachedFile)
    error("Cached features not found. Run precomputeFeaturesOnce.m first.");
end


load(cachedFile, ...
    "localZFeat", ...
    "topHatFeat", ...
    "wmhProbMap", ...
    "wmhRegionalPriorMap", ...
    "priorTrainIdx", ...
    "priorTestIdx", ...
    "usedForPrior");

fprintf("Loaded cached localZFeat, topHatFeat, wmhProbMap, regional prior, and prior split indices.\n");

% Use only the 20% of patients NOT used to build the prior
evalIdx = priorTestIdx(:)';

fprintf("\nSpatial prior was computed using %d patients.\n", numel(priorTrainIdx));
fprintf("Spatial prior will be evaluated on %d held-out patients.\n", numel(evalIdx));

% Visualize empirical WMH spatial prior

figure;
imagesc(wmhProbMap);
axis image off;
colorbar;
title("Empirical WMH spatial prior");

figure;
imagesc(wmhRegionalPriorMap);
axis image off;
colorbar;
title("Regional WMH spatial prior");

for idx = evalIdx

    img = flair(:,:,idx);
    mask = wmh(:,:,idx) > 0;

    % Brain mask
    brainMask = img > 0;
    
    % Clean the brain mask a bit
    brainMask = imfill(brainMask, "holes");
    brainMask = bwareaopen(brainMask, 50);
    
    % Keep only the largest connected component
    CC = bwconncomp(brainMask);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    
    cleanBrainMask = false(size(brainMask));
    
    if ~isempty(numPixels)
        [~, largestIdx] = max(numPixels);
        cleanBrainMask(CC.PixelIdxList{largestIdx}) = true;
    else
        cleanBrainMask = brainMask;
    end
    
    valid = cleanBrainMask;

    % Brain-relative spatial features

    [X, Y] = meshgrid(1:W, 1:H);

    props = regionprops(cleanBrainMask, "BoundingBox", "Centroid");

    if isempty(props)
        continue;
    end

    bb = props(1).BoundingBox;    % [x, y, width, height]
    cent = props(1).Centroid;     % [cx, cy]

    x0 = bb(1);
    y0 = bb(2);
    brainW = bb(3);
    brainH = bb(4);

    % X position normalized inside the brain bounding box
    xBrainNorm = (X - x0) / brainW;

    % Y position normalized inside the brain bounding box
    yBrainNorm = (Y - y0) / brainH;

    % Distance from the brain centroid, not from the image center
    cxBrain = cent(1);
    cyBrain = cent(2);

    distBrainCenter = sqrt((X - cxBrain).^2 + (Y - cyBrain).^2);

    % Normalize distance using only brain pixels
    maxDistBrain = max(distBrainCenter(cleanBrainMask));
    distBrainCenter = distBrainCenter ./ maxDistBrain;

    % Outside the brain is not meaningful
    xBrainNorm(~cleanBrainMask) = NaN;
    yBrainNorm(~cleanBrainMask) = NaN;
    distBrainCenter(~cleanBrainMask) = NaN;

    % Load precomputed features for this image
    localZ = double(localZFeat(:,:,idx));
    topHat = double(topHatFeat(:,:,idx));

    % Use the spatial priors computed only from the stratified 80% subset
    spatialPrior = double(wmhProbMap);
    spatialPrior(~cleanBrainMask) = NaN;

    regionalSpatialPrior = double(wmhRegionalPriorMap);
    regionalSpatialPrior(~cleanBrainMask) = NaN;

    % Skip images with no WMH pixels
    if sum(mask(valid)) == 0
        continue;
    end

    features = {
    meanFeat(:,:,idx),       "Mean";
    stdFeat(:,:,idx),        "Std";
    skewFeat(:,:,idx),       "Skew";
    kurtosisFeat(:,:,idx),   "Kurtosis";
    localZ,                  "LocalZ";
    topHat,                  "TopHat";
    xBrainNorm,              "X position";
    yBrainNorm,              "Y position";
    distBrainCenter,         "Distance center";
    spatialPrior,            "Spatial prior";
    regionalSpatialPrior,    "Regional spatial prior"
};

    for k = 1:size(features,1)

        F = features{k,1};

        y = mask(valid);      % labels: 1 = WMH, 0 = non-WMH
        x = F(valid);         % feature values

        % Remove NaN/Inf values
        good = ~isnan(x) & ~isinf(x);
        x = x(good);
        y = y(good);

        % Skip if only one class is present or feature is constant
        if numel(unique(y)) < 2 || numel(unique(x)) < 2
            continue;
        end

        x_wmh = x(y == 1);
        x_non = x(y == 0);

        mu_wmh = mean(x_wmh);
        mu_non = mean(x_non);

        sigma_wmh = std(x_wmh);
        sigma_non = std(x_non);

        % Cohen's d
        pooled_sigma = sqrt((sigma_wmh^2 + sigma_non^2) / 2);

        if pooled_sigma == 0
            cohen_d = NaN;
        else
            cohen_d = (mu_wmh - mu_non) / pooled_sigma;
        end

        % ROC AUC
        [~,~,~,auc] = perfcurve(y, x, true);

        % Useful if feature is inversely related to WMH
        auc_abs = max(auc, 1 - auc);

        % Save metrics
        muWMH_all(idx,k)  = mu_wmh;
        muNon_all(idx,k)  = mu_non;
        cohenD_all(idx,k) = cohen_d;
        auc_all(idx,k)    = auc;
        aucAbs_all(idx,k) = auc_abs;
    end
end

%% Print average metrics over dataset

fprintf("\n===== Average feature significance over held-out 20%% prior-test set =====\n");

for k = 1:numel(featureNames)

    fprintf("\nFeature: %s\n", featureNames(k));
    fprintf("Mean WMH:        %.4f\n", mean(muWMH_all(:,k), "omitnan"));
    fprintf("Mean non-WMH:    %.4f\n", mean(muNon_all(:,k), "omitnan"));
    fprintf("Cohen d:         %.4f\n", mean(cohenD_all(:,k), "omitnan"));
    fprintf("Abs Cohen d:     %.4f\n", mean(abs(cohenD_all(:,k)), "omitnan"));
    fprintf("AUC:             %.4f\n", mean(auc_all(:,k), "omitnan"));
    fprintf("AUC abs:         %.4f\n", mean(aucAbs_all(:,k), "omitnan"));
end

%% Interpret spatial direction using brain-relative coordinates

allX = [];
allY = [];
allDist = [];
allLabels = [];

for idx = evalIdx

    img = flair(:,:,idx);
    mask = wmh(:,:,idx) > 0;

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

    props = regionprops(cleanBrainMask, "BoundingBox", "Centroid");

    if isempty(props)
        continue;
    end

    [H, W] = size(img);
    [X, Y] = meshgrid(1:W, 1:H);

    bb = props(1).BoundingBox;
    cent = props(1).Centroid;

    x0 = bb(1);
    y0 = bb(2);
    brainW = bb(3);
    brainH = bb(4);

    xBrainNorm = (X - x0) / brainW;
    yBrainNorm = (Y - y0) / brainH;

    cxBrain = cent(1);
    cyBrain = cent(2);

    distBrainCenter = sqrt((X - cxBrain).^2 + (Y - cyBrain).^2);
    distBrainCenter = distBrainCenter ./ max(distBrainCenter(cleanBrainMask));

    valid = cleanBrainMask;

    x = xBrainNorm(valid);
    yPos = yBrainNorm(valid);
    d = distBrainCenter(valid);
    labels = mask(valid);

    good = ~isnan(x) & ~isinf(x) & ...
           ~isnan(yPos) & ~isinf(yPos) & ...
           ~isnan(d) & ~isinf(d);

    allX = [allX; x(good)];
    allY = [allY; yPos(good)];
    allDist = [allDist; d(good)];
    allLabels = [allLabels; labels(good)];
end

x_wmh = allX(allLabels == 1);
x_non = allX(allLabels == 0);

y_wmh = allY(allLabels == 1);
y_non = allY(allLabels == 0);

d_wmh = allDist(allLabels == 1);
d_non = allDist(allLabels == 0);

fprintf("\n===== Brain-relative spatial interpretation =====\n");

fprintf("\nX position:\n");
fprintf("Mean X WMH:     %.4f\n", mean(x_wmh, "omitnan"));
fprintf("Mean X non-WMH: %.4f\n", mean(x_non, "omitnan"));

fprintf("\nY position:\n");
fprintf("Mean Y WMH:     %.4f\n", mean(y_wmh, "omitnan"));
fprintf("Mean Y non-WMH: %.4f\n", mean(y_non, "omitnan"));

fprintf("\nDistance from brain center:\n");
fprintf("Mean Dist WMH:     %.4f\n", mean(d_wmh, "omitnan"));
fprintf("Mean Dist non-WMH: %.4f\n", mean(d_non, "omitnan"));


% Compute spatial AUCs only if both classes are present
if numel(unique(allLabels)) >= 2

    if numel(unique(allX)) >= 2
        [~,~,~,aucX] = perfcurve(allLabels, allX, true);
        fprintf("\nAUC X position:        %.4f | AUC abs: %.4f\n", aucX, max(aucX, 1-aucX));
    else
        warning("Cannot compute AUC for X position: feature is constant.");
    end

    if numel(unique(allY)) >= 2
        [~,~,~,aucY] = perfcurve(allLabels, allY, true);
        fprintf("AUC Y position:        %.4f | AUC abs: %.4f\n", aucY, max(aucY, 1-aucY));
    else
        warning("Cannot compute AUC for Y position: feature is constant.");
    end

    if numel(unique(allDist)) >= 2
        [~,~,~,aucD] = perfcurve(allLabels, allDist, true);
        fprintf("AUC distance center:   %.4f | AUC abs: %.4f\n", aucD, max(aucD, 1-aucD));
    else
        warning("Cannot compute AUC for distance from center: feature is constant.");
    end

else
    warning("Cannot compute spatial AUCs: only one class is present.");
end

fprintf("\nInterpretation:\n");

if mean(y_wmh, "omitnan") > mean(y_non, "omitnan")
    fprintf("- WMH pixels tend to be more in the LOWER part of the brain.\n");
else
    fprintf("- WMH pixels tend to be more in the UPPER part of the brain.\n");
end

if mean(d_wmh, "omitnan") > mean(d_non, "omitnan")
    fprintf("- WMH pixels tend to be more PERIPHERAL, farther from the brain center.\n");
else
    fprintf("- WMH pixels tend to be more CENTRAL, closer to the brain center.\n");
end

if mean(x_wmh, "omitnan") > mean(x_non, "omitnan")
    fprintf("- WMH pixels tend to be more toward the RIGHT side of the image.\n");
else
    fprintf("- WMH pixels tend to be more toward the LEFT side of the image.\n");
end



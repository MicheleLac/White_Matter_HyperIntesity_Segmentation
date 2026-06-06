clear; close all; clc;

% Data loading and size visualization
dataDir = "Data";

flair = readNPY_matlab(fullfile(dataDir, "FLAIR_dataset.npy"));
wmh   = readNPY_matlab(fullfile(dataDir, "WMH_masks.npy"));

diagnosis    = readNPY_matlab(fullfile(dataDir, "diagnosis.npy"));
kurtosisFeat = readNPY_matlab(fullfile(dataDir, "kurtosis_dataset.npy"));
meanFeat     = readNPY_matlab(fullfile(dataDir, "mean_dataset.npy"));
skewFeat     = readNPY_matlab(fullfile(dataDir, "skew_dataset.npy"));
stdFeat      = readNPY_matlab(fullfile(dataDir, "std_dataset.npy"));

% Precomputed / cached features
cachedFile = fullfile(dataDir, "cached_extra_features.mat");

if ~isfile(cachedFile)
    error("Cached features not found. Run precomputeFeaturesOnce.m first.");
end

load(cachedFile, "localZFeat", "topHatFeat", "wmhProbMap", "wmhRegionalPriorMap");

disp("Local Z-score size:"); disp(size(localZFeat))
disp("Top-hat size:");      disp(size(topHatFeat))
disp("Spatial prior size:");disp(size(wmhProbMap))
disp("Regional spatial prior size:"); disp(size(wmhRegionalPriorMap))

disp("FLAIR size:");    disp(size(flair))
disp("WMH size:");      disp(size(wmh))
disp("Diagnosis size:");disp(size(diagnosis))
disp("Kurtosis size:"); disp(size(kurtosisFeat))
disp("Mean size:");     disp(size(meanFeat))
disp("Skew size:");     disp(size(skewFeat))
disp("Std size:");      disp(size(stdFeat))

%% Display one patient
idx = 25;

img = flair(:,:,idx);
mask = wmh(:,:,idx) > 0;

brainMask = img > 0;
brainMask = imfill(brainMask, "holes");
brainMask = bwareaopen(brainMask, 50);

localZ = localZFeat(:,:,idx);
topHat = topHatFeat(:,:,idx);
spatialPrior = wmhProbMap;
regionalSpatialPrior = wmhRegionalPriorMap;

figure;
tiledlayout(3,4);

nexttile;
imshow(img, []);
title("FLAIR");

nexttile;
imshow(mask, []);
title("WMH mask");

nexttile;
imagesc(meanFeat(:,:,idx));
axis image off; colorbar;
title("Mean");

nexttile;
imagesc(stdFeat(:,:,idx));
axis image off; colorbar;
title("Std");

nexttile;
imagesc(skewFeat(:,:,idx));
axis image off; colorbar;
title("Skew");

nexttile;
imagesc(kurtosisFeat(:,:,idx));
axis image off; colorbar;
title("Kurtosis");

nexttile;
imagesc(localZ);
axis image off; colorbar;
title("Local Z-score");

nexttile;
imagesc(topHat);
axis image off; colorbar;
title("White top-hat");

nexttile;
imagesc(spatialPrior);
axis image off; colorbar;
title("Spatial prior");

nexttile;
imshow(brainMask, []);
title("Brain mask");

nexttile;
imshow(img, []);
hold on;
redOverlay = cat(3, ones(size(mask)), zeros(size(mask)), zeros(size(mask)));
h = imshow(redOverlay);
set(h, "AlphaData", 0.35 * mask);
title("FLAIR + WMH overlay");

nexttile;

% Convert FLAIR to true RGB grayscale image
% This prevents colormap(gca, "jet") from recoloring the FLAIR image.
imgGray = mat2gray(img);
imgRGB = repmat(imgGray, [1 1 3]);

imshow(imgRGB);
hold on;

% Prepare prior visualization
priorVis = spatialPrior;
priorVis(~brainMask) = NaN;

% Hide weak prior values
threshold = 0.05;
priorVis(priorVis < threshold) = NaN;

% Overlay only meaningful prior areas
hPrior = imagesc(priorVis);
axis image off;

% Transparency only where prior is meaningful
alphaData = ~isnan(priorVis);
set(hPrior, "AlphaData", 0.85 * alphaData);

colormap(gca, "jet");
colorbar;
clim([threshold max(spatialPrior(:))]);

title("Spatial prior on brain");
%% Overlay the mask on the FLAIR image
img = flair(:,:,idx);
mask = wmh(:,:,idx) > 0;

figure;
imshow(img, []);
hold on;

redOverlay = cat(3, ones(size(mask)), zeros(size(mask)), zeros(size(mask)));
h = imshow(redOverlay);
set(h, "AlphaData", 0.35 * mask);

title("FLAIR with WMH mask overlay");

%% New cached features visualization only
figure;
tiledlayout(1,3);

nexttile;
imagesc(localZ);
axis image off; colorbar;
title("Local Z-score");

nexttile;
imagesc(topHat);
axis image off; colorbar;
title("White top-hat");

nexttile;
imagesc(spatialPrior);
axis image off; colorbar;
title("Dataset spatial prior");

%%
figure;

% Show spatial prior as background
priorVis = spatialPrior;
imagesc(priorVis);
axis image off;
colormap(gca, "jet");
colorbar;
clim([0 max(spatialPrior(:))]);
hold on;

% Overlay WMH mask as white contour
contour(mask, [0.5 0.5], "w", "LineWidth", 1.8);

title("Spatial prior + WMH mask");

%% Regional spatial prior + WMH mask overlay
figure;

% Background: regional prior
regionalPriorVis = regionalSpatialPrior;

% Keep only brain area visible
regionalPriorVis(~brainMask) = NaN;

% hide very weak values so the map is easier to interpret
thr = 0.05 * max(regionalSpatialPrior(:));
regionalPriorVis(regionalPriorVis < thr) = NaN;

imagesc(regionalPriorVis);
axis image off;
colormap(gca, "jet");
colorbar;
hold on;

% Overlay WMH mask as white contour
contour(mask, [0.5 0.5], "w", "LineWidth", 1.8);

title("Regional spatial prior + WMH mask");
function [optimizedFis, bestWeights, history] = optimize_fis_rule_weights_ga(fis, opts)
% Genetic algorithm for optimizing Mamdani FIS rule weights.

% The GA chromosome is:
% [ruleWeight_1 ... ruleWeight_N threshold]


%% DEFAULT OPTIONS 

if nargin < 2
    opts = struct();
end

opts = setDefault(opts, "preprocessedFile", fullfile("Data", "preprocessed_features_for_fis.mat"));
opts = setDefault(opts, "patientIdx", []);
opts = setDefault(opts, "nPatients", 2);

opts = setDefault(opts, "populationSize", 24);
opts = setDefault(opts, "maxGenerations", 30);
opts = setDefault(opts, "eliteCount", 2);

opts = setDefault(opts, "crossoverRate", 0.85);
opts = setDefault(opts, "mutationRate", 0.15);

opts = setDefault(opts, "weightMutationSigma", 0.15);

opts = setDefault(opts, "ruleWeightMin", 0.00);
opts = setDefault(opts, "ruleWeightMax", 1.00);

opts = setDefault(opts, "threshold", 0.50);

opts = setDefault(opts, "lambdaSens", 0.50);
opts = setDefault(opts, "lambdaFPR", 0.05);
opts = setDefault(opts, "lambdaFNR", 0.60);
opts = setDefault(opts, "lambdaPrec", 0.50);

opts = setDefault(opts, "randomSeed", 1);
opts = setDefault(opts, "saveOptimizedFis", true);
opts = setDefault(opts, "optimizedFisName", "optimized_WMH_FIS_GA_mamdani.fis");

rng(opts.randomSeed);

%% LOAD PREPROCESSED DATA 

if ~isfile(opts.preprocessedFile)
    error("Could not find %s. Run preprocessing.m first.", opts.preprocessedFile);
end

load(opts.preprocessedFile, ...
    "meanNorm", ...
    "localZNorm", ...
    "topHatNorm", ...
    "priorNorm", ...
    "regionalPriorNorm", ...
    "distanceNorm", ...
    "brainMasks", ...
    "wmh", ...
    "priorTrainIdx", ...
    "priorTestIdx");

fprintf("\n===== Genetic Algorithm for Mamdani FIS Rule Weights =====\n");
fprintf("Loaded preprocessed data from:\n%s\n", opts.preprocessedFile);

%%  CHOOSE TRAINING PATIENTS 

if isempty(opts.patientIdx)

    candidateIdx = priorTrainIdx(:);
    lesionCounts = zeros(numel(candidateIdx), 1);

    for i = 1:numel(candidateIdx)
        idx = candidateIdx(i);
        lesionCounts(i) = nnz((wmh(:,:,idx) > 0) & (brainMasks(:,:,idx) > 0));
    end

    validPatients = candidateIdx(lesionCounts > 0);
    validCounts = lesionCounts(lesionCounts > 0);

    if isempty(validPatients)
        error("No WMH-positive patients found in priorTrainIdx.");
    end

    [~, order] = sort(validCounts, "descend");
    nUse = min(opts.nPatients, numel(order));

    trainPatientIdx = validPatients(order(1:nUse));

else
    trainPatientIdx = opts.patientIdx(:);
end

fprintf("Patients used by GA: ");
fprintf("%d ", trainPatientIdx);
fprintf("\n");

%% PRECOMPUTE FIS INPUT MATRICES 

data = struct();

for p = 1:numel(trainPatientIdx)

    idx = trainPatientIdx(p);

    brainMask = brainMasks(:,:,idx) > 0;
    trueMask  = wmh(:,:,idx) > 0;

    meanImg          = meanNorm(:,:,idx);
    localZImg        = localZNorm(:,:,idx);
    topHatImg        = topHatNorm(:,:,idx);
    priorImg         = priorNorm(:,:,idx);
    regionalPriorImg = regionalPriorNorm(:,:,idx);
    distanceImg      = distanceNorm(:,:,idx);

    validPixels = brainMask;
    
    X = [
        meanImg(validPixels), ...
        localZImg(validPixels), ...
        topHatImg(validPixels), ...
        priorImg(validPixels), ...
        regionalPriorImg(validPixels), ...
        distanceImg(validPixels)
    ];

    y = trueMask(validPixels);

    invalidRows = any(~isfinite(X), 2);

    X(invalidRows, :) = [];
    y(invalidRows) = [];

    epsFIS = 1e-6;
    X = max(min(X, 1 - epsFIS), epsFIS);
    X(~isfinite(X)) = 0;

    data(p).idx = idx;
    data(p).X = X;
    data(p).y = y(:) > 0;

    fprintf("Prepared patient %d: %d brain pixels, %d WMH pixels\n", ...
        idx, size(X,1), nnz(y));
end

%%  GA INITIALIZATION 

numRules = numel(fis.Rules);

if numRules == 0
    error("The FIS has no rules. Add rules before running the GA.");
end

currentRuleWeights = getRuleWeights(fis);

% Chromosome structure:
% [ruleWeight_1 ... ruleWeight_N threshold]
numRuleGenes = numel(currentRuleWeights);
numThresholdGenes = 1;
numGenes = numRuleGenes + numThresholdGenes;

fprintf("\nNumber of rule-weight genes: %d\n", numRuleGenes);
fprintf("Number of threshold genes: %d\n", numThresholdGenes);
fprintf("Total GA chromosome length: %d\n", numGenes);

initialThreshold = opts.threshold;
initialChromosome = [currentRuleWeights, initialThreshold];

lowerBounds = [
    opts.ruleWeightMin * ones(1, numRuleGenes), ...
    0.45
];

upperBounds = [
    opts.ruleWeightMax * ones(1, numRuleGenes), ...
    0.85
];

popSize = opts.populationSize;
numGenerations = opts.maxGenerations;

population = zeros(popSize, numGenes);

% Individual 1: current handcrafted rule weights + initial threshold
population(1,:) = initialChromosome;

% Individual 2: all rule weights = 1, threshold = 0.60
if popSize >= 2
    population(2,:) = [ones(1, numRuleGenes), 0.60];
end

% Individual 3: current rule weights, stricter threshold = 0.70
if popSize >= 3
    population(3,:) = [currentRuleWeights, 0.70];
end

% Other individuals: random rule weights + random threshold
for i = 4:popSize

    randomWeights = opts.ruleWeightMin + ...
        (opts.ruleWeightMax - opts.ruleWeightMin) * rand(1, numRuleGenes);

    randomThreshold = 0.45 + (0.85 - 0.45) * rand();

    population(i,:) = [randomWeights, randomThreshold];
end

population = clampPopulation(population, lowerBounds, upperBounds);


fitness = inf(popSize, 1);
metricsPop = repmat(emptyMetrics(), popSize, 1);

history.bestFitness = zeros(numGenerations, 1);
history.bestDice = zeros(numGenerations, 1);
history.bestSensitivity = zeros(numGenerations, 1);
history.bestPrecision = zeros(numGenerations, 1);
history.bestFPR = zeros(numGenerations, 1);
history.bestFNR = zeros(numGenerations, 1);

history.bestWeights = zeros(numGenerations, numRuleGenes);
history.bestThreshold = zeros(numGenerations, 1);

history.bestChromosome = zeros(numGenerations, numGenes);

bestFitness = inf;
bestChromosome = initialChromosome;
bestMetrics = emptyMetrics();

%%  GA MAIN LOOP 

for gen = 1:numGenerations

    fprintf("\nGeneration %d / %d\n", gen, numGenerations);

    for i = 1:popSize
        [fitness(i), metricsPop(i)] = evaluateIndividual( ...
            population(i,:), fis, data, opts);
    end

    [fitness, sortIdx] = sort(fitness, "ascend");
    population = population(sortIdx, :);
    metricsPop = metricsPop(sortIdx);

    if fitness(1) < bestFitness
        bestFitness = fitness(1);
        bestChromosome = population(1,:);
        bestMetrics = metricsPop(1);
    end

    bestRuleWeights = bestChromosome(1:numRuleGenes);
    bestThreshold = bestChromosome(end);
    history.bestThreshold(gen) = bestThreshold;

    history.bestFitness(gen) = bestFitness;
    history.bestDice(gen) = bestMetrics.Dice;
    history.bestSensitivity(gen) = bestMetrics.Sensitivity;
    history.bestPrecision(gen) = bestMetrics.Precision;
    history.bestFPR(gen) = bestMetrics.FPR;
    history.bestFNR(gen) = bestMetrics.FNR;

    history.bestWeights(gen,:) = bestRuleWeights;
    history.bestChromosome(gen,:) = bestChromosome;

    fprintf("Best fitness: %.4f | Dice: %.4f | Sens: %.4f | Prec: %.4f | FPR: %.4f | FNR: %.4f | Thr: %.3f\n", ...
        bestFitness, ...
        bestMetrics.Dice, ...
        bestMetrics.Sensitivity, ...
        bestMetrics.Precision, ...
        bestMetrics.FPR, ...
        bestMetrics.FNR, ...
        bestThreshold);

    newPopulation = zeros(size(population));

    eliteCount = min(opts.eliteCount, popSize);
    newPopulation(1:eliteCount,:) = population(1:eliteCount,:);

    nextIndex = eliteCount + 1;

    while nextIndex <= popSize

        parent1 = tournamentSelection(population, fitness);
        parent2 = tournamentSelection(population, fitness);

        if rand < opts.crossoverRate
            [child1, child2] = arithmeticCrossover(parent1, parent2);
        else
            child1 = parent1;
            child2 = parent2;
        end

        child1 = mutateChromosome(child1, opts);
        child2 = mutateChromosome(child2, opts);

        child1 = clampChromosome(child1, lowerBounds, upperBounds);
        child2 = clampChromosome(child2, lowerBounds, upperBounds);

        newPopulation(nextIndex,:) = child1;
        nextIndex = nextIndex + 1;

        if nextIndex <= popSize
            newPopulation(nextIndex,:) = child2;
            nextIndex = nextIndex + 1;
        end
    end

    population = newPopulation;
end

%%  RETURN OPTIMIZED FIS 

bestWeights = bestChromosome(1:numRuleGenes);
bestThreshold = bestChromosome(end);

optimizedFis = setRuleWeights(fis, bestWeights);

fprintf("Best threshold:   %.4f\n", bestThreshold);
history.finalThreshold = bestThreshold;

fprintf("\n===== GA optimization complete =====\n");
fprintf("Best Dice:        %.4f\n", bestMetrics.Dice);
fprintf("Best Sensitivity: %.4f\n", bestMetrics.Sensitivity);
fprintf("Best Precision:   %.4f\n", bestMetrics.Precision);
fprintf("Best FPR:         %.4f\n", bestMetrics.FPR);
fprintf("Best FNR:         %.4f\n", bestMetrics.FNR);

fprintf("\nBest rule weights:\n");
disp(bestWeights);

if opts.saveOptimizedFis
    try
        writeFIS(optimizedFis, opts.optimizedFisName);
    catch
        writefis(optimizedFis, opts.optimizedFisName);
    end

    fprintf("Optimized Mamdani FIS saved as:\n%s\n", opts.optimizedFisName);
end

%%  PLOT GA HISTORY 

figure("Name", "GA optimization history");

plot(history.bestDice, "LineWidth", 2);
hold on;
plot(history.bestFitness, "LineWidth", 2);
grid on;

xlabel("Generation");
ylabel("Value");
legend("Best Dice", "Best Fitness / Loss", "Location", "best");
title("GA optimization of Mamdani rule weights");

end

%% 
% LOCAL FUNCTIONS


function opts = setDefault(opts, fieldName, defaultValue)
    if ~isfield(opts, fieldName) || isempty(opts.(fieldName))
        opts.(fieldName) = defaultValue;
    end
end

function weights = getRuleWeights(fis)

    numRules = numel(fis.Rules);
    weights = zeros(1, numRules);

    for r = 1:numRules
        weights(r) = fis.Rules(r).Weight;
    end

    weights = max(min(weights, 1), 0);
end

function fisOut = setRuleWeights(fisIn, weights)

    fisOut = fisIn;

    for r = 1:numel(fisOut.Rules)
        fisOut.Rules(r).Weight = weights(r);
    end

    % Force fallback rule to stay weak.
    fallbackRuleIdx = numel(fisOut.Rules);
    fisOut.Rules(fallbackRuleIdx).Weight = 0.03;
end

function [loss, metrics] = evaluateIndividual(chromosome, baseFis, data, opts)

    
    numRules = numel(baseFis.Rules);

    ruleWeights = chromosome(1:numRules);
    threshold   = chromosome(end);

    fisCandidate = setRuleWeights(baseFis, ruleWeights);

    totalTP = 0;
    totalTN = 0;
    totalFP = 0;
    totalFN = 0;

    for p = 1:numel(data)

        X = data(p).X;
        y = data(p).y;

        scores = evalfis(fisCandidate, X);

        % Mamdani output is designed in [0, 1].
        % Clipping is still safe in case of numerical edge cases.
        scores = max(0, min(1, scores));

        % Use GA-optimized threshold, not opts.threshold
        pred = scores >= threshold;

        TP = sum(pred(:) & y(:));
        TN = sum(~pred(:) & ~y(:));
        FP = sum(pred(:) & ~y(:));
        FN = sum(~pred(:) & y(:));

        totalTP = totalTP + TP;
        totalTN = totalTN + TN;
        totalFP = totalFP + FP;
        totalFN = totalFN + FN;
    end

    metrics = computeMetricsFromCounts(totalTP, totalTN, totalFP, totalFN);

    
    % Balanced loss:
    % Main objective is Dice.
    % FNR is penalized strongly enough to avoid empty predictions.
    % FPR is still penalized, but not so much that the GA predicts nothing.
    loss = opts.lambdaDice * (1 - metrics.Dice) ...
        + opts.lambdaFNR  * metrics.FNR ...
        + opts.lambdaFPR  * metrics.FPR ...
        + opts.lambdaSens * (1 - metrics.Sensitivity) ...
        + opts.lambdaPrec * (1 - metrics.Precision);

    if metrics.Sensitivity < 0.35
        loss = loss + 1.5;
    end

    if metrics.Dice < 0.10
        loss = loss + 1.0;
    end

    % Store the threshold in metrics for printing/debugging if needed
    metrics.Threshold = threshold;
end

function metrics = computeMetricsFromCounts(TP, TN, FP, FN)

    epsVal = eps;

    metrics.Dice        = (2 * TP) / (2 * TP + FP + FN + epsVal);
    metrics.Sensitivity = TP / (TP + FN + epsVal);
    metrics.Specificity = TN / (TN + FP + epsVal);
    metrics.Precision   = TP / (TP + FP + epsVal);
    metrics.Accuracy    = (TP + TN) / (TP + TN + FP + FN + epsVal);
    metrics.FPR         = FP / (FP + TN + epsVal);
    metrics.FNR         = FN / (FN + TP + epsVal);

    metrics.TP = TP;
    metrics.TN = TN;
    metrics.FP = FP;
    metrics.FN = FN;
end

function metrics = emptyMetrics()

    metrics.Dice = 0;
    metrics.Sensitivity = 0;
    metrics.Specificity = 0;
    metrics.Precision = 0;
    metrics.Accuracy = 0;
    metrics.FPR = 0;
    metrics.FNR = 0;

    metrics.TP = 0;
    metrics.TN = 0;
    metrics.FP = 0;
    metrics.FN = 0;

    % Needed because evaluateIndividual stores the GA-optimized threshold
    metrics.Threshold = 0;
end

function selected = tournamentSelection(population, fitness)

    tournamentSize = 3;
    popSize = size(population, 1);

    candidateIdx = randi(popSize, tournamentSize, 1);
    [~, bestLocalIdx] = min(fitness(candidateIdx));

    selected = population(candidateIdx(bestLocalIdx), :);
end

function [child1, child2] = arithmeticCrossover(parent1, parent2)

    alpha = rand(size(parent1));

    child1 = alpha .* parent1 + (1 - alpha) .* parent2;
    child2 = alpha .* parent2 + (1 - alpha) .* parent1;
end

function child = mutateChromosome(child, opts)

    mutationMask = rand(size(child)) < opts.mutationRate;

    child(mutationMask) = child(mutationMask) + ...
        opts.weightMutationSigma .* randn(size(child(mutationMask)));
end

function chromosome = clampChromosome(chromosome, lowerBounds, upperBounds)

    chromosome = max(lowerBounds, min(upperBounds, chromosome));
end

function population = clampPopulation(population, lowerBounds, upperBounds)

    for i = 1:size(population, 1)
        population(i,:) = clampChromosome(population(i,:), lowerBounds, upperBounds);
    end
end
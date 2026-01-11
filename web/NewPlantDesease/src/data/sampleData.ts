// Deprecated demo module.
// All UI data should come from the Python pipeline (`data/web/*.json`) and Flask API.

export const classLabels: string[] = [];

export const classDistribution: { name: string; count: number; shortName: string }[] = [];

export const confusionMatrixData: number[][] = [];

export const hyperparameters: Record<string, unknown> = {};

export const modelMetrics: Record<string, unknown> = {};

export const trainingHistory: Array<{ epoch: number; trainLoss: number; valLoss: number; trainAcc: number; valAcc: number }> = [];

export const datasetStats: { totalImages: number; trainImages: number; validationImages: number; testImages: number; numClasses: number } = {
	totalImages: 0,
	trainImages: 0,
	validationImages: 0,
	testImages: 0,
	numClasses: 0,
};

export const sampleImages: Array<{ id: number; name: string; label: string; confidence: number }> = [];

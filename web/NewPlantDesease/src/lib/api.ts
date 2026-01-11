export type DatasetStats = {
    generatedAt?: string;
    totalImages: number;
    trainImages: number;
    validationImages: number;
    testImages?: number;
    numClasses: number;
    imageQuality?: {
        overBrightImages?: {
            label: string;
            overBright: number;
            total: number;
            ratio: number;
        }[];
    };
};

export type ClassDistributionItem = {
    label: string;
    shortLabel: string;
    count: number;
};

export type ClassDistribution = {
    items: ClassDistributionItem[];
    totalImages: number;
    numClasses: number;
};

export type ModelMetrics = {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
    trainTime?: string;
    params?: string;
    valAccuracy?: number;
    epochs?: number;
};

export type ModelInfo = {
    id: string;
    name: string;
    pthPath?: string;
    metrics: ModelMetrics | null;
    hyperparameters: Record<string, unknown> | null;
};

export type ModelsIndex = {
    models: ModelInfo[];
};

export type ConfusionMatrixPayload = {
    modelId: string;
    labels: string[];
    matrix: number[][];
    normalized: boolean;
};

export type TrainingHistoryRow = {
    epoch: number;
    trainLoss: number;
    valLoss: number;
    trainAcc: number;
    valAcc: number;
    lr?: number;
};

export type TrainingHistoryPayload = {
    generatedAt?: string;
    modelId: string;
    timeTaken?: string;
    hyperparameters?: {
        model?: string;
        optimizer?: string;
        learningRate?: number;
        batchSize?: number;
        epochs?: number;
        imageSize?: string;
        dropout?: number;
        l2Regularization?: number;
        augmentation?: string[];
        [key: string]: unknown;
    };
    history: TrainingHistoryRow[];
};

export type RocPoint = {
    fpr: number;
    mobilenet?: number;
    resnet18?: number;
    efficientnet?: number;
};

export type RocMicroPayload = {
    points: RocPoint[];
    aucs: Record<string, Record<string, number>>;
};

export type ClassesPayload = {
    labels: string[];
    shortLabels?: string[];
    numClasses?: number;
};

export type PredictResult = {
    modelId: string;
    label: string;
    confidence: number;
    topK: { label: string; confidence: number }[];
};

export type ExplainabilityResult = {
    modelId: string;
    method: "saliency";
    target: { index: number; label: string };
    image: { width: number; height: number };
    heatmapPngBase64: string;
};

function apiBase() {
    return (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:5000";
}

async function getJson<T>(path: string): Promise<T> {
    const res = await fetch(`${apiBase()}${path}`);
    if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`Request failed ${res.status}: ${text}`);
    }
    return (await res.json()) as T;
}

export const api = {
    getDatasetStats: () => getJson<DatasetStats>("/api/data/dataset-stats"),
    getClassDistribution: () => getJson<ClassDistribution>("/api/data/class-distribution"),
    getModels: () => getJson<ModelsIndex>("/api/data/models"),
    getClasses: () => getJson<ClassesPayload>("/api/data/classes"),
    getRocMicro: () => getJson<RocMicroPayload>("/api/data/roc-micro"),
    getConfusionMatrix: (modelId: string) => getJson<ConfusionMatrixPayload>(`/api/data/models/${modelId}/confusion-matrix`),
    getTrainingHistory: (modelId: string) => getJson<TrainingHistoryPayload>(`/api/data/models/${modelId}/training-history`),
    predict: async (modelId: string, file: File, topK = 5): Promise<PredictResult> => {
        const form = new FormData();
        form.append("file", file);
        const res = await fetch(`${apiBase()}/api/predict?model=${encodeURIComponent(modelId)}&top_k=${encodeURIComponent(String(topK))}`, {
            method: "POST",
            body: form,
        });
        if (!res.ok) {
            const text = await res.text().catch(() => "");
            throw new Error(`Predict failed ${res.status}: ${text}`);
        }
        return (await res.json()) as PredictResult;
    },

    explain: async (modelId: string, file: File, method: "saliency" = "saliency"): Promise<ExplainabilityResult> => {
        const form = new FormData();
        form.append("file", file);
        const res = await fetch(
            `${apiBase()}/api/explain?model=${encodeURIComponent(modelId)}&method=${encodeURIComponent(method)}`,
            { method: "POST", body: form }
        );
        if (!res.ok) {
            const text = await res.text().catch(() => "");
            throw new Error(`Explain failed ${res.status}: ${text}`);
        }
        return (await res.json()) as ExplainabilityResult;
    },
};

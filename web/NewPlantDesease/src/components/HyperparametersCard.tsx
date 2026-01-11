import { motion } from "framer-motion";
import {
    Brain,
    Zap,
    TrendingUp,
    Package,
    RefreshCw,
    ImageIcon,
    Droplet,
    SlidersHorizontal,
    Clock,
    Hash,
    Target,
    Settings
} from "lucide-react";

export type Hyperparameters = {
    model?: string;
    optimizer?: string;
    learningRate?: number;
    batchSize?: number;
    epochs?: number;
    imageSize?: string;
    augmentation?: string[];
    dropout?: number;
    l2Regularization?: number;
    earlyStopping?: { monitor?: string; patience?: number; minDelta?: number };
};

export type ModelStats = {
    trainingTime?: string;
    totalParams?: string;
    trainableParams?: string;
};

export function HyperparametersCard({
    hyperparameters,
}: {
    hyperparameters: Hyperparameters | null;
}) {
    const hp = hyperparameters ?? {};
    const params = [
        { label: "Model", value: hp.model ?? "-", Icon: Brain },
        { label: "Optimizer", value: hp.optimizer ?? "-", Icon: Zap },
        { label: "Learning Rate", value: hp.learningRate != null ? String(hp.learningRate) : "-", Icon: TrendingUp },
        { label: "Batch Size", value: hp.batchSize != null ? String(hp.batchSize) : "-", Icon: Package },
        { label: "Epochs", value: hp.epochs != null ? String(hp.epochs) : "-", Icon: RefreshCw },
        { label: "Image Size", value: hp.imageSize ?? "-", Icon: ImageIcon },
        { label: "Dropout", value: hp.dropout != null ? String(hp.dropout) : "-", Icon: Droplet },
        { label: "L2 Reg", value: hp.l2Regularization != null ? String(hp.l2Regularization) : "-", Icon: SlidersHorizontal },
    ];

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="glass-card p-8 h-full"
        >
            <div className="flex items-center gap-3 mb-6">
                <Settings className="w-5 h-5" style={{ color: '#2596be' }} />
                <h3 className="text-lg font-bold" style={{ color: '#2596be' }}>
                    Siêu tham số
                </h3>
            </div>

            <div className="grid grid-cols-2 gap-4">
                {params.map((param, index) => {
                    const Icon = param.Icon;
                    return (
                        <motion.div
                            key={param.label}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.03, duration: 0.3 }}
                            className="p-4 rounded-xl bg-gradient-to-br from-gray-50 to-white border border-gray-100"
                        >
                            <div className="flex items-center gap-2 mb-1">
                                <Icon className="w-3.5 h-3.5" style={{ color: '#2596be' }} />
                                <span className="text-xs text-gray-500">{param.label}</span>
                            </div>
                            <p className="text-sm font-semibold text-gray-800 pl-5">{param.value}</p>
                        </motion.div>
                    );
                })}
            </div>

            {/* Augmentation techniques */}
            <div className="mt-6">
                <h4 className="text-xs font-medium text-gray-500 mb-2 flex items-center gap-1">
                    <RefreshCw className="w-3 h-3" style={{ color: '#2596be' }} />
                    Data Augmentation
                </h4>
                <div className="flex flex-wrap gap-1.5">
                    {(hp.augmentation ?? []).map((aug) => (
                        <span
                            key={aug}
                            className="px-2 py-1 rounded-full text-xs border"
                            style={{
                                background: 'rgba(37, 150, 190, 0.08)',
                                borderColor: 'rgba(37, 150, 190, 0.2)',
                                color: '#2596be'
                            }}
                        >
                            {aug}
                        </span>
                    ))}
                    {!hp.augmentation?.length && (
                        <span className="text-xs text-gray-400">-</span>
                    )}
                </div>
            </div>
        </motion.div>
    );
}

export function ModelStatsCard({
    stats,
    hyperparameters,
}: {
    stats: ModelStats | null;
    hyperparameters: Hyperparameters | null;
}) {
    const st = stats ?? {};
    const hp = hyperparameters ?? {};
    const rows = [
        { label: "Training Time", value: st.trainingTime ?? "-", Icon: Clock },
        { label: "Total Parameters", value: st.totalParams ?? "-", Icon: Hash },
        { label: "Trainable Params", value: st.trainableParams ?? "-", Icon: Target },
    ];

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="glass-card p-8 h-full"
        >
            <div className="flex items-center gap-3 mb-6">
                <TrendingUp className="w-5 h-5" style={{ color: '#2596be' }} />
                <h3 className="text-lg font-bold" style={{ color: '#2596be' }}>
                    Model Statistics
                </h3>
            </div>

            <div className="space-y-4">
                {rows.map((stat, index) => {
                    const Icon = stat.Icon;
                    return (
                        <motion.div
                            key={stat.label}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.2 + index * 0.1 }}
                            className="p-4 rounded-xl bg-gradient-to-r from-blue-50 to-cyan-50 border border-blue-100 mgb-5"
                        >
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <Icon className="w-5 h-5" style={{ color: '#2596be' }} />
                                    <span className="text-sm text-gray-600">{stat.label}</span>
                                </div>
                                <span className="text-xl font-bold" style={{ color: '#2596be' }}>
                                    {stat.value}
                                </span>
                            </div>
                        </motion.div>
                    );
                })}
            </div>

            {/* Early Stopping Info */}
            <div className="mt-6 p-5 rounded-xl border border-dashed border-gray-200 bg-gray-50">
                <h4 className="text-xs font-medium text-gray-500 mb-2 flex items-center gap-1">
                    <Target className="w-3 h-3" style={{ color: '#2596be' }} />
                    Early Stopping
                </h4>
                <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>
                        <span className="text-gray-400">Monitor:</span>
                        <p className="font-medium text-gray-700">{hp.earlyStopping?.monitor ?? "-"}</p>
                    </div>
                    <div>
                        <span className="text-gray-400">Patience:</span>
                        <p className="font-medium text-gray-700">{hp.earlyStopping?.patience ?? "-"}</p>
                    </div>
                    <div>
                        <span className="text-gray-400">Min Delta:</span>
                        <p className="font-medium text-gray-700">{hp.earlyStopping?.minDelta ?? "-"}</p>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

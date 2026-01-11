import { useState, useCallback, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import {
    Upload,
    X,
    Loader2,
    CheckCircle,
    AlertCircle,
    Leaf,
    Image as ImageIcon
} from "lucide-react";

import { api, type ExplainabilityResult } from "../lib/api";

function cn(...inputs: (string | undefined | null | false)[]) {
    return twMerge(clsx(inputs));
}

interface PredictionResult {
    label: string;
    confidence: number;
    topK: { label: string; confidence: number }[];
}

export function ModelTesting({
    defaultModelId,
    models,
}: {
    defaultModelId: string;
    models?: { id: string; name: string }[];
}) {
    const modelOptions = useMemo(() => {
        const incoming = (models ?? []).filter((m) => m?.id);
        if (incoming.length > 0) return incoming;
        return [
            { id: "mobilenet", name: "MobileNetV3 Large" },
            { id: "resnet18", name: "ResNet18" },
            { id: "efficientnet", name: "EfficientNet-B0" },
        ];
    }, [models]);

    const [activeModelId, setActiveModelId] = useState(defaultModelId);
    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [prediction, setPrediction] = useState<PredictionResult | null>(null);
    const [explain, setExplain] = useState<ExplainabilityResult | null>(null);
    const [isExplaining, setIsExplaining] = useState(false);
    const [explainError, setExplainError] = useState<string | null>(null);
    const [heatmapOpacity, setHeatmapOpacity] = useState(0.55);
    const [isDragOver, setIsDragOver] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        setActiveModelId(defaultModelId);
    }, [defaultModelId]);

    const handleFileSelect = useCallback((file: File) => {
        if (file && file.type.startsWith("image/")) {
            setSelectedImage(file);
            const reader = new FileReader();
            reader.onload = (e) => {
                setImagePreview(e.target?.result as string);
            };
            reader.readAsDataURL(file);
            setPrediction(null);
            setExplain(null);
            setExplainError(null);
        }
    }, []);

    const handleDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault();
            setIsDragOver(false);
            const file = e.dataTransfer.files[0];
            if (file) handleFileSelect(file);
        },
        [handleFileSelect]
    );

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(true);
    }, []);

    const handleDragLeave = useCallback(() => {
        setIsDragOver(false);
    }, []);

    const handleAnalyze = useCallback(() => {
        if (!selectedImage) return;

        setIsAnalyzing(true);
        setError(null);
        setExplain(null);
        setExplainError(null);
        setIsExplaining(false);

        api.predict(activeModelId, selectedImage, 5)
            .then((res) => {
                setPrediction({ label: res.label, confidence: res.confidence, topK: res.topK });
                setIsExplaining(true);
                return api
                    .explain(activeModelId, selectedImage, "saliency")
                    .then((ex) => setExplain(ex))
                    .catch((e: unknown) => {
                        const message = e instanceof Error ? e.message : "Explain failed";
                        setExplainError(message);
                    })
                    .finally(() => setIsExplaining(false));
            })
            .catch((e: unknown) => {
                setExplain(null);
                const message = e instanceof Error ? e.message : "Predict failed";
                setPrediction(null);
                setError(message);
            })
            .finally(() => setIsAnalyzing(false));
    }, [selectedImage, activeModelId]);

    const handleReset = useCallback(() => {
        setSelectedImage(null);
        setImagePreview(null);
        setPrediction(null);
        setExplain(null);
        setExplainError(null);
        setError(null);
    }, []);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="glass-card p-10"
        >
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                {/* Upload section */}
                <div className="pl-2">
                    <div className="mb-5">
                        <h4 className="text-sm font-medium text-gray-600 mb-4">Test Mode</h4>
                        <div className="flex flex-wrap gap-2">
                            {modelOptions.map((m) => {
                                const isActive = m.id === activeModelId;
                                return (
                                    <button
                                        key={m.id}
                                        type="button"
                                        onClick={() => setActiveModelId(m.id)}
                                        className={cn(
                                            "px-4 py-2 rounded-full text-sm font-medium transition-all",
                                            isActive
                                                ? "text-white shadow-md"
                                                : "text-gray-600 bg-gray-100 hover:bg-gray-200"
                                        )}
                                        style={isActive ? { background: "linear-gradient(135deg, #2596be, #3db8e5)" } : {}}
                                    >
                                        {m.name}
                                    </button>
                                );
                            })}
                        </div>
                        <p className="text-xs text-gray-400 mt-2">
                            Chọn model: <span className="font-medium">{activeModelId}</span>
                        </p>
                    </div>

                    <h4 className="text-sm font-medium text-gray-600 mb-4 flex items-center gap-2">
                        <Upload className="w-4 h-4" style={{ color: '#2596be' }} />
                        Tải ảnh lên
                    </h4>

                    {/* Drop zone */}
                    <motion.div
                        className={cn(
                            "upload-zone p-8 text-center cursor-pointer transition-all duration-300",
                            isDragOver && "drag-over"
                        )}
                        onDrop={handleDrop}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onClick={() => document.getElementById("file-input")?.click()}
                        whileHover={{ scale: 1.01 }}
                        whileTap={{ scale: 0.99 }}
                    >
                        <input
                            id="file-input"
                            type="file"
                            accept="image/*"
                            className="hidden"
                            onChange={(e) => {
                                const file = e.target.files?.[0];
                                if (file) handleFileSelect(file);
                            }}
                        />

                        {imagePreview ? (
                            <div className="relative">
                                <img
                                    src={imagePreview}
                                    alt="Preview"
                                    className="max-h-48 mx-auto rounded-lg shadow-lg"
                                />
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleReset();
                                    }}
                                    className="absolute -top-2 -right-2 w-8 h-8 bg-red-500 rounded-full flex items-center justify-center text-white hover:bg-red-600 transition-colors"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                        ) : (
                            <div>
                                <Upload className="w-12 h-12 block mx-auto mb-4 text-gray-400" />
                                <p className="text-gray-600 mb-2">Tải ảnh vào đây</p>
                                <p className="text-sm text-gray-400">hoặc click để duyệt</p>
                            </div>
                        )}
                    </motion.div>

                    {/* Analyze button */}
                    <motion.button
                        onClick={handleAnalyze}
                        disabled={!imagePreview || isAnalyzing}
                        className={cn(
                            "w-full mt-4 py-3 px-6 rounded-xl font-medium transition-all duration-300 flex items-center justify-center gap-2",
                            imagePreview && !isAnalyzing
                                ? "text-white hover:shadow-lg"
                                : "bg-gray-100 text-gray-400 cursor-not-allowed"
                        )}
                        style={imagePreview && !isAnalyzing ? { background: 'linear-gradient(135deg, #2596be, #3db8e5)' } : {}}
                        whileHover={imagePreview && !isAnalyzing ? { scale: 1.02 } : {}}
                        whileTap={imagePreview && !isAnalyzing ? { scale: 0.98 } : {}}
                    >
                        {isAnalyzing ? (
                            <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                Chạy chạy chạy...
                            </>
                        ) : (
                            "Thực hiện"
                        )}
                    </motion.button>

                </div>

                {/* Results section */}
                <div>
                    <h4 className="text-sm font-medium text-gray-600 mb-3">
                        Kết quả dự đoán
                    </h4>

                    {error && (
                        <div className="mb-4 p-4 rounded-xl border border-red-200 bg-red-50 flex items-start gap-3">
                            <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
                            <div className="text-sm text-red-700 break-words">{error}</div>
                        </div>
                    )}

                    <AnimatePresence mode="wait">
                        {prediction ? (
                            <motion.div
                                key="results"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -20 }}
                                className="space-y-4"
                            >
                                {/* Main prediction */}
                                <div
                                    className="glass-card p-8 relative overflow-hidden"
                                    style={{ background: 'linear-gradient(135deg, rgba(37, 150, 190, 0.05), rgba(61, 184, 229, 0.05))' }}
                                >
                                    <p className="text-sm text-gray-500 mb-1">Lớp dự đoán</p>
                                    <h3 className="text-xl font-bold mb-2" style={{ color: '#2596be' }}>
                                        {prediction.label.replace(/___/g, ' - ').replace(/_/g, ' ')}
                                    </h3>
                                    <div className="flex items-center gap-2">
                                        <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                                            <motion.div
                                                initial={{ width: 0 }}
                                                animate={{ width: `${prediction.confidence * 100}%` }}
                                                transition={{ duration: 0.5, delay: 0.2 }}
                                                className="h-full rounded-full"
                                                style={{ background: 'linear-gradient(90deg, #2596be, #3db8e5)' }}
                                            />
                                        </div>
                                        <span className="text-sm font-medium" style={{ color: '#2596be' }}>
                                            {(prediction.confidence * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>

                                {/* Top-K predictions */}
                                <div className="glass-card p-6">
                                    <p className="text-sm text-gray-500 mb-3">Tỉ lệ dự đoán</p>
                                    <div className="space-y-2">
                                        {prediction.topK.map((item, index) => (
                                            <motion.div
                                                key={item.label}
                                                initial={{ opacity: 0, x: -10 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: 0.3 + index * 0.1 }}
                                                className="flex items-center gap-3"
                                            >
                                                <span className="w-6 text-center text-sm text-gray-400">
                                                    {index + 1}
                                                </span>
                                                <div className="flex-1">
                                                    <div className="flex justify-between items-center mb-1">
                                                        <span className="text-sm text-gray-700">
                                                            {item.label.replace(/___/g, ' - ').replace(/_/g, ' ')}
                                                        </span>
                                                        <span className="text-xs text-gray-400">
                                                            {(item.confidence * 100).toFixed(1)}%
                                                        </span>
                                                    </div>
                                                    <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                                                        <motion.div
                                                            initial={{ width: 0 }}
                                                            animate={{ width: `${item.confidence * 100}%` }}
                                                            transition={{ duration: 0.5, delay: 0.4 + index * 0.1 }}
                                                            className={cn("h-full rounded-full")}
                                                            style={{ background: index === 0 ? 'linear-gradient(90deg, #2596be, #3db8e5)' : '#d1d5db' }}
                                                        />
                                                    </div>
                                                </div>
                                            </motion.div>
                                        ))}
                                    </div>
                                </div>

                                {/* Explainability / saliency */}
                                <div className="glass-card p-6">
                                    <div className="flex items-center justify-between gap-4 mb-3">
                                        <div>
                                            <p className="text-sm text-gray-500">Phân tích dự đoán</p>
                                            <p className="text-xs text-gray-400 mt-1">
                                                Vùng màu nóng là vùng ảnh ảnh hưởng mạnh tới dự đoán.
                                            </p>
                                        </div>

                                        <div className="flex items-center gap-3">
                                            <span className="text-xs text-gray-400">Opacity</span>
                                            <input
                                                type="range"
                                                min={0}
                                                max={100}
                                                value={Math.round(heatmapOpacity * 100)}
                                                onChange={(e) => setHeatmapOpacity(Number(e.target.value) / 100)}
                                                className="w-28"
                                            />
                                        </div>
                                    </div>

                                    {explainError && (
                                        <div className="mb-3 p-3 rounded-lg border border-red-200 bg-red-50 text-sm text-red-700 break-words">
                                            {explainError}
                                        </div>
                                    )}

                                    {!imagePreview ? (
                                        <div className="text-sm text-gray-500">Tải ảnh lên để thực hiện phân tích</div>
                                    ) : explain ? (
                                        <div className="space-y-3">
                                            <div className="text-xs text-gray-500">
                                                Target: <span className="font-medium">{explain.target.label.replace(/___/g, ' - ').replace(/_/g, ' ')}</span>
                                            </div>

                                            <div className="relative w-full max-w-md mx-auto rounded-xl overflow-hidden shadow-sm border border-gray-100">
                                                <img src={imagePreview} alt="Input" className="w-full h-auto block" />
                                                <img
                                                    src={`data:image/png;base64,${explain.heatmapPngBase64}`}
                                                    alt="Saliency heatmap"
                                                    className="absolute inset-0 w-full h-full object-cover"
                                                    style={{ opacity: heatmapOpacity, mixBlendMode: "multiply" }}
                                                />
                                            </div>
                                        </div>
                                    ) : isAnalyzing || isExplaining ? (
                                        <div className="flex items-center gap-2 text-sm text-gray-500">
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                            Chạy chạy chạy...
                                        </div>
                                    ) : (
                                        <div className="text-sm text-gray-500">Click Thực hiện để tạo bản đồ saliency.</div>
                                    )}
                                </div>

                                {/* Confidence indicator */}
                                <div className="glass-card p-5 flex items-center justify-between">
                                    <span className="text-sm text-gray-600">Mức độ tin cậy</span>
                                    <span
                                        className={cn(
                                            "px-3 py-1 rounded-full text-sm font-medium flex items-center gap-1",
                                            prediction.confidence >= 0.9
                                                ? "bg-green-50 text-green-600"
                                                : prediction.confidence >= 0.7
                                                    ? "bg-yellow-50 text-yellow-600"
                                                    : "bg-red-50 text-red-600"
                                        )}
                                    >
                                        {prediction.confidence >= 0.9 ? (
                                            <CheckCircle className="w-4 h-4" />
                                        ) : (
                                            <AlertCircle className="w-4 h-4" />
                                        )}
                                        {prediction.confidence >= 0.9
                                            ? "High Confidence"
                                            : prediction.confidence >= 0.7
                                                ? "Moderate Confidence"
                                                : "Low Confidence"}
                                    </span>
                                </div>
                            </motion.div>
                        ) : (
                            <motion.div
                                key="placeholder"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="glass-card p-12 text-center"
                            >
                                <Leaf className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                                <p className="text-gray-500">
                                    Tải lên hoặc chọn một ảnh để xem dự đoán
                                </p>
                                {/* <p className="text-sm text-gray-400 mt-2">
                                    Hỗ trợ định dạng JPEG, PNG và WebP
                                </p> */}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </motion.div>
    );
}

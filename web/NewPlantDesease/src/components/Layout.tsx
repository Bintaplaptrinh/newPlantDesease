import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    BarChart3,
    Brain,
    TestTube2,
    Leaf,
    Image,
    Tag,
    Target,
    TrendingUp,
    ChevronLeft,
    ChevronRight,
    Grid3X3,
    LineChart,
    Info,
    Upload
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { NavLeft } from "./NavLeft";
import { ConfusionMatrix3D } from "./ConfusionMatrix3D";
import { AccuracyChart, MultiMetricChart } from "./AccuracyChart";
import { HyperparametersCard } from "./HyperparametersCard";
import { ModelTesting } from "./ModelTesting";
import { TrainingHistoryChart } from "./TrainingHistoryChart";
import { ClassDistributionChart } from "./ClassDistributionChart";
import { OverBrightImagesChart } from "./OverBrightImagesChart";
import { RocCurveComparison } from "./RocCurveComparison";
import { PerClassAccuracyChart } from "./PerClassAccuracyChart";
import { api, type DatasetStats, type ModelInfo, type RocPoint, type TrainingHistoryPayload } from "../lib/api";


// Section definitions with lucide icons
const MAIN_SECTIONS = [
    { id: "intro", label: "Introduction & Data", icon: BarChart3 },
    { id: "model", label: "Model & Statistics", icon: Brain },
    { id: "testing", label: "Model Testing", icon: TestTube2 },
];

// Sub-sections for each main section with icons
const SUB_SECTIONS: { [key: string]: { id: string; label: string; icon: LucideIcon }[] } = {
    intro: [
        { id: "overview", label: "Tổng quan", icon: Info },
        { id: "distribution", label: "Visualize Data", icon: BarChart3 },
    ],
    model: [
        { id: "overview", label: "Tổng quan", icon: Info },
        { id: "confusion", label: "Confusion Matrix", icon: Grid3X3 },
        { id: "statistics", label: "Phân tích", icon: LineChart },
        { id: "roc", label: "ROC-CURVE", icon: TrendingUp },
    ],
    testing: [
        { id: "upload", label: "Test Model", icon: Upload },
    ],
};

// Floating Bottom Slider Component
function FloatingBottomSlider({
    items,
    activeIndex,
    onSelect
}: {
    items: { id: string; label: string; icon: LucideIcon }[];
    activeIndex: number;
    onSelect: (index: number) => void;
}) {
    if (items.length <= 1) return null;

    return (
        <motion.div
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 100, opacity: 0 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50"
        >
            <div
                className="glass-nav rounded-full flex items-center gap-3 px-5 py-3.5 md:px-6 md:py-4"
            >
                <button
                    onClick={() => onSelect(Math.max(0, activeIndex - 1))}
                    disabled={activeIndex === 0}
                    className="p-2 rounded-full hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                >
                    <ChevronLeft className="w-5 h-5 text-gray-600" />
                </button>

                <div className="flex items-center gap-2">
                    {items.map((item, index) => {
                        const Icon = item.icon;
                        const isActive = activeIndex === index;
                        return (
                            <button
                                key={item.id}
                                onClick={() => onSelect(index)}
                                className={`flex items-center gap-2 px-5 py-3 rounded-full text-sm md:text-[15px] font-medium transition-all duration-300 ${isActive
                                    ? "text-white shadow-md"
                                    : "text-gray-600 hover:text-gray-800 hover:bg-gray-100"
                                    }`}
                                style={isActive ? {
                                    background: 'linear-gradient(135deg, #2596be, #3db8e5)',
                                    boxShadow: '0 4px 15px rgba(37, 150, 190, 0.4)',
                                } : {}}
                            >
                                <Icon className="w-4 h-4 shrink-0" />
                                <span>{item.label}</span>
                            </button>
                        );
                    })}
                </div>

                <button
                    onClick={() => onSelect(Math.min(items.length - 1, activeIndex + 1))}
                    disabled={activeIndex === items.length - 1}
                    className="p-2 rounded-full hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                >
                    <ChevronRight className="w-5 h-5 text-gray-600" />
                </button>

                {/* Page indicator dots */}
                <div className="flex items-center gap-1.5 ml-2 pl-3 border-l border-gray-200">
                    {items.map((_, index) => (
                        <button
                            key={index}
                            onClick={() => onSelect(index)}
                            className={`w-2 h-2 rounded-full transition-all duration-300 ${activeIndex === index
                                ? "w-6"
                                : "hover:bg-gray-400"
                                }`}
                            style={activeIndex === index ? { background: '#2596be' } : { background: '#d1d5db' }}
                        />
                    ))}
                </div>
            </div>
        </motion.div>
    );
}

// Model Selector Tabs
function ModelSelector({
    models,
    activeIndex,
    onSelect
}: {
    models: ModelInfo[];
    activeIndex: number;
    onSelect: (index: number) => void;
}) {
    return (
        <div className="flex items-center justify-center gap-2 mb-10 mt-6">
            {models.map((model, index) => (
                <button
                    key={model.id}
                    onClick={() => onSelect(index)}
                    className={`px-6 md:px-7 py-3 md:py-3.5 rounded-full text-sm md:text-[15px] font-medium transition-all duration-300 ${activeIndex === index
                        ? "text-white shadow-lg"
                        : "text-gray-600 hover:text-gray-800 bg-gray-100 hover:bg-gray-200"
                        }`}
                    style={activeIndex === index ? {
                        background: 'linear-gradient(135deg, #2596be, #3db8e5)',
                    } : {}}
                >
                    {model.name}
                </button>
            ))}
        </div>
    );
}

export function Layout() {
    const [activeSection, setActiveSection] = useState(0);
    const [introSubSection, setIntroSubSection] = useState(0);
    const [introDistributionSubTab, setIntroDistributionSubTab] = useState<"overall" | "overbright">("overall");
    const [modelSubSection, setModelSubSection] = useState(0);
    const [testingSubSection, setTestingSubSection] = useState(0);
    const [selectedModel, setSelectedModel] = useState(0);
    const containerRef = useRef<HTMLDivElement>(null);
    const sectionRefs = useRef<(HTMLElement | null)[]>([]);

    const [datasetStats, setDatasetStats] = useState<DatasetStats>({
        totalImages: 0,
        trainImages: 0,
        validationImages: 0,
        testImages: 0,
        numClasses: 0,
    });
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [classDistribution, setClassDistribution] = useState<{ shortName: string; count: number }[]>([]);
    const [rocPoints, setRocPoints] = useState<RocPoint[]>([]);
    const [confusionMatrix, setConfusionMatrix] = useState<number[][]>([]);
    const [confusionLabels, setConfusionLabels] = useState<string[]>([]);
    const [trainingHistory, setTrainingHistory] = useState<any[]>([]);
    const [trainingMeta, setTrainingMeta] = useState<Pick<TrainingHistoryPayload, "timeTaken" | "hyperparameters"> | null>(null);

    useEffect(() => {
        // Load dataset + model index + ROC + distribution
        Promise.all([
            api.getDatasetStats(),
            api.getModels(),
            api.getClassDistribution(),
            api.getRocMicro(),
        ])
            .then(([stats, modelIndex, dist, roc]) => {
                setDatasetStats({
                    totalImages: stats.totalImages ?? 0,
                    trainImages: stats.trainImages ?? 0,
                    validationImages: stats.validationImages ?? 0,
                    testImages: stats.testImages ?? 0,
                    numClasses: stats.numClasses ?? 0,
                    imageQuality: stats.imageQuality,
                });

                const list = modelIndex.models ?? [];
                setModels(list);

                // pick ResNet18 by default when present
                const resnetIdx = list.findIndex((m) => m.id === "resnet18");
                setSelectedModel(resnetIdx >= 0 ? resnetIdx : 0);

                setClassDistribution((dist.items ?? []).map((it) => ({ shortName: it.shortLabel, count: it.count })));
                setRocPoints(roc.points ?? []);
            })
            .catch(() => {
                // keep zero state; UI will show empty-state messaging
            });
    }, []);

    const currentModel = models[selectedModel];

    useEffect(() => {
        if (!currentModel?.id) {
            setConfusionMatrix([]);
            setConfusionLabels([]);
            setTrainingHistory([]);
            setTrainingMeta(null);
            return;
        }

        Promise.all([
            api.getConfusionMatrix(currentModel.id),
            api.getTrainingHistory(currentModel.id),
        ])
            .then(([cm, hist]) => {
                setConfusionMatrix(cm.matrix ?? []);
                setConfusionLabels(cm.labels ?? []);
                setTrainingHistory(hist.history ?? []);
                setTrainingMeta({ timeTaken: hist.timeTaken, hyperparameters: hist.hyperparameters });
            })
            .catch(() => {
                setConfusionMatrix([]);
                setConfusionLabels([]);
                setTrainingHistory([]);
                setTrainingMeta(null);
            });
    }, [currentModel?.id]);

    // Handle scroll to update active section
    const handleScroll = useCallback(() => {
        if (!containerRef.current) return;

        const scrollTop = containerRef.current.scrollTop;
        const sections = sectionRefs.current;

        // Pick the section whose top is closest to the container scrollTop.
        // This works well when sections have variable height and snapping is disabled.
        let bestIndex = activeSection;
        let bestDistance = Number.POSITIVE_INFINITY;

        for (let i = 0; i < sections.length; i++) {
            const el = sections[i];
            if (!el) continue;
            const distance = Math.abs(el.offsetTop - scrollTop);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = i;
            }
        }

        if (bestIndex !== activeSection && bestIndex >= 0 && bestIndex < MAIN_SECTIONS.length) {
            setActiveSection(bestIndex);
        }
    }, [activeSection]);

    useEffect(() => {
        const container = containerRef.current;
        if (container) {
            container.addEventListener("scroll", handleScroll);
            return () => container.removeEventListener("scroll", handleScroll);
        }
    }, [handleScroll]);

    // Navigate to section
    const navigateToSection = useCallback((index: number) => {
        const ref = sectionRefs.current[index];
        if (ref && containerRef.current) {
            ref.scrollIntoView({ behavior: "smooth", block: "start" });
            setActiveSection(index);
        }
    }, []);

    const bestAccuracy = Math.max(0, ...models.map((m) => m.metrics?.accuracy ?? 0));

    // Get current sub-section state based on active main section
    const getCurrentSubIndex = () => {
        switch (activeSection) {
            case 0: return introSubSection;
            case 1: return modelSubSection;
            case 2: return testingSubSection;
            default: return 0;
        }
    };

    const setCurrentSubIndex = (index: number) => {
        switch (activeSection) {
            case 0: setIntroSubSection(index); break;
            case 1: setModelSubSection(index); break;
            case 2: setTestingSubSection(index); break;
        }
    };

    return (
        <div className="relative min-h-screen bg-white">
            {/* Left Navigation */}
            <NavLeft
                activeSection={activeSection}
                onSectionChange={navigateToSection}
                sections={MAIN_SECTIONS}
            />

            {/* Floating Bottom Slider - Always visible */}
            <FloatingBottomSlider
                items={SUB_SECTIONS[MAIN_SECTIONS[activeSection].id]}
                activeIndex={getCurrentSubIndex()}
                onSelect={setCurrentSubIndex}
            />

            {/* Main scrollable container with scroll snap */}
            <div
                ref={containerRef}
                className="h-screen overflow-y-auto bg-white px-6 md:px-10 lg:px-14"
                style={{ scrollBehavior: 'smooth' }}
            >
                {/* ==================== SECTION 1: Introduction & Data ==================== */}
                <section
                    ref={(el) => { sectionRefs.current[0] = el }}
                    id="intro"
                    className="min-h-screen py-16 flex flex-col items-center"
                >
                    <div className="w-full max-w-5xl mx-auto flex-1 pb-24">
                        {/* Header */}
                        <motion.header className="text-center mb-14 mt-4">
                            <motion.h1
                                initial={{ opacity: 0, y: -20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.5 }}
                                className="text-4xl md:text-5xl font-bold mb-3"
                                style={{
                                    color: '#2596be',
                                    textShadow: '0 4px 20px rgba(37, 150, 190, 0.25)',
                                }}
                            >
                                New Plant Disease Classification Report
                            </motion.h1>
                            <motion.p
                                initial={{ opacity: 0, y: -10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.1 }}
                                className="text-lg text-gray-600"
                            >
                                
                            </motion.p>
                        </motion.header>

                        {/* Sub-section Content */}
                        <AnimatePresence mode="wait">
                            {introSubSection === 0 ? (
                                <motion.div
                                    key="intro-overview"
                                    initial={{ opacity: 0, x: -30 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 30 }}
                                    transition={{ duration: 0.3 }}
                                    className="space-y-10"
                                >
                                    {/* Overview Card */}
                                    <div className="glass-card p-10 mt-6" style={{ border: 'none' }}>
                                        <div className="flex items-center gap-3 mb-8">
                                            <Leaf className="w-8 h-8" style={{ color: '#2596be' }} />
                                            <h2 className="text-2xl font-bold" style={{ color: '#2596be' }}>
                                                Tổng quan về dự án
                                            </h2>
                                        </div>

                                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                                            <div className="space-y-6">
                                                <p className="text-gray-700 leading-relaxed text-lg">
                                                    Dự án này triển khai một mô hình học máy để tự động phát hiện và phân loại bệnh trên cây trồng thông qua hình ảnh lá.
                                                    Mô hình được huấn luyện trên bộ dữ liệu New Plant Desease toàn diện bao gồm 38 lớp khác nhau.
                                                </p>
                                                <p className="text-gray-700 leading-relaxed text-lg">
                                                    Bằng cách sử dụng học chuyển giao với nhiều kiến trúc backbone (MobileNet, ResNet18, EfficientNet-B0), 
                                                    đạt được hiệu suất tiên tiến với độ chính xác trên 94% trên validation dataset.
                                                </p>
                                            </div>

                                            <div className="grid grid-cols-2 gap-8">
                                                <div className="glass-card p-8 text-center">
                                                    <Image className="w-8 h-8 mx-auto mb-3" style={{ color: '#2596be' }} />
                                                    <p className="text-3xl font-bold" style={{ color: '#2596be' }}>
                                                        {datasetStats.totalImages.toLocaleString()}
                                                    </p>
                                                    <p className="text-sm text-gray-500 mt-1">Tổng ảnh</p>
                                                </div>
                                                <div className="glass-card p-8 text-center">
                                                    <Tag className="w-8 h-8 mx-auto mb-3" style={{ color: '#2596be' }} />
                                                    <p className="text-3xl font-bold" style={{ color: '#2596be' }}>
                                                        {datasetStats.numClasses}
                                                    </p>
                                                    <p className="text-sm text-gray-500 mt-1">Só lớp</p>
                                                </div>
                                                <div className="glass-card p-8 text-center">
                                                    <Target className="w-8 h-8 mx-auto mb-3" style={{ color: '#2596be' }} />
                                                    <p className="text-3xl font-bold" style={{ color: '#2596be' }}>
                                                        {(bestAccuracy * 100).toFixed(1)}%
                                                    </p>
                                                    <p className="text-sm text-gray-500 mt-1">Best Accuracy</p>
                                                </div>
                                                <div className="glass-card p-8 text-center">
                                                    <TrendingUp className="w-8 h-8 mx-auto mb-3" style={{ color: '#2596be' }} />
                                                    <p className="text-3xl font-bold" style={{ color: '#2596be' }}>
                                                        {models.length}
                                                    </p>
                                                    <p className="text-sm text-gray-500 mt-1">Models</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Dataset Split */}
                                    <div className="grid grid-cols-3 gap-8 mt-6">
                                        {[
                                            { label: "Training Set", count: datasetStats.trainImages },
                                            { label: "Validation Set", count: datasetStats.validationImages },
                                            { label: "Test Set", count: datasetStats.testImages ?? 0 },
                                        ].map((split) => (
                                            <div key={split.label} className="glass-card p-8 text-center">
                                                <p className="text-sm text-gray-500 mb-2">{split.label}</p>
                                                <p className="text-2xl font-bold" style={{ color: '#2596be' }}>
                                                    {split.count.toLocaleString()}
                                                </p>
                                                <p className="text-xs text-gray-400 mt-1">ảnh</p>
                                            </div>
                                        ))}
                                    </div>
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="intro-distribution"
                                    initial={{ opacity: 0, x: 30 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -30 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <div className="flex items-center justify-center gap-2 mb-10 mt-2">
                                        <button
                                            onClick={() => setIntroDistributionSubTab("overall")}
                                            className={`px-6 md:px-7 py-3 md:py-3.5 rounded-full text-sm md:text-[15px] font-medium transition-all duration-300 ${introDistributionSubTab === "overall"
                                                ? "text-white shadow-lg"
                                                : "text-gray-600 hover:text-gray-800 bg-gray-100 hover:bg-gray-200"
                                                }`}
                                            style={introDistributionSubTab === "overall" ? {
                                                background: 'linear-gradient(135deg, #2596be, #3db8e5)',
                                            } : {}}
                                        >
                                            Tổng thể
                                        </button>
                                        <button
                                            onClick={() => setIntroDistributionSubTab("overbright")}
                                            className={`px-6 md:px-7 py-3 md:py-3.5 rounded-full text-sm md:text-[15px] font-medium transition-all duration-300 ${introDistributionSubTab === "overbright"
                                                ? "text-white shadow-lg"
                                                : "text-gray-600 hover:text-gray-800 bg-gray-100 hover:bg-gray-200"
                                                }`}
                                            style={introDistributionSubTab === "overbright" ? {
                                                background: 'linear-gradient(135deg, #2596be, #3db8e5)',
                                            } : {}}
                                        >
                                            Ảnh quá sáng
                                        </button>
                                    </div>

                                    {introDistributionSubTab === "overall" ? (
                                        <ClassDistributionChart data={classDistribution} />
                                    ) : (
                                        <div className="space-y-10">
                                            {datasetStats.imageQuality?.overBrightImages?.length ? (
                                                <OverBrightImagesChart rows={datasetStats.imageQuality.overBrightImages} topN={11} />
                                            ) : (
                                                <div className="glass-card p-10">
                                                    <p className="text-sm text-gray-500">
                                                        No over-brightness data found in <code>dataset_stats.json</code>.
                                                    </p>
                                                </div>
                                            )}

                                            <div className="glass-card p-10">
                                                <div className="flex items-center gap-3 mb-6">
                                                    <Info className="w-6 h-6" style={{ color: '#2596be' }} />
                                                    <h3 className="text-xl font-bold" style={{ color: '#2596be' }}>
                                                        Thống kê
                                                    </h3>
                                                </div>

                                                {datasetStats.imageQuality?.overBrightImages?.length ? (
                                                    <div className="overflow-x-auto">
                                                        <table className="w-full text-left">
                                                            <thead>
                                                                <tr className="text-xs text-gray-500 border-b border-gray-100">
                                                                    <th className="py-2 pr-3">Lớp</th>
                                                                    <th className="py-2 pr-3">Ảnh quá sáng</th>
                                                                    <th className="py-2 pr-3">Tổng số</th>
                                                                    <th className="py-2">Tỷ lệ</th>
                                                                </tr>
                                                            </thead>
                                                            <tbody>
                                                                {datasetStats.imageQuality.overBrightImages.slice(0, 11).map((row) => (
                                                                    <tr key={row.label} className="border-b border-gray-50 text-sm">
                                                                        <td className="py-2 pr-3 text-gray-700">
                                                                            {row.label.replace(/___/g, ' - ').replace(/_/g, ' ')}
                                                                        </td>
                                                                        <td className="py-2 pr-3 text-gray-700">{row.overBright.toLocaleString()}</td>
                                                                        <td className="py-2 pr-3 text-gray-700">{row.total.toLocaleString()}</td>
                                                                        <td className="py-2 text-gray-700">{(row.ratio * 100).toFixed(1)}%</td>
                                                                    </tr>
                                                                ))}
                                                            </tbody>
                                                        </table>
                                                    </div>
                                                ) : (
                                                    <p className="text-sm text-gray-500">
                                                        No over-brightness data found in <code>dataset_stats.json</code>.
                                                    </p>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </section>

                {/* ==================== SECTION 2: Model & Statistics ==================== */}
                <section
                    ref={(el) => { sectionRefs.current[1] = el }}
                    id="model"
                    className="min-h-screen py-16 flex flex-col items-center"
                >
                    <div className="w-full max-w-5xl mx-auto pb-24">
                        {/* Section Header */}
                        <div className="flex items-center justify-center gap-4 mb-10 mt-4">
                            <Brain className="w-10 h-10" style={{ color: '#2596be' }} />
                            <h2 className="text-3xl font-bold" style={{ color: '#2596be' }}>
                                Model & Statistics
                            </h2>
                        </div>

                        {/* Model Selector */}
                        <ModelSelector
                            models={models}
                            activeIndex={selectedModel}
                            onSelect={setSelectedModel}
                        />

                        {/* Sub-section Content */}
                        <AnimatePresence mode="wait">
                            {modelSubSection === 0 ? (
                                // Overview - Model Info
                                <motion.div
                                    key="model-overview"
                                    initial={{ opacity: 0, x: -30 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 30 }}
                                    transition={{ duration: 0.3 }}
                                    className="space-y-10"
                                >
                                    {/* Model Quick Stats */}
                                    <div className="glass-card p-10">
                                        <h3 className="text-xl font-bold mb-8" style={{ color: '#2596be' }}>
                                            {currentModel?.name ?? "Model"} Performance
                                        </h3>
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                                            <div className="text-center p-6 rounded-xl bg-gradient-to-br from-blue-50 to-cyan-50">
                                                <p className="text-3xl font-bold" style={{ color: '#2596be' }}>
                                                    {currentModel?.metrics?.accuracy != null ? `${(currentModel.metrics.accuracy * 100).toFixed(1)}%` : "-"}
                                                </p>
                                                <p className="text-sm text-gray-500 mt-1">Train Accuracy</p>
                                            </div>
                                            <div className="text-center p-6 rounded-xl bg-gradient-to-br from-blue-50 to-cyan-50">
                                                <p className="text-3xl font-bold" style={{ color: '#2596be' }}>
                                                    {currentModel?.metrics?.valAccuracy != null ? `${(currentModel.metrics.valAccuracy * 100).toFixed(1)}%` : "-"}
                                                </p>
                                                <p className="text-sm text-gray-500 mt-1">Val Accuracy</p>
                                            </div>
                                            <div className="text-center p-6 rounded-xl bg-gradient-to-br from-blue-50 to-cyan-50">
                                                <p className="text-3xl font-bold" style={{ color: '#2596be' }}>
                                                    {currentModel?.metrics?.f1Score != null ? `${(currentModel.metrics.f1Score * 100).toFixed(1)}%` : "-"}
                                                </p>
                                                <p className="text-sm text-gray-500 mt-1">F1 Score</p>
                                            </div>
                                            <div className="text-center p-6 rounded-xl bg-gradient-to-br from-blue-50 to-cyan-50">
                                                <p className="text-3xl font-bold" style={{ color: '#2596be' }}>
                                                    {trainingMeta?.timeTaken ?? currentModel?.metrics?.trainTime ?? "-"}
                                                </p>
                                                <p className="text-sm text-gray-500 mt-1">Train Time</p>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Model Accuracy + Performance Metrics */}
                                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                                        <div className="flex justify-center">
                                            <AccuracyChart accuracy={currentModel?.metrics?.accuracy ?? 0} />
                                        </div>
                                        <MultiMetricChart
                                            metrics={[
                                                { name: "Precision", value: currentModel?.metrics?.precision ?? 0, color: "#2596be" },
                                                { name: "Recall", value: currentModel?.metrics?.recall ?? 0, color: "#3db8e5" },
                                                { name: "F1 Score", value: currentModel?.metrics?.f1Score ?? 0, color: "#1a7a9c" },
                                                { name: "Accuracy", value: currentModel?.metrics?.accuracy ?? 0, color: "#5cc8f0" },
                                            ]}
                                        />
                                    </div>

                                    {/* Hyperparameters */}
                                    <HyperparametersCard hyperparameters={(trainingMeta?.hyperparameters as any) ?? currentModel?.hyperparameters ?? null} />
                                </motion.div>
                            ) : modelSubSection === 1 ? (
                                // Confusion Matrix
                                <motion.div
                                    key="model-confusion"
                                    initial={{ opacity: 0, x: 30 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -30 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <ConfusionMatrix3D data={confusionMatrix} labels={confusionLabels} />
                                </motion.div>
                            ) : modelSubSection === 2 ? (
                                // Statistics - Training History + Per-class accuracy
                                <motion.div
                                    key="model-statistics"
                                    initial={{ opacity: 0, x: 30 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -30 }}
                                    transition={{ duration: 0.3 }}
                                    className="space-y-10"
                                >
                                    <TrainingHistoryChart data={trainingHistory} />
                                    <PerClassAccuracyChart
                                        confusionMatrix={confusionMatrix}
                                        labels={confusionLabels}
                                        title="Accuracy của từng lớp"
                                    />
                                </motion.div>
                            ) : modelSubSection === 3 ? (
                                // ROC Curve comparison
                                <motion.div
                                    key="model-roc"
                                    initial={{ opacity: 0, x: 30 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -30 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <RocCurveComparison data={rocPoints} />
                                </motion.div>
                            ) : null}
                        </AnimatePresence>
                    </div>
                </section>

                {/* ==================== SECTION 3: Model Testing ==================== */}
                <section
                    ref={(el) => { sectionRefs.current[2] = el }}
                    id="testing"
                    className="min-h-screen py-16 flex flex-col items-center"
                >
                    <div className="w-full max-w-5xl mx-auto pb-24">
                        {/* Section Header */}
                        <div className="flex items-center justify-center gap-4 mb-14 mt-4">
                            <TestTube2 className="w-10 h-10" style={{ color: '#2596be' }} />
                            <h2 className="text-3xl font-bold" style={{ color: '#2596be' }}>
                                Model Testing
                            </h2>
                        </div>

                        {/* Sub-section Content */}
                        <AnimatePresence mode="wait">
                            <motion.div
                                key="testing-upload"
                                initial={{ opacity: 0, x: -30 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 30 }}
                                transition={{ duration: 0.3 }}
                            >
                                <ModelTesting
                                    defaultModelId={currentModel?.id ?? "resnet18"}
                                    models={models.map((m) => ({ id: m.id, name: m.name }))}
                                />
                            </motion.div>
                        </AnimatePresence>
                    </div>
                </section>

                {/* Footer */}
                <footer className="py-12 text-center">
                    <p className="text-gray-400 text-sm">
                        Plant Disease Classification Report • Built with React, Three.js, and Recharts
                    </p>
                </footer>
            </div>
        </div>
    );
}

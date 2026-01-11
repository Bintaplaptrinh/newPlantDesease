import { useMemo } from "react";
import { motion } from "framer-motion";
import {
    ResponsiveContainer,
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ReferenceLine,
} from "recharts";
import { TrendingUp } from "lucide-react";

export type RocPoint = {
    fpr: number;
    mobilenet?: number;
    resnet18?: number;
    efficientnet?: number;
};

function clamp01(v: number) {
    return Math.max(0, Math.min(1, v));
}

function auc(points: { x: number; y: number }[]) {
    if (points.length < 2) return 0;
    const sorted = [...points].sort((a, b) => a.x - b.x);
    let area = 0;
    for (let i = 1; i < sorted.length; i++) {
        const x0 = clamp01(sorted[i - 1].x);
        const x1 = clamp01(sorted[i].x);
        const y0 = clamp01(sorted[i - 1].y);
        const y1 = clamp01(sorted[i].y);
        const width = x1 - x0;
        area += width * (y0 + y1) / 2;
    }
    return clamp01(area);
}

export function RocCurveComparison({
    data = [],
}: {
    data?: RocPoint[];
}) {
    const aucs = useMemo(() => {
        const mobilenet = auc(data.map((d) => ({ x: d.fpr, y: d.mobilenet ?? 0 })));
        const resnet18 = auc(data.map((d) => ({ x: d.fpr, y: d.resnet18 ?? 0 })));
        const efficientnet = auc(data.map((d) => ({ x: d.fpr, y: d.efficientnet ?? 0 })));
        return { mobilenet, resnet18, efficientnet };
    }, [data]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="glass-card p-10"
        >
            <div className="flex items-center gap-3 mb-6">
                <TrendingUp className="w-8 h-8" style={{ color: "#2596be" }} />
                <div>
                    <h3 className="text-2xl font-bold text-gray-900">ROC-CURVE (3 Models)</h3>
                    <p className="text-sm text-gray-500 mt-1">
                        So sánh ROC giữa MobileNet, ResNet18 và EfficientNet-B0.
                    </p>
                </div>
            </div>

            {data.length === 0 && (
                <div className="mb-8 text-sm text-gray-500">
                    ROC data is not available yet. Export `data/web/roc_micro.json` from the evaluation machine.
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                {[
                    { name: "MobileNet", value: aucs.mobilenet, color: "#2596be" },
                    { name: "ResNet18", value: aucs.resnet18, color: "#1a7a9c" },
                    { name: "EfficientNet-B0", value: aucs.efficientnet, color: "#3db8e5" },
                ].map((m) => (
                    <div key={m.name} className="text-center p-6 rounded-xl bg-gradient-to-br from-blue-50 to-cyan-50">
                        <p className="text-xs text-gray-500">AUC</p>
                        <p className="text-3xl font-bold" style={{ color: m.color }}>
                            {m.value.toFixed(3)}
                        </p>
                        <p className="text-sm font-semibold text-gray-800 mt-1">{m.name}</p>
                    </div>
                ))}
            </div>

            <div className="h-[520px]">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 10, right: 24, left: 10, bottom: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis
                            dataKey="fpr"
                            type="number"
                            domain={[0, 1]}
                            tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`}
                            stroke="#6b7280"
                            tickLine={false}
                            axisLine={{ stroke: "#d1d5db" }}
                        />
                        <YAxis
                            type="number"
                            domain={[0, 1]}
                            tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`}
                            stroke="#6b7280"
                            tickLine={false}
                            axisLine={{ stroke: "#d1d5db" }}
                        />
                        <Tooltip
                            formatter={(value) => [`${(Number(value ?? 0) * 100).toFixed(1)}%`, "TPR"]}
                            labelFormatter={(label) => `FPR ${(Number(label ?? 0) * 100).toFixed(1)}%`}
                            contentStyle={{
                                backgroundColor: "rgba(255,255,255,0.98)",
                                border: "1px solid #e5e7eb",
                                borderRadius: "12px",
                                boxShadow: "0 8px 24px rgba(0,0,0,0.12)",
                                padding: "12px 16px",
                                color: "#111827",
                            }}
                            labelStyle={{ fontWeight: 700, color: "#2596be", marginBottom: 6 }}
                        />
                        <Legend />
                        <ReferenceLine x={0} y={0} stroke="#9ca3af" strokeDasharray="4 4" />
                        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#9ca3af" strokeDasharray="4 4" />

                        <Line
                            type="monotone"
                            dataKey="mobilenet"
                            name="MobileNet"
                            stroke="#2596be"
                            strokeWidth={3}
                            dot={false}
                            activeDot={{ r: 5 }}
                        />
                        <Line
                            type="monotone"
                            dataKey="resnet18"
                            name="ResNet18"
                            stroke="#1a7a9c"
                            strokeWidth={3}
                            dot={false}
                            activeDot={{ r: 5 }}
                        />
                        <Line
                            type="monotone"
                            dataKey="efficientnet"
                            name="EfficientNet-B0"
                            stroke="#3db8e5"
                            strokeWidth={3}
                            dot={false}
                            activeDot={{ r: 5 }}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>

        </motion.div>
    );
}

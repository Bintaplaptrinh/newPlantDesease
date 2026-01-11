import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
    ResponsiveContainer,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Cell,
} from "recharts";
import { BarChart3 } from "lucide-react";

type AccuracyRow = {
    label: string;
    accuracy: number; // 0..100
    correct: number;
    incorrect: number;
    total: number;
};

const BAR_COLOR = "#2596be";
const BAR_MUTED = "#d1d5db";

function sum(row: number[]) {
    return row.reduce((acc, v) => acc + v, 0);
}

export function PerClassAccuracyChart({
    confusionMatrix,
    labels,
    title = "Per-class Accuracy",
}: {
    confusionMatrix: number[][];
    labels?: string[];
    title?: string;
}) {
    const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

    const data = useMemo<AccuracyRow[]>(() => {
        const n = confusionMatrix.length;
        const effectiveLabels = labels?.length ? labels : Array.from({ length: n }, (_, i) => `Class ${i + 1}`);

        return confusionMatrix.map((row, i) => {
            const total = sum(row);
            const correct = row[i] ?? 0;
            const incorrect = Math.max(0, total - correct);
            const accuracy = total > 0 ? (correct / total) * 100 : 0;
            return {
                label: effectiveLabels[i] ?? `Class ${i + 1}`,
                accuracy,
                correct,
                incorrect,
                total,
            };
        });
    }, [confusionMatrix, labels]);

    const avgAccuracy = useMemo(() => {
        if (!data.length) return 0;
        const mean = data.reduce((acc, r) => acc + r.accuracy, 0) / data.length;
        return mean;
    }, [data]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="glass-card p-10"
        >
            <div className="flex items-center gap-3 mb-6">
                <BarChart3 className="w-8 h-8" style={{ color: "#2596be" }} />
                <div>
                    <h3 className="text-2xl font-bold text-gray-900">{title}</h3>
                    <p className="text-sm text-gray-500 mt-1">
                        Accuracy theo từng lớp: đúng / tổng ảnh của lớp đó.
                    </p>
                </div>

                <div className="ml-auto text-right">
                    <p className="text-xs text-gray-500">Avg</p>
                    <p className="text-xl font-bold" style={{ color: "#2596be" }}>
                        {avgAccuracy.toFixed(1)}%
                    </p>
                </div>
            </div>

            <div className="h-[520px]">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={data}
                        margin={{ top: 10, right: 20, left: 0, bottom: 80 }}
                        onMouseMove={(state) => {
                            const idx = typeof state.activeTooltipIndex === "number" ? state.activeTooltipIndex : null;
                            setHoveredIndex(idx);
                        }}
                        onMouseLeave={() => setHoveredIndex(null)}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis
                            dataKey="label"
                            interval={0}
                            angle={-35}
                            textAnchor="end"
                            height={80}
                            stroke="#6b7280"
                            tickLine={false}
                            axisLine={{ stroke: "#d1d5db" }}
                            fontSize={11}
                        />
                        <YAxis
                            domain={[0, 100]}
                            tickFormatter={(v) => `${Number(v)}%`}
                            stroke="#6b7280"
                            tickLine={false}
                            axisLine={{ stroke: "#d1d5db" }}
                        />
                        <Tooltip
                            cursor={{ fill: "rgba(37, 150, 190, 0.06)" }}
                            content={({ active, payload }) => {
                                if (!active || !payload?.length) return null;
                                const p = payload[0]?.payload as AccuracyRow | undefined;
                                if (!p) return null;

                                return (
                                    <div
                                        className="glass-nav rounded-2xl"
                                        style={{
                                            padding: 14,
                                            minWidth: 260,
                                            border: "1px solid rgba(229, 231, 235, 0.9)",
                                            boxShadow: "0 10px 28px rgba(0,0,0,0.14)",
                                        }}
                                    >
                                        <div className="text-sm font-bold" style={{ color: "#2596be" }}>
                                            {p.label}
                                        </div>
                                        <div className="mt-2 grid grid-cols-2 gap-3 text-sm">
                                            <div className="rounded-xl bg-white/70 border border-gray-200 px-3 py-2">
                                                <div className="text-xs text-gray-500">Accuracy</div>
                                                <div className="font-bold text-gray-900">{p.accuracy.toFixed(1)}%</div>
                                            </div>
                                            <div className="rounded-xl bg-white/70 border border-gray-200 px-3 py-2">
                                                <div className="text-xs text-gray-500">Total</div>
                                                <div className="font-bold text-gray-900">{p.total.toLocaleString()}</div>
                                            </div>
                                            <div className="rounded-xl bg-white/70 border border-gray-200 px-3 py-2">
                                                <div className="text-xs text-gray-500">Đúng</div>
                                                <div className="font-bold" style={{ color: "#1a7a9c" }}>{p.correct.toLocaleString()}</div>
                                            </div>
                                            <div className="rounded-xl bg-white/70 border border-gray-200 px-3 py-2">
                                                <div className="text-xs text-gray-500">Sai</div>
                                                <div className="font-bold text-gray-800">{p.incorrect.toLocaleString()}</div>
                                            </div>
                                        </div>
                                    </div>
                                );
                            }}
                        />
                        <Bar dataKey="accuracy" radius={[10, 10, 6, 6]}>
                            {data.map((_, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={hoveredIndex === null || hoveredIndex === index ? BAR_COLOR : BAR_MUTED}
                                    opacity={hoveredIndex === null || hoveredIndex === index ? 1 : 0.92}
                                />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </motion.div>
    );
}

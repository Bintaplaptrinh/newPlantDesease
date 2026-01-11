import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
    Bar,
    BarChart,
    CartesianGrid,
    Cell,
    LabelList,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";
import { BarChart3, Sun } from "lucide-react";

export type OverBrightRow = {
    label: string;
    overBright: number;
    total: number;
    ratio: number; // 0..1
};

const BAR_COLOR = "#2596be";
const BAR_MUTED = "#d1d5db";

function formatLabel(label: string) {
    return label.replace(/___/g, " - ").replace(/_/g, " ");
}

export function OverBrightImagesChart({
    rows,
    topN = 12,
}: {
    rows: OverBrightRow[];
    topN?: number;
}) {
    const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

    const data = useMemo(() => {
        const sorted = [...(rows ?? [])]
            .filter((r) => r && typeof r.ratio === "number")
            .sort((a, b) => (b.ratio ?? 0) - (a.ratio ?? 0));

        return sorted.slice(0, topN).map((r) => ({
            name: formatLabel(r.label),
            overBright: r.overBright,
            total: r.total,
            ratioPct: (r.ratio ?? 0) * 100,
            display: `${r.overBright.toLocaleString()} / ${r.total.toLocaleString()}`,
        }));
    }, [rows, topN]);

    const maxPct = useMemo(() => {
        const m = data.reduce((acc, it) => Math.max(acc, it.ratioPct ?? 0), 0);
        return Math.max(5, Math.ceil(m / 5) * 5);
    }, [data]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="glass-card p-10"
        >
            <div className="flex items-center gap-3 mb-6">
                <Sun className="w-7 h-7" style={{ color: "#2596be" }} />
                <h3 className="text-2xl font-bold text-gray-900">Ảnh quá sáng</h3>
            </div>

            <p className="text-sm text-gray-500 mb-6">
                Biểu đồ hiển thị tỷ lệ ảnh quá sáng (%)
            </p>

            <div className="h-[520px]">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={data}
                        layout="vertical"
                        margin={{ top: 10, right: 40, left: 190, bottom: 10 }}
                        onMouseMove={(state) => {
                            const idx = typeof state.activeTooltipIndex === "number" ? state.activeTooltipIndex : null;
                            setHoveredIndex(idx);
                        }}
                        onMouseLeave={() => setHoveredIndex(null)}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis
                            type="number"
                            domain={[0, maxPct]}
                            tickFormatter={(v) => `${Number(v).toFixed(0)}%`}
                            stroke="#6b7280"
                            fontSize={12}
                            tickLine={false}
                            axisLine={{ stroke: "#d1d5db" }}
                        />
                        <YAxis
                            type="category"
                            dataKey="name"
                            stroke="#6b7280"
                            fontSize={11}
                            tickLine={false}
                            axisLine={{ stroke: "#d1d5db" }}
                            width={180}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: "rgba(255, 255, 255, 0.98)",
                                border: "1px solid #e5e7eb",
                                borderRadius: "12px",
                                color: "#1a1a1a",
                                boxShadow: "0 8px 24px rgba(0,0,0,0.12)",
                                padding: "12px 16px",
                            }}
                            formatter={(_, __, item: any) => {
                                const p = item?.payload;
                                if (!p) return ["-", ""];
                                return [`${p.display} (${Number(p.ratioPct ?? 0).toFixed(1)}%)`, "Over-bright"];
                            }}
                            labelStyle={{ fontWeight: "bold", color: "#2596be", marginBottom: "4px" }}
                        />

                        <Bar dataKey="ratioPct" radius={[0, 6, 6, 0]}>
                            <LabelList dataKey="display" position="right" fill="#374151" fontSize={11} />
                            {data.map((_, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={hoveredIndex === null || hoveredIndex === index ? BAR_COLOR : BAR_MUTED}
                                    opacity={hoveredIndex === null || hoveredIndex === index ? 1 : 0.9}
                                />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>

            <div className="flex items-center gap-2 text-xs text-gray-400 mt-4">
                <BarChart3 className="w-4 h-4" />
                <span>Ratio = overBright / total per class</span>
            </div>
        </motion.div>
    );
}

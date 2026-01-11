import { useState } from "react";
import { motion } from "framer-motion";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell,
} from "recharts";
import { BarChart3, Image, Tag, TrendingUp, TrendingDown } from "lucide-react";

// 38 Classes distribution data
export type ClassDistributionRow = {
    shortName: string;
    count: number;
};

const BAR_COLOR = "#2596be";
const BAR_MUTED = "#d1d5db";
const NUMBER_COLOR = "#1a7a9c";

export function ClassDistributionChart({
    data,
}: {
    data: ClassDistributionRow[];
}) {
    const totalImages = data.reduce((sum, item) => sum + item.count, 0);
    const avgPerClass = data.length ? Math.round(totalImages / data.length) : 0;
    const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="glass-card p-10"
        >
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
                <BarChart3 className="w-8 h-8" style={{ color: '#2596be' }} />
                <h3 className="text-2xl font-bold text-gray-900">
                    Khái quát về dữ liệu
                </h3>
            </div>

            {/* Horizontal Layout: Stats on left, Chart on right */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Stats Panel - Left Side */}
                <div className="lg:col-span-1 space-y-5 pl-2">
                    <p className="text-sm text-gray-500 mb-4">
                        {data.length
                            ? <>Dataset chứa {data.length} lớp với tổng cộng {totalImages.toLocaleString()} ảnh</>
                            : <>Dataset distribution is not available yet. Run the Python pipeline to export real JSON.</>
                        }
                    </p>

                    {/* Stats Cards */}
                    <div className="space-y-3">
                        <div className="text-center p-6 rounded-xl bg-gradient-to-br from-blue-50 to-cyan-50">
                            <div className="flex items-center gap-2 mb-1">
                                <Tag className="w-4 h-4" style={{ color: '#2596be' }} />
                                <span className="text-xs text-gray-500">Số lớp</span>
                            </div>
                            <p className="text-2xl font-bold" style={{ color: NUMBER_COLOR }}>
                                {data.length}
                            </p>
                        </div>

                        <div className="text-center p-6 rounded-xl bg-gradient-to-br from-blue-50 to-cyan-50">
                            <div className="flex items-center gap-2 mb-1">
                                <Image className="w-4 h-4" style={{ color: '#2596be' }} />
                                <span className="text-xs text-gray-500">Số lượng ảnh</span>
                            </div>
                            <p className="text-2xl font-bold" style={{ color: NUMBER_COLOR }}>
                                {totalImages.toLocaleString()}
                            </p>
                        </div>

                        <div className="text-center p-6 rounded-xl bg-gradient-to-br from-blue-50 to-cyan-50">
                            <div className="flex items-center gap-2 mb-1">
                                <TrendingUp className="w-4 h-4" style={{ color: '#2596be' }} />
                                <span className="text-xs text-gray-500">Số lượng ảnh lớn nhất</span>
                            </div>
                            <p className="text-2xl font-bold" style={{ color: NUMBER_COLOR }}>
                                {data.length ? Math.max(...data.map(d => d.count)).toLocaleString() : "-"}
                            </p>
                            <p className="text-xs text-gray-400 mt-1">Soybean Healthy</p>
                        </div>

                        <div className="text-center p-6 rounded-xl bg-gradient-to-br from-blue-50 to-cyan-50">
                            <div className="flex items-center gap-2 mb-1">
                                <TrendingDown className="w-4 h-4" style={{ color: '#2596be' }} />
                                <span className="text-xs text-gray-500">Số lượng ảnh ít nhất</span>
                            </div>
                            <p className="text-2xl font-bold" style={{ color: NUMBER_COLOR }}>
                                {data.length ? Math.min(...data.map(d => d.count)).toLocaleString() : "-"}
                            </p>
                            <p className="text-xs text-gray-400 mt-1">Corn Cercospora</p>
                        </div>

                        <div className="text-center p-6 rounded-xl bg-gradient-to-br from-blue-50 to-cyan-50">
                            <div className="flex items-center gap-2 mb-1">
                                <BarChart3 className="w-4 h-4" style={{ color: '#2596be' }} />
                                <span className="text-xs text-gray-500">Trung bình ảnh tại mỗi lớp</span>
                            </div>
                            <p className="text-2xl font-bold" style={{ color: NUMBER_COLOR }}>
                                {avgPerClass.toLocaleString()}
                            </p>
                        </div>
                    </div>
                </div>

                {/* Chart - Right Side */}
                <div className="lg:col-span-3 h-[600px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                            data={data}
                            layout="vertical"
                            margin={{ top: 10, right: 30, left: 130, bottom: 10 }}
                            onMouseMove={(state) => {
                                const idx = typeof state.activeTooltipIndex === "number" ? state.activeTooltipIndex : null;
                                setHoveredIndex(idx);
                            }}
                            onMouseLeave={() => setHoveredIndex(null)}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis
                                type="number"
                                stroke="#6b7280"
                                fontSize={12}
                                tickLine={false}
                                axisLine={{ stroke: '#d1d5db' }}
                            />
                            <YAxis
                                type="category"
                                dataKey="shortName"
                                stroke="#6b7280"
                                fontSize={11}
                                tickLine={false}
                                axisLine={{ stroke: '#d1d5db' }}
                                width={125}
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
                                formatter={(value) => [`${Number(value ?? 0).toLocaleString()} images`, ""]}
                                labelStyle={{ fontWeight: "bold", color: "#2596be", marginBottom: "4px" }}
                            />
                            <Bar dataKey="count" radius={[0, 6, 6, 0]}>
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
            </div>
        </motion.div>
    );
}

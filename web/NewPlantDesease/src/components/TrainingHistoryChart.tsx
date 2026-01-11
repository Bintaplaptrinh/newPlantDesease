import { motion } from "framer-motion";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    Area,
    AreaChart,
} from "recharts";
import { TrendingDown, TrendingUp } from "lucide-react";

export type TrainingHistoryRow = {
    epoch: number;
    trainLoss: number;
    valLoss: number;
    trainAcc: number;
    valAcc: number;
};

export function TrainingHistoryChart({
    data,
}: {
    data: TrainingHistoryRow[];
}) {
    const hasData = data.length > 0;
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="glass-card p-10"
        >
            <div className="flex items-center gap-4 mb-8">
                <TrendingUp className="w-6 h-6" style={{ color: '#2596be' }} />
                <h3 className="text-xl font-bold" style={{ color: '#2596be' }}>
                    Training History
                </h3>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                {/* Loss Chart */}
                <div>
                    <h4 className="text-sm font-medium text-gray-600 mb-6 flex items-center gap-3">
                        <TrendingDown className="w-4 h-4" style={{ color: '#2596be' }} />
                        Loss Over Epochs
                    </h4>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={data}>
                                <defs>
                                    <linearGradient id="trainLossGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#2596be" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#2596be" stopOpacity={0} />
                                    </linearGradient>
                                    <linearGradient id="valLossGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#3db8e5" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#3db8e5" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                <XAxis
                                    dataKey="epoch"
                                    stroke="#6b7280"
                                    fontSize={11}
                                    tickLine={false}
                                    axisLine={{ stroke: '#d1d5db' }}
                                />
                                <YAxis
                                    stroke="#6b7280"
                                    fontSize={11}
                                    tickLine={false}
                                    axisLine={{ stroke: '#d1d5db' }}
                                    tickFormatter={(value: number) => value.toFixed(1)}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: "rgba(255, 255, 255, 0.95)",
                                        border: "1px solid #e5e7eb",
                                        borderRadius: "8px",
                                        color: "#1a1a1a",
                                    }}
                                    labelFormatter={(label: number) => `Epoch ${label}`}
                                />
                                <Legend />
                                <Area
                                    type="monotone"
                                    dataKey="trainLoss"
                                    stroke="#2596be"
                                    fill="url(#trainLossGradient)"
                                    strokeWidth={2}
                                    name="Train Loss"
                                />
                                <Area
                                    type="monotone"
                                    dataKey="valLoss"
                                    stroke="#3db8e5"
                                    fill="url(#valLossGradient)"
                                    strokeWidth={2}
                                    name="Val Loss"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Accuracy Chart */}
                <div>
                    <h4 className="text-sm font-medium text-gray-600 mb-6 flex items-center gap-3">
                        <TrendingUp className="w-4 h-4" style={{ color: '#2596be' }} />
                        Accuracy Over Epochs
                    </h4>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                <XAxis
                                    dataKey="epoch"
                                    stroke="#6b7280"
                                    fontSize={11}
                                    tickLine={false}
                                    axisLine={{ stroke: '#d1d5db' }}
                                />
                                <YAxis
                                    stroke="#6b7280"
                                    fontSize={11}
                                    tickLine={false}
                                    axisLine={{ stroke: '#d1d5db' }}
                                    domain={[0, 1]}
                                    tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: "rgba(255, 255, 255, 0.95)",
                                        border: "1px solid #e5e7eb",
                                        borderRadius: "8px",
                                        color: "#1a1a1a",
                                    }}
                                    labelFormatter={(label: number) => `Epoch ${label}`}
                                    formatter={(value) => [`${(Number(value ?? 0) * 100).toFixed(1)}%`, ""]}
                                />
                                <Legend />
                                <Line
                                    type="monotone"
                                    dataKey="trainAcc"
                                    stroke="#2596be"
                                    strokeWidth={2}
                                    dot={{ fill: "#2596be", strokeWidth: 0, r: 3 }}
                                    activeDot={{ r: 5, stroke: "#2596be", strokeWidth: 2 }}
                                    name="Train Accuracy"
                                />
                                <Line
                                    type="monotone"
                                    dataKey="valAcc"
                                    stroke="#3db8e5"
                                    strokeWidth={2}
                                    dot={{ fill: "#3db8e5", strokeWidth: 0, r: 3 }}
                                    activeDot={{ r: 5, stroke: "#3db8e5", strokeWidth: 2 }}
                                    name="Val Accuracy"
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Summary stats */}
            <div className="mt-8 pt-6 border-t border-gray-200 grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
                <div>
                    <p className="text-xs text-gray-500 mb-1">Final Train Loss</p>
                    <p className="text-lg font-bold" style={{ color: '#2596be' }}>
                        {hasData ? data[data.length - 1].trainLoss.toFixed(3) : "-"}
                    </p>
                </div>
                <div>
                    <p className="text-xs text-gray-500 mb-1">Final Val Loss</p>
                    <p className="text-lg font-bold" style={{ color: '#3db8e5' }}>
                        {hasData ? data[data.length - 1].valLoss.toFixed(3) : "-"}
                    </p>
                </div>
                <div>
                    <p className="text-xs text-gray-500 mb-1">Final Train Acc</p>
                    <p className="text-lg font-bold" style={{ color: '#2596be' }}>
                        {hasData ? `${(data[data.length - 1].trainAcc * 100).toFixed(1)}%` : "-"}
                    </p>
                </div>
                <div>
                    <p className="text-xs text-gray-500 mb-1">Final Val Acc</p>
                    <p className="text-lg font-bold" style={{ color: '#3db8e5' }}>
                        {hasData ? `${(data[data.length - 1].valAcc * 100).toFixed(1)}%` : "-"}
                    </p>
                </div>
            </div>

            {!hasData && (
                <div className="mt-6 text-sm text-gray-500">
                    Training history JSON is empty. After you train elsewhere, export `training_history.json` into `data/web/models/&lt;modelId&gt;/training_history.json`.
                </div>
            )}
        </motion.div>
    );
}

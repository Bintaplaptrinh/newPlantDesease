import { motion } from "framer-motion";
import {
    RadialBarChart,
    RadialBar,
    PolarAngleAxis,
    ResponsiveContainer,
} from "recharts";
import { Target } from "lucide-react";

interface AccuracyChartProps {
    accuracy: number;
    label?: string;
    color?: string;
}

export function AccuracyChart({
    accuracy,
    label = "Model Accuracy",
    color = "#2596be",
}: AccuracyChartProps) {
    const percentage = Math.round(accuracy * 100 * 10) / 10;

    const data = [
        {
            name: label,
            value: percentage,
            fill: color,
        },
    ];

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            className="glass-card p-8 w-full max-w-[320px]"
        >
            <h3 className="text-lg font-bold text-gray-800 mb-6 text-center flex items-center justify-center gap-3">
                <Target className="w-5 h-5" style={{ color: '#2596be' }} />
                {label}
            </h3>

            <div className="relative w-full aspect-square">
                <ResponsiveContainer width="100%" height="100%">
                    <RadialBarChart
                        cx="50%"
                        cy="50%"
                        innerRadius="70%"
                        outerRadius="100%"
                        barSize={20}
                        data={data}
                        startAngle={90}
                        endAngle={-270}
                    >
                        <PolarAngleAxis
                            type="number"
                            domain={[0, 100]}
                            angleAxisId={0}
                            tick={false}
                        />

                        <RadialBar
                            background={{ fill: "rgba(37, 150, 190, 0.1)" }}
                            dataKey="value"
                            cornerRadius={10}
                        />
                    </RadialBarChart>
                </ResponsiveContainer>

                {/* Center text */}
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <motion.span
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3, duration: 0.5 }}
                        className="text-4xl font-bold"
                        style={{ color: '#2596be' }}
                    >
                        {percentage}%
                    </motion.span>
                    <span className="text-sm text-gray-500 mt-1">Accuracy</span>
                </div>
            </div>

            {/* Performance indicator */}
            <div className="mt-6 flex items-center justify-center gap-3">
                <div
                    className={`w-2 h-2 rounded-full ${percentage >= 90
                        ? "bg-green-400"
                        : percentage >= 75
                            ? "bg-yellow-400"
                            : "bg-red-400"
                        }`}
                />
                <span className="text-sm text-gray-600">
                    {percentage >= 90
                        ? "Excellent"
                        : percentage >= 75
                            ? "Good"
                            : "Needs Improvement"}
                </span>
            </div>
        </motion.div>
    );
}

// Multi-metric version
interface MetricData {
    name: string;
    value: number;
    color: string;
}

interface MultiMetricChartProps {
    metrics: MetricData[];
}

export function MultiMetricChart({ metrics }: MultiMetricChartProps) {
    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            className="glass-card p-8"
        >
            <h3 className="text-lg font-bold text-gray-800 mb-8 text-center">
                Hiệu suất mô hình
            </h3>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                {metrics.map((metric, index) => (
                    <motion.div
                        key={metric.name}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1, duration: 0.4 }}
                        className="relative flex flex-col items-center"
                    >
                        <div className="relative w-24 h-24">
                            <ResponsiveContainer width="100%" height="100%">
                                <RadialBarChart
                                    cx="50%"
                                    cy="50%"
                                    innerRadius="75%"
                                    outerRadius="100%"
                                    barSize={8}
                                    data={[{ value: metric.value * 100, fill: metric.color }]}
                                    startAngle={90}
                                    endAngle={-270}
                                >
                                    <PolarAngleAxis
                                        type="number"
                                        domain={[0, 100]}
                                        tick={false}
                                    />
                                    <RadialBar
                                        background={{ fill: "rgba(37, 150, 190, 0.1)" }}
                                        dataKey="value"
                                        cornerRadius={5}
                                    />
                                </RadialBarChart>
                            </ResponsiveContainer>

                            <div className="absolute inset-0 flex items-center justify-center">
                                <span
                                    className="text-xl font-bold"
                                    style={{ color: metric.color }}
                                >
                                    {Math.round(metric.value * 100)}%
                                </span>
                            </div>
                        </div>

                        <span className="text-sm text-gray-600 mt-2">{metric.name}</span>
                    </motion.div>
                ))}
            </div>
        </motion.div>
    );
}

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Eye, SlidersHorizontal } from "lucide-react";

type HeatmapPreset = {
    id: string;
    name: string;
    heatCss: string;
};

const BASE_IMAGE_SVG = encodeURIComponent(`
<svg xmlns='http://www.w3.org/2000/svg' width='900' height='600' viewBox='0 0 900 600'>
  <defs>
    <linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0' stop-color='#eff6ff'/>
      <stop offset='1' stop-color='#ecfeff'/>
    </linearGradient>
    <linearGradient id='leaf' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0' stop-color='#34d399'/>
      <stop offset='1' stop-color='#10b981'/>
    </linearGradient>
  </defs>
  <rect width='900' height='600' fill='url(#bg)'/>
  <g opacity='0.9'>
    <path d='M470 110 C350 170 260 280 250 380 C240 480 320 530 420 500 C540 455 640 340 650 250 C660 160 590 80 470 110 Z'
          fill='url(#leaf)'/>
    <path d='M420 490 C470 390 520 300 600 210' stroke='#065f46' stroke-width='10' stroke-linecap='round' opacity='0.35'/>
    <path d='M380 420 C450 360 520 300 600 240' stroke='#065f46' stroke-width='6' stroke-linecap='round' opacity='0.25'/>
    <path d='M350 360 C430 320 520 270 610 230' stroke='#065f46' stroke-width='5' stroke-linecap='round' opacity='0.18'/>
  </g>
  <text x='32' y='560' font-family='Arial, sans-serif' font-size='18' fill='#6b7280'>Placeholder leaf image (for explainability demo)</text>
</svg>
`);

const PRESETS: HeatmapPreset[] = [
    {
        id: "mobilenet",
        name: "MobileNet",
        heatCss:
            "radial-gradient(circle at 35% 42%, rgba(255, 0, 0, 0.62), transparent 56%)," +
            "radial-gradient(circle at 60% 58%, rgba(255, 140, 0, 0.48), transparent 56%)," +
            "radial-gradient(circle at 52% 28%, rgba(255, 0, 0, 0.35), transparent 52%)",
    },
    {
        id: "resnet18",
        name: "ResNet18",
        heatCss:
            "radial-gradient(circle at 42% 50%, rgba(255, 0, 0, 0.66), transparent 58%)," +
            "radial-gradient(circle at 72% 52%, rgba(255, 80, 0, 0.46), transparent 56%)," +
            "radial-gradient(circle at 55% 22%, rgba(255, 0, 0, 0.28), transparent 50%)",
    },
    {
        id: "efficientnet",
        name: "EfficientNet-B0",
        heatCss:
            "radial-gradient(circle at 46% 46%, rgba(255, 0, 0, 0.70), transparent 60%)," +
            "radial-gradient(circle at 68% 60%, rgba(255, 120, 0, 0.48), transparent 58%)," +
            "radial-gradient(circle at 50% 30%, rgba(255, 0, 0, 0.32), transparent 52%)",
    },
];

export function ModelExplainabilityDetail({
    defaultPresetId = "efficientnet",
}: {
    defaultPresetId?: string;
}) {
    const [presetId, setPresetId] = useState(defaultPresetId);
    const [opacity, setOpacity] = useState(62);

    const preset = useMemo(() => PRESETS.find((p) => p.id === presetId) ?? PRESETS[0], [presetId]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="glass-card p-10"
        >
            <div className="flex items-center gap-3 mb-6">
                <Eye className="w-8 h-8" style={{ color: "#2596be" }} />
                <div>
                    <h3 className="text-2xl font-bold text-gray-900">Detail (Explainability)</h3>
                    <p className="text-sm text-gray-500 mt-1">
                        Trích xuất “vùng quan trọng” (Grad-CAM/Saliency) mô hình dựa vào để ra quyết định.
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Controls */}
                <div className="lg:col-span-1">
                    <div className="rounded-2xl bg-gradient-to-br from-blue-50 to-cyan-50 p-6">
                        <div className="flex items-center gap-2 mb-4">
                            <SlidersHorizontal className="w-5 h-5" style={{ color: "#2596be" }} />
                            <h4 className="font-bold text-gray-900">Controls</h4>
                        </div>

                        <label className="text-sm text-gray-600">Model</label>
                        <div className="mt-2 grid grid-cols-1 gap-2">
                            {PRESETS.map((p) => (
                                <button
                                    key={p.id}
                                    onClick={() => setPresetId(p.id)}
                                    className={`px-4 py-2 rounded-xl text-sm font-medium transition-all border ${
                                        presetId === p.id
                                            ? "text-white border-transparent"
                                            : "text-gray-700 border-gray-200 bg-white/70 hover:bg-white"
                                    }`}
                                    style={presetId === p.id ? { background: "linear-gradient(135deg, #2596be, #3db8e5)" } : {}}
                                >
                                    {p.name}
                                </button>
                            ))}
                        </div>

                        <div className="mt-6">
                            <label className="text-sm text-gray-600">Heatmap opacity</label>
                            <div className="mt-2 flex items-center gap-3">
                                <input
                                    type="range"
                                    min={0}
                                    max={100}
                                    value={opacity}
                                    onChange={(e) => setOpacity(Number(e.target.value))}
                                    className="w-full"
                                />
                                <div className="w-14 text-right text-sm font-bold" style={{ color: "#2596be" }}>
                                    {opacity}%
                                </div>
                            </div>
                        </div>

                        <div className="mt-6 text-sm text-gray-700 leading-relaxed">
                            <p className="font-semibold text-gray-900">Ý nghĩa</p>
                            <p className="mt-2">Vùng “đỏ/đậm” = mô hình chú ý nhiều hơn khi dự đoán.</p>
                            <p className="mt-2 text-gray-600">
                                Tích hợp thật: backend trả về heatmap (base64 PNG) theo ảnh + mô hình (Grad-CAM).
                            </p>
                        </div>
                    </div>
                </div>

                {/* Visualization */}
                <div className="lg:col-span-2">
                    <div className="rounded-3xl overflow-hidden border border-gray-200 bg-white shadow-sm">
                        <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
                            <div>
                                <p className="text-sm text-gray-500">Selected</p>
                                <p className="text-lg font-bold text-gray-900">{preset.name}</p>
                            </div>
                            <div className="text-right">
                                <p className="text-xs text-gray-500">Heatmap</p>
                                <p className="text-sm font-semibold" style={{ color: "#2596be" }}>Grad-CAM (demo)</p>
                            </div>
                        </div>

                        <div className="relative w-full aspect-[3/2]">
                            <img
                                src={`data:image/svg+xml;utf8,${BASE_IMAGE_SVG}`}
                                alt="Leaf sample"
                                className="absolute inset-0 w-full h-full object-cover"
                            />
                            <div
                                className="absolute inset-0"
                                style={{
                                    backgroundImage: preset.heatCss,
                                    opacity: opacity / 100,
                                    mixBlendMode: "multiply",
                                }}
                            />
                        </div>

                        <div className="px-6 py-5">
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                {[
                                    { k: "What", v: "Most influential regions" },
                                    { k: "Why", v: "Supports the predicted class" },
                                    { k: "How", v: "Grad-CAM on last conv layers" },
                                ].map((x) => (
                                    <div key={x.k} className="rounded-2xl bg-gradient-to-br from-blue-50 to-cyan-50 p-4">
                                        <div className="text-xs text-gray-500">{x.k}</div>
                                        <div className="mt-1 font-semibold text-gray-900">{x.v}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

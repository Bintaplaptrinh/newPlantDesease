import { useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera, Html, Text } from "@react-three/drei";
import * as THREE from "three";
import { motion } from "framer-motion";

const defaultClassLabels: string[] = [];

interface ConfusionMatrix3DProps {
    data: number[][];
    labels?: string[];
}

interface HoveredBar {
    row: number;
    col: number;
    value: number;
    position: [number, number, number];
}

const BAR_SIZE: [number, number] = [0.8, 0.8];
const MIN_BAR_HEIGHT = 0.08;
const MAX_BAR_HEIGHT = 5;

const BASE_BAR_COLOR = "#2596be";
const HOVER_BAR_COLOR = "#1a7a9c";

function clamp01(value: number) {
    return Math.max(0, Math.min(1, value));
}

function opacityForIntensity(intensity01: number, isHovered: boolean) {
    if (isHovered) return 1;
    const minOpacity = 0.18;
    const maxOpacity = 1;
    return minOpacity + (maxOpacity - minOpacity) * clamp01(intensity01);
}

type BarItem = {
    id: number;
    row: number;
    col: number;
    value: number;
    intensity: number;
    x: number;
    z: number;
    targetHeight: number;
};

function BarCell({
    item,
    isHovered,
    onHover,
    onUnhover,
}: {
    item: BarItem;
    isHovered: boolean;
    onHover: (bar: HoveredBar) => void;
    onUnhover: () => void;
}) {
    const meshRef = useRef<THREE.Mesh>(null);
    const currentHeight = useRef(MIN_BAR_HEIGHT);

    useFrame(() => {
        const mesh = meshRef.current;
        if (!mesh) return;

        const next = THREE.MathUtils.lerp(currentHeight.current, item.targetHeight, 0.12);
        currentHeight.current = next;

        mesh.position.set(item.x, next / 2, item.z);
        mesh.scale.set(BAR_SIZE[0], next, BAR_SIZE[1]);
    });

    const opacity = opacityForIntensity(item.intensity, isHovered);
    const color = isHovered ? HOVER_BAR_COLOR : BASE_BAR_COLOR;

    return (
        <mesh
            ref={meshRef}
            castShadow
            receiveShadow
            onPointerEnter={(e) => {
                e.stopPropagation();
                onHover({
                    row: item.row,
                    col: item.col,
                    value: item.value,
                    position: [item.x, currentHeight.current, item.z],
                });
            }}
            onPointerLeave={(e) => {
                e.stopPropagation();
                onUnhover();
            }}
        >
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial
                color={color}
                transparent
                opacity={opacity}
                metalness={0.12}
                roughness={0.28}
            />
        </mesh>
    );
}

// Axis Labels Component
function AxisLabels({ gridSize }: { gridSize: number }) {
    const offset = (gridSize * 1.0) / 2;
    const axisLength = offset + 2;

    return (
        <group>
            {/* Axes Helper */}
            <axesHelper args={[axisLength]} />

            {/* Y-Axis Label (Count) */}
            <Text
                position={[-offset - 1.5, 4, -offset - 1.5]}
                fontSize={0.6}
                color="#2596be"
                anchorX="center"
                anchorY="middle"
                rotation={[0, Math.PI / 4, 0]}
            >
                Count
            </Text>

            {/* X-Axis Label (Actual Class) */}
            <Text
                position={[0, -0.8, offset + 1.5]}
                fontSize={0.5}
                color="#10b981"
                anchorX="center"
                anchorY="middle"
                rotation={[-Math.PI / 2, 0, 0]}
            >
                Actual Class →
            </Text>

            {/* Z-Axis Label (Predicted Class) */}
            <Text
                position={[offset + 1.5, -0.8, 0]}
                fontSize={0.5}
                color="#f59e0b"
                anchorX="center"
                anchorY="middle"
                rotation={[-Math.PI / 2, 0, Math.PI / 2]}
            >
                Predicted Class →
            </Text>

            {/* Y-Axis Arrow and Numbers */}
            <group position={[-offset - 0.5, 0, -offset - 0.5]}>
                {/* Vertical line */}
                <mesh position={[0, 3, 0]}>
                    <cylinderGeometry args={[0.03, 0.03, 6, 8]} />
                    <meshStandardMaterial color="#2596be" />
                </mesh>

                {/* Arrow head */}
                <mesh position={[0, 6.2, 0]} rotation={[0, 0, 0]}>
                    <coneGeometry args={[0.12, 0.4, 8]} />
                    <meshStandardMaterial color="#2596be" />
                </mesh>

                {/* Scale markers */}
                {[1, 2, 3, 4, 5].map((i) => (
                    <group key={i} position={[0, i, 0]}>
                        <mesh position={[-0.15, 0, 0]}>
                            <boxGeometry args={[0.3, 0.02, 0.02]} />
                            <meshStandardMaterial color="#6b7280" />
                        </mesh>
                    </group>
                ))}
            </group>
        </group>
    );
}

// Bars Grid Component
function BarsGrid({
    data,
    labels,
}: {
    data: number[][];
    labels: string[];
}) {
    const [hoveredBar, setHoveredBar] = useState<HoveredBar | null>(null);

    const { bars, gridSize } = useMemo(() => {
        const rows = data.length;
        const colsLocal = data[0]?.length || 0;

        // Find max value for height normalization
        let max = 0;
        data.forEach((row) =>
            row.forEach((val) => {
                if (val > max) max = val;
            })
        );

        const size = Math.max(rows, colsLocal);
        const spacing = 1.0;
        const offset = (size * spacing) / 2 - spacing / 2;

        const barList: BarItem[] = [];
        let id = 0;

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < colsLocal; j++) {
                const value = data[i][j];
                const intensity = max > 0 ? value / max : 0;
                const normalizedHeight = max > 0 ? intensity * MAX_BAR_HEIGHT : 0;

                barList.push({
                    id,
                    row: i,
                    col: j,
                    value,
                    intensity,
                    x: j * spacing - offset,
                    z: i * spacing - offset,
                    targetHeight: Math.max(normalizedHeight, MIN_BAR_HEIGHT),
                });
                id += 1;
            }
        }

        return { bars: barList, gridSize: size };
    }, [data]);

    const handleHover = (bar: HoveredBar) => setHoveredBar(bar);
    const handleUnhover = () => setHoveredBar(null);

    return (
        <group rotation={[0, 0, 0]}>
            {/* Axis Labels */}
            <AxisLabels gridSize={gridSize} />

            {/* Bars */}
            {bars.map((item) => (
                <BarCell
                    key={item.id}
                    item={item}
                    isHovered={hoveredBar?.row === item.row && hoveredBar?.col === item.col}
                    onHover={handleHover}
                    onUnhover={handleUnhover}
                />
            ))}

            {/* HTML Tooltip */}
            {hoveredBar && (
                <Html
                    position={[
                        hoveredBar.position[0],
                        hoveredBar.position[1] + 1.5,
                        hoveredBar.position[2],
                    ]}
                    center
                    style={{ pointerEvents: "none" }}
                >
                    <div
                        style={{
                            background: "rgba(255, 255, 255, 0.98)",
                            backdropFilter: "blur(12px)",
                            padding: "14px 18px",
                            borderRadius: "14px",
                            boxShadow: "0 12px 40px rgba(0,0,0,0.18)",
                            border: "1px solid rgba(37, 150, 190, 0.25)",
                            whiteSpace: "nowrap",
                            minWidth: "200px",
                        }}
                    >
                        <div style={{ fontSize: "13px", color: "#6b7280", marginBottom: "6px" }}>
                            <span style={{ color: "#2596be", fontWeight: 600 }}>Actual:</span>{" "}
                            {labels[hoveredBar.row]}
                        </div>
                        <div style={{ fontSize: "13px", color: "#6b7280", marginBottom: "10px" }}>
                            <span style={{ color: "#2596be", fontWeight: 600 }}>Predicted:</span>{" "}
                            {labels[hoveredBar.col]}
                        </div>
                        <div
                            style={{
                                fontSize: "24px",
                                fontWeight: 700,
                                color: "#2596be",
                            }}
                        >
                            Count: {hoveredBar.value}
                        </div>
                        <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "6px", fontWeight: 500 }}>
                            Higher bars appear darker
                        </div>
                    </div>
                </Html>
            )}

            {/* Grid Floor */}
            <gridHelper args={[14, 14, "#cbd5e1", "#e2e8f0"]} position={[0, -0.01, 0]} />
        </group>
    );
}

export function ConfusionMatrix3D({ data, labels = defaultClassLabels }: ConfusionMatrix3DProps) {
    const effectiveLabels = labels?.length ? labels : Array.from({ length: data.length }, (_, i) => `Class ${i + 1}`);
    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            className="relative w-full h-[600px] glass-card overflow-hidden"
        >
            {/* Title */}
            <div className="absolute top-6 left-6 z-10">
                <h3 className="text-lg font-bold" style={{ color: "#2596be" }}>
                    Confusion Matrix
                </h3>
                {/* <p className="text-sm text-gray-500">Height represents prediction frequency</p> */}
            </div>

            {/* Controls hint */}
            <div className="absolute top-6 right-6 z-10 glass-card px-4 py-3">
                <p className="text-xs text-gray-500">Drag to rotate • Scroll to zoom</p>
            </div>

            {/* Legend */}
            <div className="absolute bottom-6 left-6 z-10 glass-card px-5 py-4">
                <p className="text-xs text-gray-500 mb-3">Legend</p>
                <div className="flex items-center gap-2">
                    <div
                        className="w-24 h-3 rounded-full"
                        style={{ background: "linear-gradient(90deg, rgba(37,150,190,0.15), rgba(37,150,190,1))" }}
                    />
                    <span className="text-xs text-gray-600">Low → High</span>
                </div>
                {/* <p className="text-[11px] text-gray-500 mt-2">Color intensity follows bar height</p> */}
            </div>

            {/* Axis Guide */}
            <div className="absolute bottom-6 right-6 z-10 glass-card px-5 py-4">
                <p className="text-xs text-gray-500 mb-2">Axes</p>
                <div className="flex flex-col gap-1 text-xs">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-1 rounded" style={{ backgroundColor: "#ff0000" }} />
                        <span className="text-gray-600">X: Actual</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-1 rounded" style={{ backgroundColor: "#00ff00" }} />
                        <span className="text-gray-600">Y: Count</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-1 rounded" style={{ backgroundColor: "#0000ff" }} />
                        <span className="text-gray-600">Z: Predicted</span>
                    </div>
                </div>
                <p className="text-sm font-bold mt-2" style={{ color: "#2596be" }}>
                    {data.length} × {data[0]?.length || 0}
                </p>
            </div>

            {/* Canvas */}
            <Canvas shadows className="canvas-container">
                <PerspectiveCamera makeDefault position={[12, 10, 12]} fov={45} />
                <OrbitControls
                    enableDamping
                    dampingFactor={0.05}
                    minDistance={10}
                    maxDistance={35}
                    maxPolarAngle={Math.PI / 2.1}
                    minPolarAngle={Math.PI / 8}
                />

                {/* Lighting */}
                <ambientLight intensity={0.5} />
                <directionalLight
                    position={[15, 20, 15]}
                    intensity={1.2}
                    castShadow
                    shadow-mapSize={[2048, 2048]}
                />
                <directionalLight position={[-10, 15, -10]} intensity={0.4} />
                <pointLight position={[0, 12, 0]} intensity={0.6} color="#3db8e5" />
                <hemisphereLight
                    args={["#ffffff", "#e0f2fe", 0.4]}
                />

                {/* Bars Grid */}
                <BarsGrid data={data} labels={effectiveLabels} />

                {/* Background */}
                <color attach="background" args={["#f8fafc"]} />
            </Canvas>
        </motion.div>
    );
}

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import DeckGL from "@deck.gl/react";
import { GridLayer } from "@deck.gl/aggregation-layers";
import { Map } from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";

interface DataPoint {
    position: [number, number];
    weight: number;
    count: number;
    diseaseType: string;
}

interface DataVizGridProps {
    data: DataPoint[];
    title?: string;
}

const INITIAL_VIEW_STATE = {
    longitude: -122.4,
    latitude: 37.8,
    zoom: 11,
    pitch: 45,
    bearing: 0,
};

// Color scale for the grid
const COLOR_RANGE: [number, number, number][] = [
    [30, 58, 95],      // Deep blue
    [16, 185, 129],    // Emerald
    [52, 211, 153],    // Light emerald
    [251, 191, 36],    // Amber
    [249, 115, 22],    // Orange
    [239, 68, 68],     // Red
];

export function DataVizGrid({ data, title = "Disease Distribution" }: DataVizGridProps) {
    const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
    const [hoveredObject, setHoveredObject] = useState<{
        count: number;
        position: [number, number];
    } | null>(null);

    const layers = [
        new GridLayer({
            id: "grid-layer",
            data,
            pickable: true,
            extruded: true,
            cellSize: 200,
            elevationScale: 4,
            getPosition: (d: DataPoint) => d.position,
            getColorWeight: (d: DataPoint) => d.weight,
            getElevationWeight: (d: DataPoint) => d.count,
            colorRange: COLOR_RANGE,
            coverage: 0.8,
            onHover: (info: any) => {
                if (info?.object) {
                    setHoveredObject({
                        count: info.object.count,
                        position: info.object.position,
                    });
                } else {
                    setHoveredObject(null);
                }
                return true;
            },
        }),
    ];

    const onViewStateChange = useCallback(
        (params: any) => {
            setViewState(params.viewState);
        },
        []
    );

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="glass-card overflow-hidden"
        >
            {/* Header */}
            <div className="p-4 border-b border-white/10">
                <h3 className="text-lg font-bold gradient-text">{title}</h3>
                {/* <p className="text-sm text-white/50">
                    Interactive 3D grid visualization of sample distribution
                </p> */}
            </div>

            {/* Map container */}
            <div className="deck-container relative">
                <DeckGL
                    viewState={viewState}
                    onViewStateChange={onViewStateChange}
                    controller={true}
                    layers={layers}
                >
                    <Map
                        mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
                    />
                </DeckGL>

                {/* Hover tooltip */}
                {hoveredObject && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="absolute top-4 right-4 glass-card px-4 py-3"
                    >
                        <p className="text-sm text-white/70">Sample Count</p>
                        <p className="text-2xl font-bold gradient-text">
                            {hoveredObject.count}
                        </p>
                    </motion.div>
                )}

                {/* Controls hint */}
                <div className="absolute bottom-4 left-4 glass-card px-3 py-2">
                    <p className="text-xs text-white/50">
                        Drag to pan • Right-click to rotate • Scroll to zoom
                    </p>
                </div>
            </div>

            {/* Legend */}
            <div className="p-4 border-t border-white/10">
                <div className="flex items-center justify-between">
                    <span className="text-xs text-white/50">Density</span>
                    <div className="flex items-center gap-1">
                        {COLOR_RANGE.map((color, i) => (
                            <div
                                key={i}
                                className="w-6 h-3 first:rounded-l last:rounded-r"
                                style={{
                                    backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
                                }}
                            />
                        ))}
                    </div>
                    <span className="text-xs text-white/50">High</span>
                </div>
            </div>
        </motion.div>
    );
}

// Simple placeholder for when MapLibre is not available
export function DataVizGridPlaceholder({ title = "Disease Distribution" }: { title?: string }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="glass-card overflow-hidden"
        >
            <div className="p-4 border-b border-white/10">
                <h3 className="text-lg font-bold gradient-text">{title}</h3>
                <p className="text-sm text-white/50">
                    Interactive 3D grid visualization of sample distribution
                </p>
            </div>

            <div className="h-[400px] flex items-center justify-center bg-gradient-to-br from-slate-800/50 to-slate-900/50">
                <div className="text-center">
                    <p className="text-white/70">Interactive Map Visualization</p>
                    <p className="text-sm text-white/50 mt-2">
                        Deck.gl GridLayer with 3D elevation
                    </p>
                </div>
            </div>
        </motion.div>
    );
}

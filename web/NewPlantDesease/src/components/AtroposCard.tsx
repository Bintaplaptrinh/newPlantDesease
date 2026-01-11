import Atropos from "atropos/react";
import "atropos/atropos.css";
import { type ReactNode } from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: (string | undefined | null | false)[]) {
    return twMerge(clsx(inputs));
}

interface AtroposCardProps {
    children: ReactNode;
    className?: string;
    shadow?: boolean;
    highlight?: boolean;
    rotateXMax?: number;
    rotateYMax?: number;
}

export function AtroposCard({
    children,
    className,
    shadow = true,
    highlight = true,
    rotateXMax = 15,
    rotateYMax = 15,
}: AtroposCardProps) {
    return (
        <Atropos
            className={cn("atropos-card", className)}
            shadow={shadow}
            highlight={highlight}
            rotateXMax={rotateXMax}
            rotateYMax={rotateYMax}
            shadowScale={1.05}
            activeOffset={40}
        >
            <div className="atropos-scale">
                <div className="atropos-rotate">
                    <div className="atropos-inner">
                        {children}
                    </div>
                </div>
            </div>
        </Atropos>
    );
}

// Simple wrapper without 3D effect for consistency
export function Card({
    children,
    className,
}: {
    children: ReactNode;
    className?: string;
}) {
    return (
        <div className={cn("glass-card", className)}>
            {children}
        </div>
    );
}

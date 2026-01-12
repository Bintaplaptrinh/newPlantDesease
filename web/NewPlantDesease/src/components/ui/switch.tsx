import * as React from "react";

import { cn } from "@/lib/utils";

export type SwitchProps = Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "onChange"> & {
    checked?: boolean;
    defaultChecked?: boolean;
    onCheckedChange?: (checked: boolean) => void;
};

export const Switch = React.forwardRef<HTMLButtonElement, SwitchProps>(
    ({
        className,
        checked: checkedProp,
        defaultChecked,
        onCheckedChange,
        disabled,
        onClick,
        ...props
    }, ref) => {
        const isControlled = checkedProp !== undefined;
        const [checkedState, setCheckedState] = React.useState<boolean>(Boolean(defaultChecked));
        const checked = isControlled ? Boolean(checkedProp) : checkedState;

        const toggle = React.useCallback(() => {
            const next = !checked;
            if (!isControlled) setCheckedState(next);
            onCheckedChange?.(next);
        }, [checked, isControlled, onCheckedChange]);

        return (
            <button
                ref={ref}
                type="button"
                role="switch"
                aria-checked={checked}
                data-state={checked ? "checked" : "unchecked"}
                disabled={disabled}
                onClick={(e) => {
                    onClick?.(e);
                    if (e.defaultPrevented || disabled) return;
                    toggle();
                }}
                className={cn(
                    "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full border transition-colors",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3db8e5] focus-visible:ring-offset-2",
                    "ring-offset-white",
                    "backdrop-blur-md",
                    checked
                        ? "bg-[#3db8e5]/90 border-[#3db8e5] shadow-[0_0_0_3px_rgba(61,184,229,0.18)]"
                        : "bg-white/40 border-[#3db8e5]/35",
                    disabled && "opacity-50 cursor-not-allowed",
                    className
                )}
                {...props}
            >
                <span
                    className={cn(
                        "pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow transition-transform",
                        checked ? "translate-x-5" : "translate-x-0.5"
                    )}
                />
            </button>
        );
    }
);
Switch.displayName = "Switch";

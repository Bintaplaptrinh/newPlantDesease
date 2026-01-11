import { motion } from "framer-motion";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: (string | undefined | null | false)[]) {
    return twMerge(clsx(inputs));
}

interface NavBottomProps {
    activeSubSection: number;
    onSubSectionChange: (index: number) => void;
    subSections: { id: string; label: string }[];
    visible: boolean;
}

export function NavBottom({
    activeSubSection,
    onSubSectionChange,
    subSections,
    visible,
}: NavBottomProps) {
    if (!visible || subSections.length <= 1) return null;

    return (
        <motion.nav
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 100, opacity: 0 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50"
        >
            <div className="glass-nav rounded-full px-4 py-3 md:px-6 md:py-3.5 flex items-center gap-4 md:gap-6">
                {subSections.map((subSection, index) => (
                    <div key={subSection.id} className="flex items-center gap-4">
                        {/* Sub-section button */}
                        <motion.button
                            onClick={() => onSubSectionChange(index)}
                            className="relative group flex items-center gap-3"
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                        >
                            {/* Dot */}
                            <div
                                className={cn(
                                    "w-3 h-3 rounded-full transition-all duration-300",
                                    activeSubSection === index
                                        ? "shadow-[0_0_12px_rgba(37,150,190,0.5)]"
                                        : "bg-gray-300 group-hover:bg-gray-400"
                                )}
                                style={activeSubSection === index ? { background: 'linear-gradient(to right, #2596be, #3db8e5)' } : {}}
                            />

                            {/* Label */}
                            <span
                                className={cn(
                                    "text-sm font-medium transition-colors duration-300",
                                    activeSubSection === index
                                        ? "text-gray-800"
                                        : "text-gray-500 group-hover:text-gray-700"
                                )}
                            >
                                {subSection.label}
                            </span>

                            {/* Active underline */}
                            {activeSubSection === index && (
                                <motion.div
                                    layoutId="active-subsection"
                                    className="absolute -bottom-2 left-0 right-0 h-0.5"
                                    style={{ background: 'linear-gradient(to right, #2596be, #3db8e5)' }}
                                    initial={false}
                                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                                />
                            )}
                        </motion.button>

                        {/* Separator */}
                        {index < subSections.length - 1 && (
                            <div className="w-px h-4 bg-gray-200" />
                        )}
                    </div>
                ))}
            </div>
        </motion.nav>
    );
}

import { motion } from "framer-motion";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { type LucideIcon } from "lucide-react";

function cn(...inputs: (string | undefined | null | false)[]) {
    return twMerge(clsx(inputs));
}

interface NavLeftProps {
    activeSection: number;
    onSectionChange: (index: number) => void;
    sections: { id: string; label: string; icon: LucideIcon }[];
}

export function NavLeft({ activeSection, onSectionChange, sections }: NavLeftProps) {
    return (
        <motion.nav
            initial={{ x: -100, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="fixed left-6 top-1/2 -translate-y-1/2 z-50 flex flex-col items-center gap-4"
        >
            {/* Background glass panel */}
            <div className="glass-nav rounded-3xl p-4 flex flex-col items-center gap-4">
                {sections.map((section, index) => {
                    const Icon = section.icon;
                    return (
                        <div key={section.id} className="relative group">
                            {/* Tooltip */}
                            <motion.div
                                initial={{ opacity: 0, x: -10 }}
                                whileHover={{ opacity: 1, x: 0 }}
                                className="absolute left-14 top-1/2 -translate-y-1/2 pointer-events-none"
                            >
                                <div className="glass-nav rounded-2xl px-3 py-1.5 whitespace-nowrap text-sm font-medium text-gray-800">
                                    {section.label}
                                </div>
                            </motion.div>

                            {/* Icon button */}
                            <motion.button
                                onClick={() => onSectionChange(index)}
                                className={cn(
                                    "w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300",
                                    activeSection === index
                                        ? "bg-gradient-to-br from-[#2596be] to-[#3db8e5] text-white shadow-lg"
                                        : "bg-gray-100 text-gray-500 hover:bg-gray-200"
                                )}
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.95 }}
                                aria-label={`Navigate to ${section.label}`}
                                style={activeSection === index ? { boxShadow: '0 4px 15px rgba(37, 150, 190, 0.4)' } : {}}
                            >
                                <Icon className="w-5 h-5" />
                            </motion.button>

                            {/* Connection line to next dot */}
                            {index < sections.length - 1 && (
                                <div
                                    className={cn(
                                        "w-0.5 h-6 mx-auto mt-2 transition-colors duration-300",
                                        activeSection > index
                                            ? "bg-gradient-to-b from-[#2596be] to-[#3db8e5]"
                                            : "bg-gray-200"
                                    )}
                                />
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Section counter */}
            <motion.div
                className="glass-nav rounded-2xl px-3 py-2 text-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
            >
                <div className="text-xs text-gray-500 uppercase tracking-wider">Section</div>
                <div className="text-lg font-bold" style={{ color: '#2596be' }}>
                    {String(activeSection + 1).padStart(2, "0")}
                </div>
            </motion.div>
        </motion.nav>
    );
}

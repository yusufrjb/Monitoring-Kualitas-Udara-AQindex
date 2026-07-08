"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { ChevronLeft, ChevronRight, Activity, Calendar } from "lucide-react";
import { cn } from "@/lib/utils";

interface HeatmapData {
    date: string; // YYYY-MM-DD
    avg_pm25: number;
}

const MONTHS = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
];

const DAYS = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"];

export default function HeatmapCalendar() {
    const [currentYear, setCurrentYear] = useState<number | null>(null);
    const [currentMonth, setCurrentMonth] = useState<number | null>(null);
    const [data, setData] = useState<HeatmapData[]>([]);
    const [loading, setLoading] = useState(false);
    const [hoveredDay, setHoveredDay] = useState<{
        dateStr: string;
        displayDate: string;
        value: number | null;
        x: number;
        y: number;
    } | null>(null);

    useEffect(() => {
        const today = new Date();
        setCurrentYear(today.getFullYear());
        setCurrentMonth(today.getMonth() + 1);
    }, []);

    const fetchData = useCallback(async (year: number, month: number) => {
        setLoading(true);
        try {
            const res = await fetch(`/api/aggregates/daily-pm25?year=${year}&month=${month}`);
            if (!res.ok) throw new Error("Gagal mengambil data kalender");
            const json = await res.json();
            setData(json);
        } catch (err) {
            console.error(err);
            setData([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (currentYear !== null && currentMonth !== null) {
            fetchData(currentYear, currentMonth);
        }
    }, [currentYear, currentMonth, fetchData]);

    const prevMonth = () => {
        if (currentMonth === null) return;
        if (currentMonth === 1) {
            setCurrentMonth(12);
            setCurrentYear((y) => (y !== null ? y - 1 : y));
        } else {
            setCurrentMonth((m) => (m !== null ? m - 1 : m));
        }
    };

    const nextMonth = () => {
        if (currentMonth === null) return;
        if (currentMonth === 12) {
            setCurrentMonth(1);
            setCurrentYear((y) => (y !== null ? y + 1 : y));
        } else {
            setCurrentMonth((m) => (m !== null ? m + 1 : m));
        }
    };

    // Generate calendar grid (6 rows, 7 cols)
    const calendarDays = useMemo(() => {
        if (currentYear === null || currentMonth === null) return [];

        const firstDayObj = new Date(currentYear, currentMonth - 1, 1);
        const lastDayObj = new Date(currentYear, currentMonth, 0);
        const daysInMonth = lastDayObj.getDate();

        // adjust so Monday is first column
        let startingDayOfWeek = firstDayObj.getDay();
        if (startingDayOfWeek === 0) startingDayOfWeek = 7;
        startingDayOfWeek -= 1; // 0 index for Monday

        const days = [];

        // Padding before 1st of month
        for (let i = 0; i < startingDayOfWeek; i++) {
            days.push({ day: null, dateStr: null, value: null });
        }

        // Actual days
        for (let i = 1; i <= daysInMonth; i++) {
            const dateStr = `${currentYear}-${String(currentMonth).padStart(2, "0")}-${String(i).padStart(2, "0")}`;
            const match = data.find(d => d.date === dateStr);
            days.push({
                day: i,
                dateStr,
                value: match ? match.avg_pm25 : null
            });
        }

        // Padding after end of month
        while (days.length % 7 !== 0) {
            days.push({ day: null, dateStr: null, value: null });
        }

        return days;
    }, [currentYear, currentMonth, data]);

    const getColorClass = (val: number | null) => {
        if (val === null) return "bg-slate-100 hover:bg-slate-200";
        if (val <= 50) return "bg-emerald-500 hover:bg-emerald-600";
        if (val <= 100) return "bg-blue-500 hover:bg-blue-600";
        if (val <= 200) return "bg-amber-500 hover:bg-amber-600";
        if (val <= 300) return "bg-red-500 hover:bg-red-600";
        return "bg-purple-600 hover:bg-purple-700";
    };

    const getStatusText = (val: number) => {
        if (val <= 50) return "Baik";
        if (val <= 100) return "Sedang";
        if (val <= 200) return "Tidak Sehat";
        if (val <= 300) return "Sangat Tidak Sehat";
        return "Berbahaya";
    };

    // Tooltip handler
    const handleMouseEnter = (e: React.MouseEvent, day: any) => {
        if (!day.day) return;

        const d = new Date(day.dateStr);
        const displayDate = d.toLocaleDateString('id-ID', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });

        const rect = e.currentTarget.getBoundingClientRect();

        setHoveredDay({
            dateStr: day.dateStr,
            displayDate,
            value: day.value,
            x: rect.left + rect.width / 2,
            y: rect.top - 8
        });
    };

    return (
        <div className="rounded-xl border border-border bg-card p-4 shadow-sm flex flex-col h-full relative">
            <div className="mb-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                    <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground">
                        <Calendar size={15} className="text-muted-foreground" />
                        Kalender Kontribusi Bulan (ISPU PM2.5)
                    </h3>
                    <p className="mt-0.5 text-xs text-muted-foreground">Rata-rata harian dari agregasi hourly</p>
                </div>

                {/* Navigation */}
                <div className="flex items-center gap-1.5 shrink-0">
                    <button onClick={prevMonth} className="p-1 rounded-md bg-muted/40 hover:bg-muted/80 text-muted-foreground transition">
                        <ChevronLeft size={16} />
                    </button>
                    <div className="text-[11px] font-medium w-28 text-center bg-slate-50 px-2 py-1 rounded-md border border-border">
                        {currentMonth !== null && currentYear !== null ? `${MONTHS[currentMonth - 1]} ${currentYear}` : "--"}
                    </div>
                    <button onClick={nextMonth} className="p-1 rounded-md bg-muted/40 hover:bg-muted/80 text-muted-foreground transition" disabled={currentMonth === null || currentYear === null ? false : currentMonth === new Date().getMonth() + 1 && currentYear === new Date().getFullYear()}>
                        <ChevronRight size={16} />
                    </button>
                </div>
            </div>

            {loading || currentYear === null || currentMonth === null ? (
                <div className="flex-1 w-full flex items-center justify-center min-h-[160px]">
                    <div className="h-6 w-6 rounded-full border-2 border-muted border-t-primary animate-spin" />
                </div>
            ) : (
                <div className="flex-1 flex flex-col justify-center">
                    <div className="w-full max-w-[320px] mx-auto">
                        {/* Days header */}
                        <div className="grid grid-cols-7 gap-1 sm:gap-1.5 mb-1.5">
                            {DAYS.map(d => (
                                <div key={d} className="text-center text-[9px] font-medium text-slate-400 uppercase tracking-wider">
                                    {d}
                                </div>
                            ))}
                        </div>

                        {/* Grid */}
                        <div className="grid grid-cols-7 gap-1 sm:gap-1.5">
                            {calendarDays.map((d, i) => (
                                <div
                                    key={`day-${i}`}
                                    onMouseEnter={(e) => handleMouseEnter(e, d)}
                                    onMouseLeave={() => setHoveredDay(null)}
                                    className={cn(
                                        "aspect-square rounded flex items-center justify-center text-xs font-medium cursor-default transition-transform transform",
                                        !d.day ? "bg-transparent pointer-events-none" : getColorClass(d.value),
                                        d.day ? "hover:scale-105 shadow-sm border border-black/5" : "",
                                        d.day && d.value !== null && d.value > 50 ? "text-white" : "text-slate-700"
                                    )}
                                >
                                    {d.day}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Legend */}
            <div className="mt-4 pt-4 border-t border-border flex flex-wrap gap-x-3 gap-y-1.5 items-center justify-center text-[9px] sm:text-[10px]">
                <div className="text-muted-foreground font-medium">ISPU:</div>
                {[
                    { label: "Baik", col: "bg-emerald-500" },
                    { label: "Sedang", col: "bg-blue-500" },
                    { label: "T.Sehat", col: "bg-amber-500" },
                    { label: "Sangat T.Sehat", col: "bg-red-500" },
                    { label: "Berbahaya", col: "bg-purple-600" },
                ].map(leg => (
                    <div key={leg.label} className="flex items-center gap-1.5">
                        <span className={`w-3 h-3 rounded-sm ${leg.col} opacity-90`} />
                        <span className="text-slate-600">{leg.label}</span>
                    </div>
                ))}
            </div>

            {/* Tooltip Overlay */}
            {hoveredDay && (
                <div
                    className="fixed z-50 pointer-events-none -translate-x-1/2 -translate-y-full transform bg-slate-900 border border-slate-700 text-white px-3 py-2 rounded-lg shadow-xl"
                    style={{ left: hoveredDay.x, top: hoveredDay.y }}
                >
                    <p className="text-xs font-semibold whitespace-nowrap mb-0.5">{hoveredDay.displayDate}</p>
                    {hoveredDay.value !== null ? (
                        <p className="text-[11px] text-slate-300">
                            ISPU: <span className="text-white font-medium">{hoveredDay.value.toFixed(0)}</span>
                            <span className="mx-1.5 text-slate-500">•</span>
                            <span className={getColorClass(hoveredDay.value).split(' ')[0].replace('bg-', 'text-')}>{getStatusText(hoveredDay.value)}</span>
                        </p>
                    ) : (
                        <p className="text-[11px] text-slate-400">Tidak ada data sensor</p>
                    )}
                </div>
            )}
        </div>
    );
}

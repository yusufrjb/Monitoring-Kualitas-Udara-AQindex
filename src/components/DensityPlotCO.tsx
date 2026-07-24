"use client";

import React, { useEffect, useState } from "react";
import { AreaChart, Area, XAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

interface BinData {
    value: number;
    count: number;
}

export default function DensityPlotCO({ days = 7 }: { days?: number }) {
    const [bins, setBins] = useState<BinData[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function fetchData() {
            try {
                const res = await fetch(`/api/aggregates/co-density?days=${days}`);
                if (!res.ok) throw new Error("Gagal mengambil data sebaran CO");
                const data = await res.json();
                setBins(data.bins);
            } catch (err: unknown) {
                setError(err instanceof Error ? err.message : "Terjadi kesalahan");
            } finally {
                setLoading(false);
            }
        }
        fetchData();
    }, [days]);

    if (loading) {
        return (
            <div className="flex h-40 items-center justify-center rounded-xl border border-border bg-card p-4 animate-pulse">
                <p className="text-xs font-medium text-muted-foreground">Memuat Data CO...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex h-40 items-center justify-center rounded-xl border border-red-200 bg-red-50 p-4">
                <p className="text-xs font-medium text-red-600">Gagal memuat sebaran CO: {error}</p>
            </div>
        );
    }

    const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: BinData }> }) => {
        if (active && payload && payload.length) {
            const d = payload[0].payload;
            return (
                <div className="bg-white border border-gray-200 rounded-lg shadow-lg px-2.5 py-1.5">
                    <p className="text-[11px] font-semibold text-gray-700">ISPU: {d.value}</p>
                    <p className="text-[10px] text-gray-500">Frekuensi: {d.count}</p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="rounded-xl border border-border bg-card p-3 sm:p-4 shadow-sm max-w-full overflow-hidden flex flex-col w-full h-full">
            <div className="mb-2">
                <h3 className="text-xs font-semibold text-foreground">Sebaran CO (ISPU)</h3>
                <p className="mt-0.5 text-[10px] text-muted-foreground">120 Data Terakhir</p>
            </div>

            <div className="flex-1 min-h-[180px]">
                <ResponsiveContainer width="100%" height={180}>
                    <AreaChart data={bins} margin={{ top: 5, right: 5, left: -10, bottom: 0 }}>
                        <defs>
                            <linearGradient id="coGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.5} />
                                <stop offset="100%" stopColor="#fbbf24" stopOpacity={0.05} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
                        <XAxis
                            dataKey="value"
                            type="number"
                            domain={['dataMin', 'dataMax']}
                            tick={{ fontSize: 9, fill: "#9ca3af" }}
                            tickFormatter={(v: number) => v.toFixed(0)}
                            tickLine={false}
                            axisLine={false}
                        />
                        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,0,0,0.04)' }} />
                        <Area
                            type="monotone"
                            dataKey="count"
                            stroke="#f59e0b"
                            strokeWidth={1.5}
                            fill="url(#coGrad)"
                            isAnimationActive={false}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            <div className="flex justify-between items-center mt-1 text-[9px] text-muted-foreground">
                <span>ISPU CO →</span>
                <span>← Frekuensi</span>
            </div>
        </div>
    );
}

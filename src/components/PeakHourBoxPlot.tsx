"use client";

import React, { useEffect, useState } from "react";
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis, Tooltip, Cell, ReferenceLine } from "recharts";

interface HistogramData {
    range: string;
    min: number;
    max: number;
    count: number;
    category: string;
    color: string;
}

export default function PeakHourDistribution({ days = 7, refreshKey = 0 }: { days?: number, refreshKey?: number }) {
    const [histogramData, setHistogramData] = useState<HistogramData[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [totalCount, setTotalCount] = useState(0);

    useEffect(() => {
        async function fetchData() {
            try {
                const res = await fetch(`/api/aggregates/peak-hour-distribution?days=${days}`);
                if (!res.ok) throw new Error("Gagal mengambil data distribusi");
                const data = await res.json();
                
                // Create histogram bins from box plot data
                const bins = createHistogramBins(data);
                setHistogramData(bins.data);
                setTotalCount(bins.totalCount);
            } catch (err: any) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        }
        fetchData();
    }, [days, refreshKey]);

    // Create histogram bins based on ISPU categories
    function createHistogramBins(data: any) {
        const bins: HistogramData[] = [
            { range: "0-50", min: 0, max: 50, count: 0, category: "Baik", color: "#10b981" },
            { range: "51-100", min: 51, max: 100, count: 0, category: "Sedang", color: "#3b82f6" },
            { range: "101-200", min: 101, max: 200, count: 0, category: "Tidak Sehat", color: "#f59e0b" },
            { range: "201-300", min: 201, max: 300, count: 0, category: "Sangat Tidak Sehat", color: "#ef4444" },
            { range: "301-500", min: 301, max: 500, count: 0, category: "Berbahaya", color: "#7c3aed" },
        ];

        // Count data points in each bin
        const allValues = [
            data.min, data.q1, data.median, data.q3, data.max,
            ...data.outliers
        ].filter(v => v !== undefined && v !== null && !isNaN(v));

        allValues.forEach(val => {
            for (const bin of bins) {
                if (val >= bin.min && val <= bin.max) {
                    bin.count++;
                    break;
                }
            }
        });

        return {
            data: bins,
            totalCount: allValues.length
        };
    }

    const maxCount = Math.max(...histogramData.map(d => d.count), 1);

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            const percentage = totalCount > 0 ? ((data.count / totalCount) * 100).toFixed(1) : 0;
            return (
                <div className="bg-white border border-gray-200 rounded-lg shadow-lg px-3 py-2">
                    <p className="text-xs font-semibold text-gray-700 mb-1">{data.category}</p>
                    <p className="text-xs text-gray-600">Rentang ISPU: {data.range}</p>
                    <p className="text-sm font-bold text-gray-800">{data.count} data ({percentage}%)</p>
                </div>
            );
        }
        return null;
    };

    if (loading) {
        return (
            <div className="flex h-40 items-center justify-center rounded-xl border border-border bg-card p-5 animate-pulse">
                <p className="text-sm font-medium text-muted-foreground">Menghitung Data Statistik...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex h-40 items-center justify-center rounded-xl border border-red-200 bg-red-50 p-5">
                <p className="text-sm font-medium text-red-600">Gagal memuat distribusi: {error}</p>
            </div>
        );
    }

    return (
        <div className="rounded-xl border border-border bg-card p-4 sm:p-5 shadow-sm max-w-full overflow-hidden flex flex-col w-full h-full">
            <div className="mb-4">
                <h3 className="text-sm font-semibold text-foreground">Distribusi PM2.5 (Jam Sibuk)</h3>
                <p className="mt-0.5 text-[11px] text-muted-foreground">
                    Rentang 07:00 - 09:00 WIB (Senin - Jumat) selama 7 Hari Terakhir · Total: {totalCount} data
                </p>
            </div>

            <div className="flex-1 flex flex-col justify-center min-h-[160px]">
                <ResponsiveContainer width="100%" height={160}>
                    <BarChart data={histogramData} margin={{ top: 10, right: 10, left: 0, bottom: 30 }}>
                        <XAxis 
                            dataKey="range" 
                            axisLine={false} 
                            tickLine={false}
                            tick={{ fontSize: 9, fill: "#9ca3af" }}
                            dy={15}
                        />
                        <YAxis 
                            axisLine={false} 
                            tickLine={false}
                            tick={{ fontSize: 10, fill: "#9ca3af" }}
                            width={30}
                        />
                        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0, 0, 0, 0.05)' }} />
                        <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                            {histogramData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                        </Bar>
                        <ReferenceLine y={0} stroke="#e5e7eb" strokeWidth={1} />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Legend - similar to heatmap */}
            <div className="mt-3 pt-3 border-t border-border flex flex-wrap gap-x-2 gap-y-1.5 items-center justify-center text-[9px] sm:text-[10px]">
                <div className="text-muted-foreground font-medium">ISPU:</div>
                {histogramData.map(bin => (
                    <div key={bin.range} className="flex items-center gap-1">
                        <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: bin.color }} />
                        <span className="text-slate-600">{bin.category}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

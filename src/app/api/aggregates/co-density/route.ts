import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';
import { withCache } from '@/lib/cache';

export async function GET() {
    try {
        const result = await withCache("co-density", 5 * 60 * 1000, async () => {
            const { data, error } = await supabase
                .from('tb_prediksi_kualitas_udara')
                .select('created_at, co_ispu')
                .order('created_at', { ascending: false })
                .limit(120);

            if (error) throw error;

            const rows = data || [];

            const coISPUValues: number[] = [];
            for (const row of rows) {
                if (row.co_ispu != null) {
                    coISPUValues.push(row.co_ispu);
                }
            }

            if (coISPUValues.length === 0) {
                return { bins: [], stats: { mean: 0, median: 0, max: 0, min: 0, count: 0 } };
            }

            coISPUValues.sort((a, b) => a - b);

            const avg = coISPUValues.reduce((sum, val) => sum + val, 0) / coISPUValues.length;
            const median = coISPUValues[Math.floor(coISPUValues.length / 2)];
            const min = coISPUValues[0];
            const max = coISPUValues[coISPUValues.length - 1];

            const numBins = 20;
            const range = max - min || 1;
            const binSize = range / numBins;
            const bins: { value: number; count: number }[] = [];

            for (let i = 0; i < numBins; i++) {
                const center = min + (i + 0.5) * binSize;
                bins.push({ value: Math.round(center * 10) / 10, count: 0 });
            }

            coISPUValues.forEach(val => {
                const idx = Math.min(Math.floor((val - min) / binSize), numBins - 1);
                bins[idx].count++;
            });

            return {
                bins,
                stats: {
                    mean: Math.round(avg),
                    median: Math.round(median),
                    max: Math.round(max),
                    min: Math.round(min),
                    count: coISPUValues.length
                }
            };
        });

        return NextResponse.json(result);
    } catch (error) {
        console.error("Error generating CO density:", error);
        return NextResponse.json({ error: "Failed to generate CO density" }, { status: 500 });
    }
}

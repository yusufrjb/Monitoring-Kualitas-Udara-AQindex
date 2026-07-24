import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';
import { withCache } from '@/lib/cache';

function pm25ToISPU(ugm3: number): number {
    if (ugm3 <= 15) return Math.round((ugm3 / 15) * 50);
    if (ugm3 <= 35) return Math.round(50 + ((ugm3 - 15) / 20) * 50);
    if (ugm3 <= 55) return Math.round(100 + ((ugm3 - 35) / 20) * 100);
    if (ugm3 <= 150) return Math.round(200 + ((ugm3 - 55) / 95) * 100);
    if (ugm3 <= 250) return Math.round(300 + ((ugm3 - 150) / 100) * 100);
    return Math.round(400 + ((ugm3 - 250) / 100) * 100);
}

function percentile(arr: number[], p: number) {
    if (arr.length === 0) return 0;
    if (p <= 0) return arr[0];
    if (p >= 100) return arr[arr.length - 1];
    const index = (arr.length - 1) * p / 100;
    const lower = Math.floor(index);
    const upper = lower + 1;
    const weight = index - lower;
    if (upper >= arr.length) return arr[lower];
    return arr[lower] * (1 - weight) + arr[upper] * weight;
}

export async function GET(request: Request) {
    try {
        const { searchParams } = new URL(request.url);
        const days = parseInt(searchParams.get('days') || '7');

        const result = await withCache(`peak-hour-${days}`, 10 * 60 * 1000, async () => {
            const since = new Date();
            since.setDate(since.getDate() - days);

            const { data, error } = await supabase
                .from('air_quality_hourly_agg')
                .select('time, pm25_ugm3')
                .gte('time', since.toISOString())
                .order('time', { ascending: true });

            if (error) throw error;

            const rows = data || [];
            const peakHourValues: number[] = [];

            for (const row of rows) {
                if (!row.time || row.pm25_ugm3 == null) continue;

                const date = new Date(row.time);
                const dateInJakarta = new Date(date.toLocaleString('en-US', { timeZone: 'Asia/Jakarta' }));
                const hour = dateInJakarta.getHours();
                const day = dateInJakarta.getDay();

                if (isNaN(hour) || isNaN(day)) continue;

                const isWeekday = day >= 1 && day <= 5;
                if (isWeekday && hour >= 7 && hour < 9) {
                    const ispuValue = pm25ToISPU(row.pm25_ugm3);
                    peakHourValues.push(ispuValue);
                }
            }

            peakHourValues.sort((a, b) => a - b);

            if (peakHourValues.length === 0) {
                return { min: 0, q1: 0, median: 0, q3: 0, max: 0, outliers: [] };
            }

            const q1 = percentile(peakHourValues, 25);
            const median = percentile(peakHourValues, 50);
            const q3 = percentile(peakHourValues, 75);
            const iqr = q3 - q1;

            const lowerBound = q1 - 1.5 * iqr;
            const upperBound = q3 + 1.5 * iqr;

            const normalValues = peakHourValues.filter(v => v >= lowerBound && v <= upperBound);
            const outliers = peakHourValues.filter(v => v < lowerBound || v > upperBound);

            const min = normalValues.length > 0 ? normalValues[0] : q1;
            const max = normalValues.length > 0 ? normalValues[normalValues.length - 1] : q3;

            return {
                min: Number(min.toFixed(2)),
                q1: Number(q1.toFixed(2)),
                median: Number(median.toFixed(2)),
                q3: Number(q3.toFixed(2)),
                max: Number(max.toFixed(2)),
                outliers: outliers.map(v => Number(v.toFixed(2)))
            };
        });

        return NextResponse.json(result);
    } catch (error) {
        console.error("Error generating peak hour distribution:", error);
        return NextResponse.json({ error: "Failed to generate distribution" }, { status: 500 });
    }
}

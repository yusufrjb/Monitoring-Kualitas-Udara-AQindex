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

export async function GET(request: Request) {
    try {
        const { searchParams } = new URL(request.url);
        const month = parseInt(searchParams.get('month') || String(new Date().getMonth() + 1));
        const year = parseInt(searchParams.get('year') || String(new Date().getFullYear()));

        const startDate = new Date(`${year}-${String(month).padStart(2, "0")}-01T00:00:00.000Z`);
        const endDate = new Date(year, month, 0, 23, 59, 59, 999);

        const result = await withCache(`daily-pm25-${year}-${month}`, 5 * 60 * 1000, async () => {
            const { data, error } = await supabase
                .from('air_quality_hourly_agg')
                .select('time, pm25_ugm3')
                .gte('time', startDate.toISOString())
                .lte('time', endDate.toISOString())
                .order('time', { ascending: true });

            if (error) throw error;

            const daysMap: Record<string, { values: number[]; sum: number; count: number }> = {};
            (data || []).forEach(r => {
                if (!r.time || r.pm25_ugm3 == null) return;

                const date = new Date(r.time);
                const dateInJakarta = new Date(date.toLocaleString('en-US', { timeZone: 'Asia/Jakarta' }));

                const y = dateInJakarta.getFullYear();
                const m = String(dateInJakarta.getMonth() + 1).padStart(2, '0');
                const dayLine = String(dateInJakarta.getDate()).padStart(2, '0');
                const dateStr = `${y}-${m}-${dayLine}`;

                if (!daysMap[dateStr]) {
                    daysMap[dateStr] = { values: [], sum: 0, count: 0 };
                }

                const ispuValue = pm25ToISPU(r.pm25_ugm3);
                daysMap[dateStr].values.push(ispuValue);
                daysMap[dateStr].sum += ispuValue;
                daysMap[dateStr].count++;
            });

            return Object.keys(daysMap).map(k => {
                const dayData = daysMap[k];
                const avg = dayData.sum / dayData.count;
                return {
                    date: k,
                    avg_pm25: Number(avg.toFixed(2))
                };
            }).sort((a, b) => a.date.localeCompare(b.date));
        });

        return NextResponse.json(result);
    } catch (error) {
        console.error("Error generating daily heatmap data:", error);
        return NextResponse.json({ error: "Failed to fetch data" }, { status: 500 });
    }
}

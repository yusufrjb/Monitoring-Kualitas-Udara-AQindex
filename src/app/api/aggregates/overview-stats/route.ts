import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';
import { withCache } from '@/lib/cache';

export async function GET(request: Request) {
    try {
        const { searchParams } = new URL(request.url);
        const days = parseInt(searchParams.get('days') || '7');

        const result = await withCache(`overview-stats-${days}`, 5 * 60 * 1000, async () => {
            const since = new Date();
            since.setDate(since.getDate() - days);

            const [hourlyAggResult, coDensityResult] = await Promise.all([
                supabase
                    .from('air_quality_hourly_agg')
                    .select('time, pm25_ugm3, pm10_corrected_ugm3, no2_ugm3, co_ugm3')
                    .gte('time', since.toISOString())
                    .order('time', { ascending: true }),
                supabase
                    .from('tb_prediksi_kualitas_udara')
                    .select('created_at, co_ispu')
                    .order('created_at', { ascending: false })
                    .limit(120),
            ]);

            if (hourlyAggResult.error) throw hourlyAggResult.error;
            if (coDensityResult.error) throw coDensityResult.error;

            return {
                hourlyAgg: hourlyAggResult.data || [],
                coDensity: coDensityResult.data || [],
            };
        });

        return NextResponse.json(result);
    } catch (error) {
        console.error("Error fetching overview stats:", error);
        return NextResponse.json({ error: "Failed to fetch overview stats" }, { status: 500 });
    }
}

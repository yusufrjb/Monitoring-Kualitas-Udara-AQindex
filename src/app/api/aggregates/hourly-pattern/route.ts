import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';

function pm25ToISPU(ugm3: number): number {
    if (ugm3 <= 15.5) return Math.round((ugm3 / 15.5) * 50);
    if (ugm3 <= 55.4) return Math.round(50 + ((ugm3 - 15.5) / 39.9) * 50);
    if (ugm3 <= 150.4) return Math.round(100 + ((ugm3 - 55.4) / 95) * 100);
    if (ugm3 <= 250.4) return Math.round(200 + ((ugm3 - 150.4) / 100) * 100);
    return Math.round(300 + ((ugm3 - 250.4) / 249.6) * 200);
}

export async function GET(request: Request) {
    try {
        const { searchParams } = new URL(request.url);
        const days = parseInt(searchParams.get('days') || '90');

        const since = new Date();
        since.setDate(since.getDate() - days);

        const { data, error } = await supabase
            .from('air_quality_hourly_agg')
            .select('time, pm25_ugm3')
            .gte('time', since.toISOString())
            .order('time', { ascending: true });

        if (error) throw error;

        const rows = data || [];
        console.log('Hourly agg data:', {
            totalRows: rows.length,
            dateRange: { from: since.toISOString(), to: new Date().toISOString() }
        });

        const hourlyData: Record<number, {
            weekday: number[];
            weekend: number[];
        }> = {};

        for (let h = 0; h < 24; h++) {
            hourlyData[h] = { weekday: [], weekend: [] };
        }

        for (const row of rows) {
            if (!row.time || row.pm25_ugm3 == null) continue;

            const date = new Date(row.time);
            const dateInJakarta = new Date(date.toLocaleString('en-US', { timeZone: 'Asia/Jakarta' }));

            const dayOfWeek = dateInJakarta.getDay();
            const hour = dateInJakarta.getHours();
            const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;

            const ispuValue = pm25ToISPU(row.pm25_ugm3);

            if (isWeekend) {
                hourlyData[hour].weekend.push(ispuValue);
            } else {
                hourlyData[hour].weekday.push(ispuValue);
            }
        }

        const hourlyAverages = Array.from({ length: 24 }, (_, hour) => {
            const weekdayValues = hourlyData[hour].weekday;
            const weekendValues = hourlyData[hour].weekend;

            const wDayAvg = weekdayValues.length > 0
                ? weekdayValues.reduce((a, b) => a + b, 0) / weekdayValues.length
                : 0;

            const wEndAvg = weekendValues.length > 0
                ? weekendValues.reduce((a, b) => a + b, 0) / weekendValues.length
                : 0;

            return {
                hour,
                label: `${String(hour).padStart(2, "0")}:00`,
                weekday_avg: Number(wDayAvg.toFixed(2)),
                weekend_avg: Number(wEndAvg.toFixed(2))
            };
        });

        const weekendDataPoints = Object.values(hourlyData).reduce((sum, d) => sum + d.weekend.length, 0);
        const weekdayDataPoints = Object.values(hourlyData).reduce((sum, d) => sum + d.weekday.length, 0);
        const weekendHoursWithData = hourlyAverages.filter(h => h.weekend_avg > 0).length;
        const weekdayHoursWithData = hourlyAverages.filter(h => h.weekday_avg > 0).length;

        const weekendDaysSet = new Set<string>();
        const weekdayDaysSet = new Set<string>();

        for (const row of rows) {
            if (!row.time || row.pm25_ugm3 == null) continue;
            const date = new Date(row.time);
            const dateInJakarta = new Date(date.toLocaleString('en-US', { timeZone: 'Asia/Jakarta' }));
            const dayOfWeek = dateInJakarta.getDay();
            const year = dateInJakarta.getFullYear();
            const month = String(dateInJakarta.getMonth() + 1).padStart(2, '0');
            const day = String(dateInJakarta.getDate()).padStart(2, '0');
            const dateStr = `${year}-${month}-${day}`;
            if (dayOfWeek === 0 || dayOfWeek === 6) {
                weekendDaysSet.add(dateStr);
            } else {
                weekdayDaysSet.add(dateStr);
            }
        }

        return NextResponse.json({
            data: hourlyAverages,
            meta: {
                hasWeekendData: weekendDataPoints > 0,
                weekendDataPoints,
                weekdayDataPoints,
                weekendHoursWithData,
                weekdayHoursWithData,
                totalWeekendDays: weekendDaysSet.size,
                totalWeekdayDays: weekdayDaysSet.size,
                dateRange: {
                    from: since.toISOString(),
                    to: new Date().toISOString()
                }
            }
        });
    } catch (error) {
        console.error("Error generating hourly pattern:", error);
        return NextResponse.json({ error: "Failed to generate pattern" }, { status: 500 });
    }
}

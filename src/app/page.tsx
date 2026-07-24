"use client";

import { useEffect, useState } from "react";
import OverviewTab from "@/components/OverviewTab";
import { supabase } from "@/lib/supabase";

import { RealtimeData } from "@/types";

export default function Home() {
  const [realtimeData, setRealtimeData] = useState<RealtimeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [currentTime, setCurrentTime] = useState<Date | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [gasResponse, predResponse] = await Promise.all([
          supabase
            .from("tb_konsentrasi_gas")
            .select("temperature, humidity, created_at")
            .order("created_at", { ascending: false })
            .limit(1)
            .single(),
          supabase
            .from("tb_prediksi_kualitas_udara")
            .select("pm2_5_ispu, pm10_ispu, no2_ispu, co_ispu, o3_ispu, created_at")
            .order("created_at", { ascending: false })
            .limit(1)
            .single()
        ]);

        if (gasResponse.error && gasResponse.error.code !== 'PGRST116') throw gasResponse.error;
        if (predResponse.error && predResponse.error.code !== 'PGRST116') throw predResponse.error;

        const dataGas = gasResponse.data;
        const dataPred = predResponse.data;

        const isStale = (timestamp: string) => {
          return Date.now() - new Date(timestamp).getTime() > 30 * 60 * 1000;
        };
        const gasTime = dataGas?.created_at;
        const predTime = dataPred?.created_at;
        const sensorOffline = Boolean(gasTime && isStale(gasTime));

        if (dataPred) {
          setRealtimeData({
            pm25: dataPred.pm2_5_ispu || 0,
            pm10: dataPred.pm10_ispu || 0,
            no2: dataPred.no2_ispu || 0,
            co: dataPred.co_ispu || 0,
            o3: dataPred.o3_ispu || 0,
            temperature: dataGas?.temperature || 0,
            humidity: dataGas?.humidity || 0,
            sensorOffline,
          });
        }
      } catch (err) {
        console.error("Error fetching realtime data:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    const interval = setInterval(fetchData, 300000); // 5 minutes (data is also updated via Realtime)

    // Subscribe to changes from the prediction table since that has the core ISPU
    const channel = supabase
      .channel("realtime-updates")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "tb_prediksi_kualitas_udara" },
        (payload) => {
          const newData = payload.new as any;
          if (!newData) return;

          setRealtimeData((prev) => ({
            ...prev,
            pm25: newData.pm2_5_ispu ?? prev?.pm25 ?? 0,
            pm10: newData.pm10_ispu ?? prev?.pm10 ?? 0,
            no2: newData.no2_ispu ?? prev?.no2 ?? 0,
            co: newData.co_ispu ?? prev?.co ?? 0,
            o3: newData.o3_ispu ?? prev?.o3 ?? 0,
            temperature: prev?.temperature || 0,
            humidity: prev?.humidity || 0,
            sensorOffline: false,
          }));
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(interval); // Cleanup interval on component unmount
  }, []);

  return (
    <div className="min-h-full">
      {/* Header */}
      <div className="border-b border-border bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="w-full px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-lg font-semibold text-foreground">Monitoring Kualitas Udara</h1>
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-flex flex-col items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700 ring-1 ring-emerald-600/20">
                {currentTime ? (
                  <>
                    <span>{currentTime.toLocaleDateString("id-ID", { year: "numeric", month: "long", day: "numeric" })}</span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                      {currentTime.toLocaleTimeString("id-ID", { timeZone: "Asia/Jakarta", hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                    </span>
                  </>
                ) : (
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    --
                  </span>
                )}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="w-full px-6 py-6">
        {loading ? (
          <div className="flex h-64 items-center justify-center">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          </div>
        ) : (
          <OverviewTab realtimeData={realtimeData} />
        )}
      </div>
    </div>
  );
}

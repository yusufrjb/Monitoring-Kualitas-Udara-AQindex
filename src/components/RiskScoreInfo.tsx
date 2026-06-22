"use client"

import { Info, X } from "lucide-react"
import * as PopoverPrimitive from "@radix-ui/react-popover"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"

const RISK_TABLE = [
  { range: "0% – 20%", label: "Sangat Aman", action: "Aktivitas luar ruangan normal." },
  { range: "21% – 40%", label: "Waspada Ringan", action: "Pantau kesehatan jika memiliki asma." },
  { range: "41% – 60%", label: "Perlu Perhatian", action: "Kurangi aktivitas fisik berat di luar." },
  { range: "61% – 80%", label: "Risiko Tinggi", action: "Gunakan masker, hindari paparan lama." },
  { range: "81% – 100%", label: "Bahaya (Darurat)", action: "Wajib masker, tetap di dalam ruangan." },
]

export default function RiskScoreInfo() {
  return (
    <Popover>
      <PopoverTrigger className="inline-flex items-center justify-center rounded hover:bg-slate-100 size-5 transition-colors cursor-pointer">
        <Info size={13} className="text-slate-400" />
      </PopoverTrigger>
      <PopoverContent side="top" align="center" sideOffset={4} className="w-auto p-0 overflow-hidden rounded-lg">
        <div className="relative">
          <div className="flex items-center justify-between px-3 py-1.5 bg-slate-50 border-b border-slate-200">
            <span className="text-[11px] font-semibold text-slate-600">Panduan Risk Score</span>
            <PopoverPrimitive.Close className="inline-flex items-center justify-center rounded hover:bg-slate-200 size-5 transition-colors cursor-pointer">
              <X size={12} className="text-slate-400" />
            </PopoverPrimitive.Close>
          </div>
          <div className="p-2">
            <table className="text-[10px] leading-tight">
              <thead>
                <tr className="border-b border-slate-100">
                  <th className="text-left font-semibold text-slate-500 pr-3 py-1">Rentang</th>
                  <th className="text-left font-semibold text-slate-500 pr-3 py-1">Interpretasi</th>
                  <th className="text-left font-semibold text-slate-500 py-1">Tindakan</th>
                </tr>
              </thead>
              <tbody>
                {RISK_TABLE.map((row, i) => (
                  <tr key={i} className="border-b border-slate-50 last:border-0">
                    <td className="pr-3 py-1 text-slate-600 whitespace-nowrap">{row.range}</td>
                    <td className="pr-3 py-1 font-medium text-slate-700 whitespace-nowrap">{row.label}</td>
                    <td className="py-1 text-slate-500 max-w-[180px]">{row.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}

import { NextResponse } from 'next/server';
import { execSync } from 'child_process';
import path from 'path';

const SCRIPT_PATH = path.resolve(process.cwd(), 'ml_model', 'predict_hourly_multi.py');

export async function GET() {
    try {
        execSync(`python "${SCRIPT_PATH}"`, {
            timeout: 30000,
            encoding: 'utf-8',
            stdio: ['pipe', 'pipe', 'pipe'],
            cwd: path.resolve(process.cwd()),
        });
        return NextResponse.json({ success: true });
    } catch {
        // prediction script gagal, bukan fatal — data lama tetap bisa ditampilkan
        return NextResponse.json({ success: false });
    }
}

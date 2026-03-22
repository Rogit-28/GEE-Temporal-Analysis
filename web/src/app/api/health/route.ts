import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json(
    {
      ok: true,
      app: "satchange-web",
      version: "1.0.0-alpha.1",
    },
    { status: 200 }
  );
}

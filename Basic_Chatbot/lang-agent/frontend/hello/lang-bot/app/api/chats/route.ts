// app/api/chats/route.ts

import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';
    
    const response = await fetch(`${FASTAPI_URL}/chats`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`FastAPI server error: ${response.status}`, errorText);
      throw new Error(`FastAPI server responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('Error fetching chats:', error);
    return NextResponse.json(
      { error: 'Failed to fetch chat history' },
      { status: 500 }
    );
  }
}

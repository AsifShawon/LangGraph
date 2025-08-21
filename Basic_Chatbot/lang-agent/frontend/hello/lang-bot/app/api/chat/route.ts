// app/api/chat/route.ts

import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { message, thread_id } = body;

    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { error: 'Message is required and must be a string' },
        { status: 400 }
      );
    }

    const FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';
    
    const backendRequestBody: { message: string; thread_id?: string } = {
      message,
    };

    if (thread_id) {
      backendRequestBody.thread_id = thread_id;
    }

    const response = await fetch(`${FASTAPI_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(backendRequestBody),
    });

    console.log('Response status:', response);

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`FastAPI server error: ${response.status}`, errorText);
      throw new Error(`FastAPI server responded with status: ${response.status}`);
    }

    // Correctly forward the streaming response
    return new Response(response.body, {
      headers: {
        'Content-Type': 'text/event-stream',
      },
    });

  } catch (error) {
    console.error('Error in Next.js API route:', error);
    
    return NextResponse.json(
      { error: 'Failed to connect to the chat service.' },
      { status: 500 }
    );
  }
}

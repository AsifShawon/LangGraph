// app/api/chats/[threadId]/route.ts

import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: { threadId: string } }
) {
  try {
    const threadId = params.threadId;
    
    if (!threadId) {
      return NextResponse.json(
        { error: 'Thread ID is required' },
        { status: 400 }
      );
    }

    const FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';
    
    const response = await fetch(`${FASTAPI_URL}/chats/${threadId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      if (response.status === 404) {
        return NextResponse.json(
          { error: 'Chat not found' },
          { status: 404 }
        );
      }
      const errorText = await response.text();
      console.error(`FastAPI server error: ${response.status}`, errorText);
      throw new Error(`FastAPI server responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('Error fetching chat history:', error);
    return NextResponse.json(
      { error: 'Failed to fetch chat history' },
      { status: 500 }
    );
  }
}

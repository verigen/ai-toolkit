/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getOpenAIApiKey, getOpenAIBaseUrl, getRefineModel, getRefineSystemPrompt } from '@/server/settings';

const contentTypeMap: { [key: string]: string } = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp',
  '.gif': 'image/gif',
};

function isUnderRoot(filepath: string, root: string): boolean {
  const resolved = path.resolve(filepath);
  return resolved === root || resolved.startsWith(root + path.sep);
}

export async function POST(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    // Client aborted the request before body was fully sent
    return new NextResponse(null, { status: 499 });
  }

  if (request.signal.aborted) {
    return new NextResponse(null, { status: 499 });
  }

  const { imgPath, currentCaption = '', userPrompt = '' } = body;

  if (!userPrompt || !(userPrompt as string).trim()) {
    return NextResponse.json({ error: 'Refinement instruction is empty' }, { status: 400 });
  }

  const root = await getDatasetsRoot();
  if (!imgPath || !isUnderRoot(imgPath, root)) {
    console.warn(`Access denied: ${imgPath} not in ${root}`);
    return NextResponse.json({ error: 'Access denied' }, { status: 403 });
  }

  const [baseUrl, model, systemPrompt, apiKey] = await Promise.all([
    getOpenAIBaseUrl(),
    getRefineModel(),
    getRefineSystemPrompt(),
    getOpenAIApiKey(),
  ]);

  let b64: string;
  try {
    const buf = await fs.promises.readFile(imgPath);
    b64 = buf.toString('base64');
  } catch (err: any) {
    if (err?.code === 'ENOENT') {
      return NextResponse.json({ error: 'Image not found' }, { status: 404 });
    }
    console.error('Error reading image for refine:', err);
    return NextResponse.json({ error: 'Failed to read image' }, { status: 500 });
  }

  const ext = path.extname(imgPath).toLowerCase();
  const mime = contentTypeMap[ext] || 'image/jpeg';

  const userText = `Current caption: ${currentCaption}\n\nYour task: ${userPrompt}`;
  const payload = {
    model,
    max_tokens: 1024,
    messages: [
      ...(systemPrompt.trim() ? [{ role: 'system', content: systemPrompt }] : []),
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: `data:${mime};base64,${b64}` } },
          { type: 'text', text: userText },
        ],
      },
    ],
  };

  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;

  try {
    const resp = await fetch(baseUrl.replace(/\/+$/, '') + '/chat/completions', {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: request.signal,
    });
    if (!resp.ok) {
      const detail = await resp.text().catch(() => '');
      return NextResponse.json(
        { error: `LLM server returned ${resp.status}`, detail: detail.slice(0, 500) },
        { status: 502 },
      );
    }
    const data = await resp.json();
    const refined = data?.choices?.[0]?.message?.content?.trim() ?? '';
    if (!refined) {
      return NextResponse.json({ error: 'Empty response from LLM' }, { status: 502 });
    }
    return NextResponse.json({ refined });
  } catch (e: any) {
    if (request.signal.aborted) return new NextResponse(null, { status: 499 });
    console.error('Error calling refine LLM:', e);
    return NextResponse.json({ error: `Request failed: ${e?.message ?? e}` }, { status: 502 });
  }
}

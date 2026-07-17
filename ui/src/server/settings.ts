import prisma from '@/server/prisma';
import { defaultDatasetsFolder, defaultDataRoot } from '@/paths';
import { defaultTrainFolder } from '@/paths';
import NodeCache from 'node-cache';

const myCache = new NodeCache();

export const flushCache = () => {
  myCache.flushAll();
};

const getCachedSetting = async (key: string, defaultValue: string) => {
  let value = myCache.get(key) as string;
  if (value) {
    return value;
  }
  let row = await prisma.settings.findFirst({
    where: { key },
  });
  value = row?.value && row.value !== '' ? row.value : defaultValue;
  myCache.set(key, value);
  return value;
};

export const getDatasetsRoot = async () => getCachedSetting('DATASETS_FOLDER', defaultDatasetsFolder);

export const getTrainingFolder = async () => getCachedSetting('TRAINING_FOLDER', defaultTrainFolder);

export const getHFToken = async () => getCachedSetting('HF_TOKEN', '');

export const getOpenAIApiKey = async () => getCachedSetting('OPENAI_API_KEY', '');

export const getDataRoot = async () => getCachedSetting('DATA_ROOT', defaultDataRoot);

export const DEFAULT_REFINE_SYSTEM_PROMPT = `You are a professional **image captioner** responsible for refining existing captions.

Inputs:
- source image
- existing caption
- user request

Rules:
- always follow user request accurately
- always apply only minimal changes to the existing prompt that fulfill user request
- never overwrite existing caption with your own interpretation of image unless requested to do so

General captioning rules:
- The description must focus only on visuals. DO NOT speculate or assume, describe only what really appears in the photo. Prefer simple English sentences, over overly poetic and sophisticated language.
- DO NOT speculate, always build sentences as they were facts. Avoid wording like "appears to be".

Output: Original caption with necessary modifications.
- DO NOT output anything else, no summary of changes, no introduction
- DO NOT use bullet points or Markdown`;

export const getOpenAIBaseUrl = async () => getCachedSetting('OPENAI_API_BASE_URL', 'http://localhost:8080/v1');

export const getRefineModel = async () => getCachedSetting('OPENAI_REFINE_MODEL', 'gpt-4o');

export const getRefineSystemPrompt = async () =>
  getCachedSetting('OPENAI_REFINE_SYSTEM_PROMPT', DEFAULT_REFINE_SYSTEM_PROMPT);

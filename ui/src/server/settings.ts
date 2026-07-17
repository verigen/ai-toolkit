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

export const DEFAULT_REFINE_SYSTEM_PROMPT = `You are helping refine image captions for an image-generation training dataset.
You are shown an image, its current caption, and an instruction describing what to
fix or improve. Rewrite the caption applying the instruction. Keep everything in the
current caption that is still accurate; only change what the instruction asks for.
Describe only what is visible. Output ONLY the final caption text — no preamble,
no explanations, no quotation marks.`;

export const getRefineBaseUrl = async () => getCachedSetting('OPENAI_REFINE_BASE_URL', 'http://localhost:8080/v1');

export const getRefineModel = async () => getCachedSetting('OPENAI_REFINE_MODEL', 'gpt-4o');

export const getRefineSystemPrompt = async () =>
  getCachedSetting('OPENAI_REFINE_SYSTEM_PROMPT', DEFAULT_REFINE_SYSTEM_PROMPT);

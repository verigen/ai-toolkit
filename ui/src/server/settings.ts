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

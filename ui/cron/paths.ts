import path from 'path';
import prisma from './prisma';

export const TOOLKIT_ROOT = path.resolve('@', '..', '..');
export const defaultTrainFolder = path.join(TOOLKIT_ROOT, 'output');
export const defaultDatasetsFolder = path.join(TOOLKIT_ROOT, 'datasets');
export const defaultDataRoot = path.join(TOOLKIT_ROOT, 'data');

// Forked file-server workers set AI_TOOLKIT_QUIET_PATHS so this line prints
// once per launched process group, not once per worker.
if (!process.env.AI_TOOLKIT_QUIET_PATHS) {
  console.log('TOOLKIT_ROOT:', TOOLKIT_ROOT);
}

const getSetting = async (key: string, defaultValue: string) => {
  let row = await prisma.settings.findFirst({
    where: { key },
  });
  return row?.value && row.value !== '' ? row.value : defaultValue;
};

export const getTrainingFolder = async () => getSetting('TRAINING_FOLDER', defaultTrainFolder);

export const getHFToken = async () => getSetting('HF_TOKEN', '');

export const getOpenAIApiKey = async () => getSetting('OPENAI_API_KEY', '');

export const getOpenAIBaseUrl = async () => getSetting('OPENAI_API_BASE_URL', 'http://localhost:8080/v1');

'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export interface Settings {
  HF_TOKEN: string;
  TRAINING_FOLDER: string;
  DATASETS_FOLDER: string;
  OPENAI_API_KEY: string;
  OPENAI_API_BASE_URL: string;
  OPENAI_REFINE_MODEL: string;
  OPENAI_REFINE_SYSTEM_PROMPT: string;
  MODELS_PATH: string;
}

export default function useSettings() {
  const [settings, setSettings] = useState({
    HF_TOKEN: '',
    TRAINING_FOLDER: '',
    DATASETS_FOLDER: '',
    OPENAI_API_KEY: '',
    OPENAI_API_BASE_URL: '',
    OPENAI_REFINE_MODEL: '',
    OPENAI_REFINE_SYSTEM_PROMPT: '',
    MODELS_PATH: '',
  });
  const [isSettingsLoaded, setIsLoaded] = useState(false);
  useEffect(() => {
    apiClient
      .get('/api/settings')
      .then(res => res.data)
      .then(data => {
        console.log('Settings:', data);
        setSettings({
          HF_TOKEN: data.HF_TOKEN || '',
          TRAINING_FOLDER: data.TRAINING_FOLDER || '',
          DATASETS_FOLDER: data.DATASETS_FOLDER || '',
          OPENAI_API_KEY: data.OPENAI_API_KEY || '',
          OPENAI_API_BASE_URL: data.OPENAI_API_BASE_URL || '',
          OPENAI_REFINE_MODEL: data.OPENAI_REFINE_MODEL || '',
          OPENAI_REFINE_SYSTEM_PROMPT: data.OPENAI_REFINE_SYSTEM_PROMPT || '',
          MODELS_PATH: data.MODELS_PATH || '',
        });
        setIsLoaded(true);
      })
      .catch(error => console.error('Error fetching settings:', error));
  }, []);

  return { settings, setSettings, isSettingsLoaded };
}

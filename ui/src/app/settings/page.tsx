'use client';

import { useEffect, useState } from 'react';
import useSettings from '@/hooks/useSettings';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';

export default function Settings() {
  const { settings, setSettings } = useSettings();
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('saving');

    apiClient
      .post('/api/settings', settings)
      .then(() => {
        setStatus('success');
      })
      .catch(error => {
        console.error('Error saving settings:', error);
        setStatus('error');
      })
      .finally(() => {
        setTimeout(() => setStatus('idle'), 2000);
      });
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setSettings(prev => ({ ...prev, [name]: value }));
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Settings</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <div className="space-y-4">
                <div>
                  <label htmlFor="HF_TOKEN" className="block text-sm font-medium mb-2">
                    Hugging Face Token
                    <div className="text-gray-500 text-sm ml-1">
                      Create a Read token on{' '}
                      <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                        {' '}
                        Huggingface
                      </a>{' '}
                      if you need to access gated/private models.
                    </div>
                  </label>
                  <input
                    type="password"
                    id="HF_TOKEN"
                    name="HF_TOKEN"
                    value={settings.HF_TOKEN}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter your Hugging Face token"
                  />
                </div>

                <div>
                  <label htmlFor="TRAINING_FOLDER" className="block text-sm font-medium mb-2">
                    Training Folder Path
                    <div className="text-gray-500 text-sm ml-1">
                      We will store your training information here. Must be an absolute path. If blank, it will default
                      to the output folder in the project root.
                    </div>
                  </label>
                  <input
                    type="text"
                    id="TRAINING_FOLDER"
                    name="TRAINING_FOLDER"
                    value={settings.TRAINING_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter training folder path"
                  />
                </div>

                <div>
                  <label htmlFor="DATASETS_FOLDER" className="block text-sm font-medium mb-2">
                    Dataset Folder Path
                    <div className="text-gray-500 text-sm ml-1">
                      Where we store and find your datasets.{' '}
                      <span className="text-orange-800">
                        Warning: This software may modify datasets so it is recommended you keep a backup somewhere else
                        or have a dedicated folder for this software.
                      </span>
                    </div>
                  </label>
                  <input
                    type="text"
                    id="DATASETS_FOLDER"
                    name="DATASETS_FOLDER"
                    value={settings.DATASETS_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter datasets folder path"
                  />
                </div>

                <div className="pt-2 mt-2 border-t border-gray-800">
                  <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-4">
                    OpenAI-Compatible API
                  </h2>
                  <div className="space-y-4">
                    <div>
                      <label htmlFor="OPENAI_API_KEY" className="block text-sm font-medium mb-2">
                        OpenAI-Compatible API Key
                        <div className="text-gray-500 text-sm ml-1">
                          Used to authenticate to the API base URL below. Leave empty for local servers (e.g.
                          llama.cpp).
                        </div>
                      </label>
                      <input
                        type="password"
                        id="OPENAI_API_KEY"
                        name="OPENAI_API_KEY"
                        value={settings.OPENAI_API_KEY}
                        onChange={handleChange}
                        className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                        placeholder="Enter your OpenAI-compatible API key"
                      />
                    </div>

                    <div>
                      <label htmlFor="OPENAI_API_BASE_URL" className="block text-sm font-medium mb-2">
                        OpenAI-Compatible API Base URL
                        <div className="text-gray-500 text-sm ml-1">
                          Shared by the OpenAI-compatible batch captioner and the &quot;Refine with LLM&quot; button in
                          the dataset image viewer (e.g. a local llama.cpp server).
                        </div>
                      </label>
                      <input
                        type="text"
                        id="OPENAI_API_BASE_URL"
                        name="OPENAI_API_BASE_URL"
                        value={settings.OPENAI_API_BASE_URL}
                        onChange={handleChange}
                        className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                        placeholder="http://localhost:8080/v1"
                      />
                    </div>

                    <div>
                      <label htmlFor="OPENAI_REFINE_MODEL" className="block text-sm font-medium mb-2">
                        Refinement Model Name
                        <div className="text-gray-500 text-sm ml-1">
                          Model name to send to the API base URL above when using &quot;Refine with LLM&quot;.
                        </div>
                      </label>
                      <input
                        type="text"
                        id="OPENAI_REFINE_MODEL"
                        name="OPENAI_REFINE_MODEL"
                        value={settings.OPENAI_REFINE_MODEL}
                        onChange={handleChange}
                        className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                        placeholder="gpt-4o"
                      />
                    </div>

                    <div>
                      <label htmlFor="OPENAI_REFINE_SYSTEM_PROMPT" className="block text-sm font-medium mb-2">
                        Refinement System Prompt
                        <div className="text-gray-500 text-sm ml-1">
                          System prompt sent along with the image, current caption, and your instruction when
                          refining a caption. Leave blank to use the built-in default.
                        </div>
                      </label>
                      <textarea
                        id="OPENAI_REFINE_SYSTEM_PROMPT"
                        name="OPENAI_REFINE_SYSTEM_PROMPT"
                        value={settings.OPENAI_REFINE_SYSTEM_PROMPT}
                        onChange={handleChange}
                        rows={5}
                        className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                        placeholder="You are helping refine image captions..."
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={status === 'saving'}
            className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {status === 'saving' ? 'Saving...' : 'Save Settings'}
          </button>

          {status === 'success' && <p className="text-green-500 text-center">Settings saved successfully!</p>}
          {status === 'error' && <p className="text-red-500 text-center">Error saving settings. Please try again.</p>}
        </form>
      </MainContent>
    </>
  );
}

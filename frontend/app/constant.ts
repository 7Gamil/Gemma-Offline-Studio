import { prebuiltAppConfig } from "@mlc-ai/web-llm";
import { ModelRecord } from "./client/api";
import { getQuantization, getSize } from "./utils";

export const OWNER = "7Gamil";
export const REPO = "Gemma-Offline-Studio";
export const REPO_URL = `https://github.com/${OWNER}/${REPO}`;
export const ISSUE_URL = `https://github.com/${OWNER}/${REPO}/issues`;

export enum Path {
  Home = "/",
  Chat = "/chat",
  Settings = "/settings",
  Templates = "/templates",
}

export enum ApiPath {
  Cors = "",
}

export enum SlotID {
  AppBody = "app-body",
  CustomModel = "custom-model",
}

export enum FileName {
  Templates = "templates.json",
  Prompts = "prompts.json",
}

export enum StoreKey {
  Chat = "chat-next-web-store",
  Access = "access-control",
  Config = "app-config",
  Templates = "templates-store",
  Prompt = "prompt-store",
  Update = "chat-update",
  Sync = "sync",
}

export const DEFAULT_SIDEBAR_WIDTH = 320;
export const MAX_SIDEBAR_WIDTH = 500;
export const MIN_SIDEBAR_WIDTH = 260;
export const NARROW_SIDEBAR_WIDTH = 100;

export const ACCESS_CODE_PREFIX = "nk-";

export const LAST_INPUT_KEY = "last-input";
export const UNFINISHED_INPUT = (name: string) => "unfinished-input-" + name;

export const STORAGE_KEY = "chatgpt-next-web";

export const REQUEST_TIMEOUT_MS = 60000;

export const EXPORT_MESSAGE_CLASS_NAME = "export-markdown";

export const DEFAULT_INPUT_TEMPLATE = `{{input}}`; // input / time / model / lang

export const DEFAULT_SYSTEM_TEMPLATE = `
You are an AI large language model assistant trained by {{provider}}.
You are currently engaging with users on Gemma Offline Studio, an open-source AI Chatbot.
Model display_name:  {{model}}
The current date and time is {{time}}.
Latex inline format: \\(x^2\\) 
Latex block format: $$e=mc^2$$
`;

export enum ModelFamily {
  LLAMA = "llama",
  PHI = "phi",
  MISTRAL = "mistral",
  GEMMA = "gemma",
  QWEN = "qwen",
  SMOL_LM = "smollm",
  WIZARD_MATH = "wizardmath",
  STABLE_LM = "stablelm",
  REDPAJAMA = "redpajama",
  DEEPSEEK = "DeepSeek",
}

const qwen3_common_configs = {
  display_name: "Qwen",
  provider: "Alibaba",
  family: ModelFamily.QWEN,
  // Recommended config is for non-thinking mode
  // For thinking mode, see webllm.ts where temperature=0.6 and top_p=0.95 are applied
  recommended_config: {
    temperature: 0.7,
    presence_penalty: 0,
    frequency_penalty: 0,
    top_p: 0.8,
  },
};

const DEFAULT_MODEL_BASES: ModelRecord[] = [
  // Gemma 2
  {
    name: "gemma-2-2b-it-q4f16_1-MLC",
    display_name: "Gemma",
    provider: "Google",
    family: ModelFamily.GEMMA,
    recommended_config: {
      temperature: 0.7,
      presence_penalty: 0,
      frequency_penalty: 1,
      top_p: 0.95,
    },
  },
  {
    name: "gemma-2-2b-it-q4f32_1-MLC",
    display_name: "Gemma",
    provider: "Google",
    family: ModelFamily.GEMMA,
    recommended_config: {
      temperature: 0.7,
      presence_penalty: 0,
      frequency_penalty: 1,
      top_p: 0.95,
    },
  },
  {
    name: "gemma-2-2b-it-q4f16_1-MLC-1k",
    display_name: "Gemma",
    provider: "Google",
    family: ModelFamily.GEMMA,
    recommended_config: {
      temperature: 0.7,
      presence_penalty: 0,
      frequency_penalty: 1,
      top_p: 0.95,
    },
  },
  {
    name: "gemma-2-2b-it-q4f32_1-MLC-1k",
    display_name: "Gemma",
    provider: "Google",
    family: ModelFamily.GEMMA,
    recommended_config: {
      temperature: 0.7,
      presence_penalty: 0,
      frequency_penalty: 1,
      top_p: 0.95,
    },
  },
  {
    name: "gemma-2-9b-it-q4f16_1-MLC",
    display_name: "Gemma",
    provider: "Google",
    family: ModelFamily.GEMMA,
    recommended_config: {
      temperature: 0.7,
      presence_penalty: 0,
      frequency_penalty: 1,
      top_p: 0.95,
    },
  },
  {
    name: "gemma-2-9b-it-q4f32_1-MLC",
    display_name: "Gemma",
    provider: "Google",
    family: ModelFamily.GEMMA,
    recommended_config: {
      temperature: 0.7,
      presence_penalty: 0,
      frequency_penalty: 1,
      top_p: 0.95,
    },
  },
  {
    name: "gemma-2-2b-jpn-it-q4f16_1-MLC",
    display_name: "Gemma",
    provider: "Google",
    family: ModelFamily.GEMMA,
    recommended_config: {
      temperature: 0.7,
      presence_penalty: 0,
      frequency_penalty: 1,
      top_p: 0.9,
    },
  },
  {
    name: "gemma-2-2b-jpn-it-q4f32_1-MLC",
    display_name: "Gemma",
    provider: "Google",
    family: ModelFamily.GEMMA,
    recommended_config: {
      temperature: 0.7,
      presence_penalty: 0,
      frequency_penalty: 1,
      top_p: 0.9,
    },
  },
];

export const DEFAULT_MODELS: ModelRecord[] = [
  {
    name: "gemma-3n",
    display_name: "Gemma 3n",
    provider: "Google",
    family: ModelFamily.GEMMA,
    recommended_config: {
      temperature: 0.7,
      top_p: 0.95,
    },
  },
];

export const CHAT_PAGE_SIZE = 15;
export const MAX_RENDER_MSG_COUNT = 45;

export const LOG_LEVELS = {
  TRACE: 0,
  DEBUG: 1,
  INFO: 2,
  WARN: 3,
  ERROR: 4,
  SILENT: 5,
};

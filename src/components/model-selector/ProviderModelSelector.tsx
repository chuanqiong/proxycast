/**
 * @file ProviderModelSelector 组件
 * @description 双栏模型选择器：左侧 Provider 列表，右侧模型列表
 * @module components/model-selector/ProviderModelSelector
 */

import React, { useState, useMemo, useCallback, useEffect } from "react";
import { cn } from "@/lib/utils";
import { useModelRegistry } from "@/hooks/useModelRegistry";
import { useProviderPool } from "@/hooks/useProviderPool";
import { useApiKeyProvider } from "@/hooks/useApiKeyProvider";
import { getProviderAliasConfig } from "@/lib/api/modelRegistry";
import {
  Check,
  ChevronRight,
  Eye,
  Wrench,
  Brain,
  Loader2,
  AlertCircle,
} from "lucide-react";
import type {
  EnhancedModelMetadata,
  ProviderAliasConfig,
} from "@/lib/types/modelRegistry";

// ============================================================================
// 类型定义
// ============================================================================

export interface ProviderModelSelectorProps {
  /** 选择模型回调 */
  onSelect?: (model: EnhancedModelMetadata, providerId: string) => void;
  /** 初始选中的 Provider */
  initialProviderId?: string;
  /** 初始选中的模型 */
  initialModelId?: string;
  /** 自定义类名 */
  className?: string;
}

/** 已配置的 Provider 信息 */
interface ConfiguredProvider {
  id: string;
  name: string;
  registryId: string;
  source: "oauth" | "apikey";
  credentialCount: number;
}

// ============================================================================
// 常量
// ============================================================================

/** OAuth 凭证类型到 Provider ID 的映射 */
const CREDENTIAL_TYPE_TO_PROVIDER_ID: Record<string, string> = {
  kiro: "kiro",
  gemini: "gemini_oauth",
  qwen: "alibaba",
  antigravity: "antigravity",
  codex: "openai",
  claude_oauth: "anthropic",
  iflow: "iflow",
  openai: "openai",
  claude: "anthropic",
  gemini_api_key: "gemini_api_key",
};

/** API Key Provider 类型到 Registry ID 的映射 */
const PROVIDER_TYPE_TO_REGISTRY_ID: Record<string, string> = {
  anthropic: "anthropic",
  openai: "openai",
  "openai-response": "openai",
  gemini: "google",
  "azure-openai": "openai",
  vertexai: "google",
  "aws-bedrock": "anthropic",
  ollama: "ollama",
  "new-api": "custom",
  gateway: "custom",
};

/** Provider 显示名称 */
const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  anthropic: "Anthropic",
  openai: "OpenAI",
  google: "Google",
  gemini_oauth: "Gemini OAuth",
  gemini_api_key: "Gemini API Key",
  alibaba: "阿里云",
  ollama: "Ollama",
  custom: "自定义",
  antigravity: "Antigravity",
  kiro: "Kiro",
  iflow: "iFlow",
};

/** 别名 Provider 列表（使用别名配置而非标准模型注册表） */
const ALIAS_PROVIDERS = ["antigravity", "kiro"];

/** Provider ID 到模型注册表 Provider ID 的映射（用于过滤模型） */
const PROVIDER_TO_REGISTRY_MAPPING: Record<string, string> = {
  gemini_oauth: "google",
  gemini_api_key: "google",
};

// ============================================================================
// 子组件
// ============================================================================

interface ProviderItemProps {
  provider: ConfiguredProvider;
  isSelected: boolean;
  onClick: () => void;
}

/** Provider 列表项 */
const ProviderItem: React.FC<ProviderItemProps> = ({
  provider,
  isSelected,
  onClick,
}) => {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "w-full flex items-center justify-between px-3 py-2 text-sm rounded-md transition-colors",
        isSelected
          ? "bg-primary text-primary-foreground"
          : "hover:bg-muted text-foreground",
      )}
      data-testid={`provider-item-${provider.id}`}
    >
      <div className="flex items-center gap-2 min-w-0">
        <ChevronRight
          className={cn(
            "h-4 w-4 flex-shrink-0 transition-transform",
            isSelected && "rotate-90",
          )}
        />
        <span className="truncate">{provider.name}</span>
      </div>
      <span
        className={cn(
          "text-xs px-1.5 py-0.5 rounded",
          isSelected
            ? "bg-primary-foreground/20 text-primary-foreground"
            : "bg-muted text-muted-foreground",
        )}
      >
        {provider.credentialCount}
      </span>
    </button>
  );
};

interface ModelItemProps {
  model: EnhancedModelMetadata;
  isSelected: boolean;
  onClick: () => void;
}

/** 模型列表项 */
const ModelItem: React.FC<ModelItemProps> = ({
  model,
  isSelected,
  onClick,
}) => {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "w-full flex items-center justify-between px-3 py-2 text-sm rounded-md transition-colors",
        isSelected
          ? "bg-primary/10 border border-primary"
          : "hover:bg-muted border border-transparent",
      )}
      data-testid={`model-item-${model.id}`}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium truncate">{model.display_name}</span>
          {model.is_latest && (
            <span className="text-[10px] bg-green-100 text-green-700 px-1 py-0.5 rounded">
              最新
            </span>
          )}
        </div>
        <div className="text-xs text-muted-foreground truncate">{model.id}</div>
      </div>

      {/* 能力标签 */}
      <div className="flex items-center gap-1.5 ml-2">
        {model.capabilities.vision && (
          <span title="支持视觉">
            <Eye className="h-3.5 w-3.5 text-blue-500" />
          </span>
        )}
        {model.capabilities.tools && (
          <span title="支持工具">
            <Wrench className="h-3.5 w-3.5 text-orange-500" />
          </span>
        )}
        {model.capabilities.reasoning && (
          <span title="支持推理">
            <Brain className="h-3.5 w-3.5 text-purple-500" />
          </span>
        )}
        {isSelected && <Check className="h-4 w-4 text-primary ml-1" />}
      </div>
    </button>
  );
};

// ============================================================================
// 主组件
// ============================================================================

/**
 * 双栏模型选择器组件
 *
 * 左侧显示已配置凭证的 Provider 列表（单选）
 * 右侧显示选中 Provider 对应的模型列表（单选）
 *
 * @example
 * ```tsx
 * <ProviderModelSelector
 *   onSelect={(model, providerId) => {
 *     console.log("选中模型:", model.display_name);
 *   }}
 * />
 * ```
 */
export const ProviderModelSelector: React.FC<ProviderModelSelectorProps> = ({
  onSelect,
  initialProviderId,
  initialModelId,
  className,
}) => {
  // 状态
  const [selectedProviderId, setSelectedProviderId] = useState<string | null>(
    initialProviderId || null,
  );
  const [selectedModelId, setSelectedModelId] = useState<string | null>(
    initialModelId || null,
  );
  const [aliasConfig, setAliasConfig] = useState<ProviderAliasConfig | null>(
    null,
  );
  const [aliasLoading, setAliasLoading] = useState(false);

  // 获取凭证池数据
  const { overview: oauthCredentials, loading: oauthLoading } =
    useProviderPool();
  const { providers: apiKeyProviders, loading: apiKeyLoading } =
    useApiKeyProvider();

  // 获取模型注册表数据
  const {
    models,
    loading: modelsLoading,
    error: modelsError,
  } = useModelRegistry({
    autoLoad: true,
  });

  // 当选中别名 Provider 时，加载别名配置
  useEffect(() => {
    if (selectedProviderId && ALIAS_PROVIDERS.includes(selectedProviderId)) {
      setAliasLoading(true);
      getProviderAliasConfig(selectedProviderId)
        .then((config) => {
          setAliasConfig(config);
        })
        .catch((error) => {
          console.error("加载别名配置失败:", error);
          setAliasConfig(null);
        })
        .finally(() => {
          setAliasLoading(false);
        });
    } else {
      setAliasConfig(null);
    }
  }, [selectedProviderId]);

  // 计算已配置的 Provider 列表
  const configuredProviders = useMemo(() => {
    const providerMap = new Map<string, ConfiguredProvider>();

    // 从 OAuth 凭证提取 Provider
    oauthCredentials.forEach((overview) => {
      const registryId = CREDENTIAL_TYPE_TO_PROVIDER_ID[overview.provider_type];
      if (registryId && overview.credentials.length > 0) {
        const existing = providerMap.get(registryId);
        if (existing) {
          existing.credentialCount += overview.credentials.length;
        } else {
          providerMap.set(registryId, {
            id: registryId,
            name: PROVIDER_DISPLAY_NAMES[registryId] || registryId,
            registryId,
            source: "oauth",
            credentialCount: overview.credentials.length,
          });
        }
      }
    });

    // 从 API Key Provider 提取（只包含有 API Key 的）
    apiKeyProviders
      .filter((p) => p.api_key_count > 0 && p.enabled)
      .forEach((provider) => {
        const registryId =
          PROVIDER_TYPE_TO_REGISTRY_ID[provider.type] || provider.type;
        const existing = providerMap.get(registryId);
        if (existing) {
          existing.credentialCount += provider.api_key_count;
        } else {
          providerMap.set(registryId, {
            id: registryId,
            name: PROVIDER_DISPLAY_NAMES[registryId] || provider.name,
            registryId,
            source: "apikey",
            credentialCount: provider.api_key_count,
          });
        }
      });

    return Array.from(providerMap.values()).sort((a, b) =>
      a.name.localeCompare(b.name),
    );
  }, [oauthCredentials, apiKeyProviders]);

  // 默认选中第一个 Provider
  useEffect(() => {
    if (!selectedProviderId && configuredProviders.length > 0) {
      setSelectedProviderId(configuredProviders[0].registryId);
    }
  }, [selectedProviderId, configuredProviders]);

  // 过滤当前 Provider 的模型
  const filteredModels = useMemo((): EnhancedModelMetadata[] => {
    if (!selectedProviderId) return [];

    // 对于别名 Provider（Antigravity、Kiro），使用别名配置中的模型列表
    if (ALIAS_PROVIDERS.includes(selectedProviderId) && aliasConfig) {
      // 将别名配置中的模型转换为 EnhancedModelMetadata 格式
      return aliasConfig.models.map((modelName): EnhancedModelMetadata => {
        const aliasInfo = aliasConfig.aliases[modelName];
        return {
          id: modelName,
          display_name: modelName,
          provider_id: selectedProviderId,
          provider_name:
            PROVIDER_DISPLAY_NAMES[selectedProviderId] || selectedProviderId,
          family: aliasInfo?.provider || null,
          tier: "pro" as const,
          capabilities: {
            vision: false,
            tools: true,
            streaming: true,
            json_mode: true,
            function_calling: true,
            reasoning: modelName.includes("thinking"),
          },
          pricing: null,
          limits: {
            context_length: null,
            max_output_tokens: null,
            requests_per_minute: null,
            tokens_per_minute: null,
          },
          status: "active" as const,
          release_date: null,
          is_latest: false,
          description:
            aliasInfo?.description || `${aliasInfo?.actual || modelName}`,
          source: "custom" as const,
          created_at: Date.now() / 1000,
          updated_at: Date.now() / 1000,
        };
      });
    }

    // 对于标准 Provider，从模型注册表过滤
    // 使用映射表将 UI Provider ID 转换为模型注册表 Provider ID
    const registryProviderId =
      PROVIDER_TO_REGISTRY_MAPPING[selectedProviderId] || selectedProviderId;
    return models.filter((m) => m.provider_id === registryProviderId);
  }, [models, selectedProviderId, aliasConfig]);

  // 选择 Provider
  const handleSelectProvider = useCallback((providerId: string) => {
    setSelectedProviderId(providerId);
    setSelectedModelId(null); // 切换 Provider 时清除模型选择
  }, []);

  // 选择模型
  const handleSelectModel = useCallback(
    (model: EnhancedModelMetadata) => {
      setSelectedModelId(model.id);
      if (selectedProviderId) {
        onSelect?.(model, selectedProviderId);
      }
    },
    [selectedProviderId, onSelect],
  );

  const isLoading = oauthLoading || apiKeyLoading || modelsLoading;

  // 空状态
  if (!isLoading && configuredProviders.length === 0) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center py-12 text-muted-foreground",
          className,
        )}
        data-testid="provider-model-selector-empty"
      >
        <AlertCircle className="h-12 w-12 mb-4 opacity-50" />
        <p className="text-sm">暂无已配置的 Provider</p>
        <p className="text-xs mt-1">请先在凭证池中添加凭证</p>
      </div>
    );
  }

  return (
    <div
      className={cn("flex border rounded-lg overflow-hidden", className)}
      data-testid="provider-model-selector"
    >
      {/* 左侧：Provider 列表 */}
      <div className="w-48 border-r bg-muted/30 flex flex-col">
        <div className="px-3 py-2 border-b bg-muted/50">
          <h4 className="text-sm font-medium">Providers</h4>
          <p className="text-xs text-muted-foreground">已配置凭证的</p>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : (
            configuredProviders.map((provider) => (
              <ProviderItem
                key={provider.id}
                provider={provider}
                isSelected={selectedProviderId === provider.registryId}
                onClick={() => handleSelectProvider(provider.registryId)}
              />
            ))
          )}
        </div>
      </div>

      {/* 右侧：模型列表 */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="px-3 py-2 border-b bg-muted/50">
          <h4 className="text-sm font-medium">Models</h4>
          <p className="text-xs text-muted-foreground">
            {selectedProviderId
              ? `${PROVIDER_DISPLAY_NAMES[selectedProviderId] || selectedProviderId} 的模型`
              : "请选择 Provider"}
          </p>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {modelsLoading || aliasLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : modelsError ? (
            <div className="flex flex-col items-center justify-center py-8 text-red-500">
              <AlertCircle className="h-8 w-8 mb-2" />
              <p className="text-sm">{modelsError}</p>
            </div>
          ) : filteredModels.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
              <p className="text-sm">暂无模型数据</p>
            </div>
          ) : (
            filteredModels.map((model) => (
              <ModelItem
                key={model.id}
                model={model}
                isSelected={selectedModelId === model.id}
                onClick={() => handleSelectModel(model)}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default ProviderModelSelector;

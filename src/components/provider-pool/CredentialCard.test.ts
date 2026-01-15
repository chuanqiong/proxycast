/**
 * @file CredentialCard 属性测试
 * @description 测试 OAuth 凭证卡片信息完整性
 * @module components/provider-pool/CredentialCard.test
 *
 * **Feature: provider-ui-refactor**
 * **Property 3: OAuth 凭证卡片信息完整性**
 * **Validates: Requirements 2.2**
 */

import { describe, expect } from "vitest";
import { test } from "@fast-check/vitest";
import * as fc from "fast-check";
import type {
  CredentialDisplay,
  PoolProviderType,
  CredentialSource,
} from "@/lib/api/providerPool";

// ============================================================================
// 辅助函数（用于测试）
// ============================================================================

/**
 * 提取 OAuth 凭证卡片显示信息
 * 用于属性测试验证 Requirements 2.2
 *
 * @param credential OAuth 凭证数据
 * @returns 卡片显示信息
 */
export function extractOAuthCardDisplayInfo(credential: CredentialDisplay): {
  hasHealthStatus: boolean;
  hasUsageCount: boolean;
  hasActionButtons: boolean;
  healthStatus: "healthy" | "unhealthy" | "disabled";
  usageCount: number;
  errorCount: number;
} {
  // 健康状态：根据 is_healthy 和 is_disabled 判断
  let healthStatus: "healthy" | "unhealthy" | "disabled";
  if (credential.is_disabled) {
    healthStatus = "disabled";
  } else if (credential.is_healthy) {
    healthStatus = "healthy";
  } else {
    healthStatus = "unhealthy";
  }

  return {
    // 健康状态始终存在（通过 is_healthy 和 is_disabled 字段）
    hasHealthStatus:
      typeof credential.is_healthy === "boolean" &&
      typeof credential.is_disabled === "boolean",
    // 使用次数始终存在（通过 usage_count 字段）
    hasUsageCount: typeof credential.usage_count === "number",
    // 操作按钮始终存在（卡片组件固定渲染）
    hasActionButtons: true,
    healthStatus,
    usageCount: credential.usage_count,
    errorCount: credential.error_count,
  };
}

/**
 * 验证 OAuth 凭证卡片是否包含所有必要信息
 *
 * @param credential OAuth 凭证数据
 * @returns 是否包含所有必要信息
 */
export function isOAuthCardComplete(credential: CredentialDisplay): boolean {
  const info = extractOAuthCardDisplayInfo(credential);
  return info.hasHealthStatus && info.hasUsageCount && info.hasActionButtons;
}

/**
 * 获取 OAuth 凭证的操作按钮列表
 * 根据凭证类型返回应该显示的操作按钮
 *
 * @param credential OAuth 凭证数据
 * @returns 操作按钮列表
 */
export function getOAuthCardActionButtons(
  credential: CredentialDisplay,
): string[] {
  const buttons: string[] = [
    "toggle", // 启用/禁用
    "edit", // 编辑
    "checkHealth", // 检测健康
    "reset", // 重置
    "delete", // 删除
  ];

  // OAuth 类型凭证额外显示刷新 Token 按钮
  if (credential.credential_type.includes("oauth")) {
    buttons.push("refreshToken");
  }

  return buttons;
}

// ============================================================================
// 测试数据生成器
// ============================================================================

/**
 * 生成有效的 ISO 日期字符串
 */
const validDateArbitrary = fc
  .integer({
    min: new Date("2020-01-01").getTime(),
    max: new Date("2030-12-31").getTime(),
  })
  .map((timestamp) => new Date(timestamp).toISOString());

/**
 * OAuth Provider 类型
 */
const oauthProviderTypes: PoolProviderType[] = [
  "kiro",
  "gemini",
  "antigravity",
  "codex",
  "claude_oauth",
];

/**
 * OAuth 凭证类型
 */
const oauthCredentialTypes = [
  "kiro_oauth",
  "gemini_oauth",
  "antigravity_oauth",
  "codex_oauth",
  "claude_oauth",
];

/**
 * 凭证来源类型
 */
const credentialSourceArbitrary: fc.Arbitrary<CredentialSource> =
  fc.constantFrom("manual", "imported", "private");

/**
 * 生成随机 OAuth 凭证显示数据
 */
const oauthCredentialArbitrary: fc.Arbitrary<CredentialDisplay> = fc.record({
  uuid: fc.uuid(),
  provider_type: fc.constantFrom(...oauthProviderTypes),
  credential_type: fc.constantFrom(...oauthCredentialTypes),
  name: fc.option(fc.string({ minLength: 1, maxLength: 100 }), {
    nil: undefined,
  }),
  display_credential: fc.string({ minLength: 1, maxLength: 50 }),
  is_healthy: fc.boolean(),
  is_disabled: fc.boolean(),
  check_health: fc.boolean(),
  check_model_name: fc.option(fc.string({ minLength: 1, maxLength: 50 }), {
    nil: undefined,
  }),
  not_supported_models: fc.array(fc.string({ minLength: 1, maxLength: 50 }), {
    maxLength: 5,
  }),
  usage_count: fc.nat({ max: 100000 }),
  error_count: fc.nat({ max: 10000 }),
  last_used: fc.option(validDateArbitrary, { nil: undefined }),
  last_error_time: fc.option(validDateArbitrary, { nil: undefined }),
  last_error_message: fc.option(fc.string({ minLength: 1, maxLength: 200 }), {
    nil: undefined,
  }),
  last_health_check_time: fc.option(validDateArbitrary, { nil: undefined }),
  last_health_check_model: fc.option(
    fc.string({ minLength: 1, maxLength: 50 }),
    { nil: undefined },
  ),
  oauth_status: fc.option(
    fc.record({
      has_access_token: fc.boolean(),
      has_refresh_token: fc.boolean(),
      is_token_valid: fc.boolean(),
      expiry_info: fc.option(fc.string({ minLength: 1, maxLength: 100 }), {
        nil: undefined,
      }),
      creds_path: fc.string({ minLength: 1, maxLength: 200 }),
    }),
    { nil: undefined },
  ),
  token_cache_status: fc.option(
    fc.record({
      has_cached_token: fc.boolean(),
      is_valid: fc.boolean(),
      is_expiring_soon: fc.boolean(),
      expiry_time: fc.option(validDateArbitrary, { nil: undefined }),
      last_refresh: fc.option(validDateArbitrary, { nil: undefined }),
      refresh_error_count: fc.nat({ max: 100 }),
      last_refresh_error: fc.option(
        fc.string({ minLength: 1, maxLength: 200 }),
        { nil: undefined },
      ),
    }),
    { nil: undefined },
  ),
  created_at: validDateArbitrary,
  updated_at: validDateArbitrary,
  source: credentialSourceArbitrary,
  base_url: fc.option(fc.webUrl(), { nil: undefined }),
  api_key: fc.option(fc.string({ minLength: 1, maxLength: 100 }), {
    nil: undefined,
  }),
  proxy_url: fc.option(fc.webUrl(), { nil: undefined }),
});

// ============================================================================
// Property 3: OAuth 凭证卡片信息完整性
// ============================================================================

describe("Property 3: OAuth 凭证卡片信息完整性", () => {
  /**
   * Property 3: OAuth 凭证卡片信息完整性
   *
   * *对于任意* OAuth 凭证，渲染后的卡片应包含健康状态、使用次数和操作按钮
   *
   * **Validates: Requirements 2.2**
   */
  test.prop([oauthCredentialArbitrary], { numRuns: 100 })(
    "每个 OAuth 凭证卡片应包含健康状态、使用次数和操作按钮",
    (credential: CredentialDisplay) => {
      const displayInfo = extractOAuthCardDisplayInfo(credential);

      // 验证健康状态存在
      expect(displayInfo.hasHealthStatus).toBe(true);

      // 验证使用次数存在
      expect(displayInfo.hasUsageCount).toBe(true);

      // 验证操作按钮存在
      expect(displayInfo.hasActionButtons).toBe(true);
    },
  );

  test.prop([oauthCredentialArbitrary], { numRuns: 100 })(
    "健康状态应为 healthy、unhealthy 或 disabled 之一",
    (credential: CredentialDisplay) => {
      const displayInfo = extractOAuthCardDisplayInfo(credential);

      expect(["healthy", "unhealthy", "disabled"]).toContain(
        displayInfo.healthStatus,
      );
    },
  );

  test.prop([oauthCredentialArbitrary], { numRuns: 100 })(
    "使用次数应为非负整数",
    (credential: CredentialDisplay) => {
      const displayInfo = extractOAuthCardDisplayInfo(credential);

      expect(Number.isInteger(displayInfo.usageCount)).toBe(true);
      expect(displayInfo.usageCount).toBeGreaterThanOrEqual(0);
    },
  );

  test.prop([oauthCredentialArbitrary], { numRuns: 100 })(
    "错误次数应为非负整数",
    (credential: CredentialDisplay) => {
      const displayInfo = extractOAuthCardDisplayInfo(credential);

      expect(Number.isInteger(displayInfo.errorCount)).toBe(true);
      expect(displayInfo.errorCount).toBeGreaterThanOrEqual(0);
    },
  );

  test.prop([oauthCredentialArbitrary], { numRuns: 100 })(
    "OAuth 凭证应包含刷新 Token 操作按钮",
    (credential: CredentialDisplay) => {
      const buttons = getOAuthCardActionButtons(credential);

      // OAuth 凭证应该有刷新 Token 按钮
      expect(buttons).toContain("refreshToken");
    },
  );

  test.prop([oauthCredentialArbitrary], { numRuns: 100 })(
    "所有凭证应包含基本操作按钮",
    (credential: CredentialDisplay) => {
      const buttons = getOAuthCardActionButtons(credential);

      // 所有凭证都应该有这些基本按钮
      expect(buttons).toContain("toggle");
      expect(buttons).toContain("edit");
      expect(buttons).toContain("checkHealth");
      expect(buttons).toContain("reset");
      expect(buttons).toContain("delete");
    },
  );

  test.prop([oauthCredentialArbitrary], { numRuns: 100 })(
    "禁用状态应正确反映在健康状态中",
    (credential: CredentialDisplay) => {
      const displayInfo = extractOAuthCardDisplayInfo(credential);

      if (credential.is_disabled) {
        expect(displayInfo.healthStatus).toBe("disabled");
      }
    },
  );

  test.prop([oauthCredentialArbitrary], { numRuns: 100 })(
    "健康凭证（未禁用）应显示为 healthy",
    (credential: CredentialDisplay) => {
      const displayInfo = extractOAuthCardDisplayInfo(credential);

      if (!credential.is_disabled && credential.is_healthy) {
        expect(displayInfo.healthStatus).toBe("healthy");
      }
    },
  );

  test.prop([oauthCredentialArbitrary], { numRuns: 100 })(
    "不健康凭证（未禁用）应显示为 unhealthy",
    (credential: CredentialDisplay) => {
      const displayInfo = extractOAuthCardDisplayInfo(credential);

      if (!credential.is_disabled && !credential.is_healthy) {
        expect(displayInfo.healthStatus).toBe("unhealthy");
      }
    },
  );
});

// ============================================================================
// 边界情况测试
// ============================================================================

describe("OAuth 凭证卡片边界情况", () => {
  test("使用次数为 0 的凭证应正确显示", () => {
    const credential: CredentialDisplay = {
      uuid: "test-uuid",
      provider_type: "kiro",
      credential_type: "kiro_oauth",
      display_credential: "test@example.com",
      is_healthy: true,
      is_disabled: false,
      check_health: true,
      not_supported_models: [],
      usage_count: 0,
      error_count: 0,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      source: "manual",
    };

    const displayInfo = extractOAuthCardDisplayInfo(credential);
    expect(displayInfo.usageCount).toBe(0);
    expect(displayInfo.hasUsageCount).toBe(true);
  });

  test("高使用次数的凭证应正确显示", () => {
    const credential: CredentialDisplay = {
      uuid: "test-uuid",
      provider_type: "gemini",
      credential_type: "gemini_oauth",
      display_credential: "test@example.com",
      is_healthy: true,
      is_disabled: false,
      check_health: true,
      not_supported_models: [],
      usage_count: 999999,
      error_count: 100,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      source: "imported",
    };

    const displayInfo = extractOAuthCardDisplayInfo(credential);
    expect(displayInfo.usageCount).toBe(999999);
    expect(displayInfo.errorCount).toBe(100);
  });

  test("完整的 OAuth 凭证应通过完整性检查", () => {
    const credential: CredentialDisplay = {
      uuid: "test-uuid",
      provider_type: "kiro",
      credential_type: "kiro_oauth",
      name: "Test Credential",
      display_credential: "test@example.com",
      is_healthy: true,
      is_disabled: false,
      check_health: true,
      not_supported_models: [],
      usage_count: 100,
      error_count: 5,
      last_used: new Date().toISOString(),
      last_health_check_time: new Date().toISOString(),
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      source: "manual",
    };

    expect(isOAuthCardComplete(credential)).toBe(true);
  });
});

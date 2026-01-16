/**
 * 窗口控制 API
 *
 * 提供窗口大小调整、位置控制等功能
 */

import { safeInvoke } from "@/lib/dev-bridge";

/**
 * 窗口大小
 */
export interface WindowSize {
  width: number;
  height: number;
}

/**
 * 窗口大小选项
 */
export interface WindowSizeOption {
  id: string;
  name: string;
  description: string;
  size: WindowSize;
}

/**
 * 窗口控制 API
 */
export const windowApi = {
  /**
   * 获取当前窗口大小
   *
   * @returns 当前窗口大小
   */
  async getWindowSize(): Promise<WindowSize> {
    return safeInvoke("get_window_size");
  },

  /**
   * 设置窗口大小
   *
   * @param size - 新的窗口大小
   */
  async setWindowSize(size: WindowSize): Promise<void> {
    return safeInvoke("set_window_size", { size });
  },

  /**
   * 获取所有可用的窗口大小选项
   *
   * @returns 窗口大小选项列表
   */
  async getWindowSizeOptions(): Promise<WindowSizeOption[]> {
    return safeInvoke("get_window_size_options");
  },

  /**
   * 设置窗口为指定的预设大小
   *
   * @param optionId - 窗口大小选项 ID
   * @returns 之前的窗口大小（用于恢复）
   */
  async setWindowSizeByOption(optionId: string): Promise<WindowSize> {
    return safeInvoke("set_window_size_by_option", { optionId });
  },

  /**
   * 切换全屏模式
   *
   * @returns 是否进入了全屏模式
   */
  async toggleFullscreen(): Promise<boolean> {
    return safeInvoke("toggle_fullscreen");
  },

  /**
   * 检查是否处于全屏模式
   *
   * @returns 是否处于全屏模式
   */
  async isFullscreen(): Promise<boolean> {
    return safeInvoke("is_fullscreen");
  },

  /**
   * 居中窗口
   */
  async centerWindow(): Promise<void> {
    return safeInvoke("center_window");
  },
};

/**
 * 预定义的窗口大小
 */
export const WindowSizes = {
  /** 紧凑模式 */
  compact: { width: 1000, height: 700 } as WindowSize,

  /** 默认窗口大小 */
  default: { width: 1200, height: 800 } as WindowSize,

  /** 大屏模式 */
  large: { width: 1920, height: 1200 } as WindowSize,

  /** 超大屏模式 */
  extraLarge: { width: 2560, height: 1440 } as WindowSize,
};

export default windowApi;

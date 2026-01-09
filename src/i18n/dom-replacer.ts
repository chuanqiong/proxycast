/**
 * DOM Text Replacer Utility
 *
 * Replaces Chinese text in the DOM with translated text using a TreeWalker.
 * This is the core of the Patch Layer architecture.
 *
 * Key features:
 * - Walks the entire DOM tree to find text nodes
 * - Replaces Chinese text with translations based on the current language
 * - Skips script, style, and already patched nodes
 * - Handles multiple Chinese segments in a single text node
 * - Marks patched nodes to avoid double-patching
 */

import { getTextMap, Language } from "./text-map";

/**
 * Escape special regex characters in a string
 */
function escapeRegExp(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Replace text in DOM nodes with translations
 *
 * @param language - Target language ('zh' or 'en')
 */
export function replaceTextInDOM(language: Language): void {
  const patches = getTextMap(language);
  const startTime = performance.now();

  // Sort patches by length (longest first) to avoid partial replacements
  // This ensures "初次设置向导" is replaced before "初次" or "设置"
  const sortedPatches = Object.entries(patches)
    .filter(([zh]) => !zh.startsWith("//")) // Skip comment entries
    .sort(([a], [b]) => b.length - a.length); // Sort by length descending

  // Create a TreeWalker to traverse all text nodes
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode: (node) => {
        // Skip script, style, and already patched nodes
        const parent = node.parentElement;
        if (!parent) return NodeFilter.FILTER_REJECT;

        const tagName = parent.tagName;
        if (
          tagName === "SCRIPT" ||
          tagName === "STYLE" ||
          parent.hasAttribute("data-i18n-patched")
        ) {
          return NodeFilter.FILTER_REJECT;
        }

        // 跳过输入框和文本域（避免影响用户输入）
        if (tagName === "INPUT" || tagName === "TEXTAREA") {
          return NodeFilter.FILTER_REJECT;
        }

        // 跳过可编辑元素
        if (parent.isContentEditable) {
          return NodeFilter.FILTER_REJECT;
        }

        return NodeFilter.FILTER_ACCEPT;
      },
    },
  );

  const nodesToReplace: Array<{ node: Text; text: string }> = [];

  let node: Node | null;
  while ((node = walker.nextNode())) {
    const text = node.textContent;
    if (!text) continue;

    // Apply patches from longest to shortest to avoid partial replacements
    let newText = text;
    let hasMatch = false;

    for (const [zh, replacement] of sortedPatches) {
      // Use 'g' flag for global replacement (all occurrences)
      // Escape regex special characters to avoid errors
      const escaped = escapeRegExp(zh);
      const regex = new RegExp(escaped, "g");
      const replaced = newText.replace(regex, replacement);
      if (replaced !== newText) {
        newText = replaced;
        hasMatch = true;
      }
    }

    if (hasMatch) {
      nodesToReplace.push({
        node: node as Text,
        text: newText,
      });
    }
  }

  // Apply replacements (batch for performance)
  nodesToReplace.forEach(({ node, text }) => {
    node.textContent = text;
    // Mark as patched to avoid double-patching
    node.parentElement?.setAttribute("data-i18n-patched", "true");
  });

  const endTime = performance.now();
  const duration = endTime - startTime;

  // Log if slow (> 50ms)
  if (duration > 50) {
    console.warn(`[i18n] DOM replacement took ${duration.toFixed(2)}ms`);
  } else {
    console.debug(`[i18n] DOM replacement took ${duration.toFixed(2)}ms`);
  }

  // Track for analytics (optional)
  if (window.__I18N_METRICS__) {
    window.__I18N_METRICS__.patchTimes.push(duration);
  }
}

// Declare global type for metrics
declare global {
  interface Window {
    __I18N_METRICS__?: {
      patchTimes: number[];
      languageChanges: number;
    };
  }
}

window.__I18N_METRICS__ = {
  patchTimes: [],
  languageChanges: 0,
};

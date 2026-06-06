import { app } from "../../../scripts/app.js";

const SPECIAL_OPTIONS = ["(Auto-detect)", "(Not required)"];
const TARGET_NODES = {
    VisionLLMNode: { modelWidget: "model", localPrefix: "" },
    QwenImageEditPromptGenerator: { modelWidget: "llm_model", localPrefix: "Local: " },
    WanVideoPromptGenerator: { modelWidget: "llm_model", localPrefix: "Local: " },
};
const QWEN_NODE_NAME = "QwenImageEditPromptGenerator";
const WAN_NODE_NAME = "WanVideoPromptGenerator";

function normalizePath(value) {
    return String(value || "").replaceAll("\\", "/");
}

function dirname(value) {
    const normalized = normalizePath(value);
    const index = normalized.lastIndexOf("/");
    return index >= 0 ? normalized.slice(0, index) : "";
}

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name) || null;
}

function normalizeApiModelName(value) {
    return String(value || "").split(" (", 1)[0];
}

function isVisionApiModel(value) {
    const model = normalizeApiModelName(value);
    return model.startsWith("qwen3.7-plus")
        || model.startsWith("qwen3.6-")
        || model.startsWith("qwen-vl-");
}

function isLocalModel(value, config) {
    return Boolean(config.localPrefix && String(value || "").startsWith(config.localPrefix));
}

function isPlaceholderOption(value) {
    return String(value || "").startsWith("(");
}

function chooseReplacementModel(previousValue, nextOptions, config) {
    const wasLocal = isLocalModel(previousValue, config);
    const sameKind = nextOptions.find((value) => isLocalModel(value, config) === wasLocal && !isPlaceholderOption(value));
    return sameKind || nextOptions.find((value) => !isPlaceholderOption(value)) || nextOptions[0];
}

function getModelRelativePath(rawValue, config) {
    if (typeof rawValue !== "string") {
        return null;
    }

    if (config.localPrefix) {
        if (!rawValue.startsWith(config.localPrefix)) {
            return null;
        }
        return normalizePath(rawValue.slice(config.localPrefix.length));
    }

    if (rawValue.startsWith("(")) {
        return null;
    }

    return normalizePath(rawValue);
}

function syncMmprojOptions(node, config) {
    const modelWidget = getWidget(node, config.modelWidget);
    const mmprojWidget = getWidget(node, "mmproj");
    if (!modelWidget || !mmprojWidget?.options) {
        return;
    }

    if (!Array.isArray(node.__allMmprojOptions)) {
        node.__allMmprojOptions = [...(mmprojWidget.options.values || [])].filter(
            (value) => !SPECIAL_OPTIONS.includes(value)
        );
    }

    const modelRelativePath = getModelRelativePath(modelWidget.value, config);
    const modelDir = modelRelativePath ? dirname(modelRelativePath) : null;

    const filtered = modelDir
        ? node.__allMmprojOptions.filter((value) => dirname(value) === modelDir)
        : [];

    const nextOptions = [...filtered, ...SPECIAL_OPTIONS];
    mmprojWidget.options.values = nextOptions;

    if (!nextOptions.includes(mmprojWidget.value)) {
        mmprojWidget.value = "(Auto-detect)";
    }

    node.setDirtyCanvas?.(true, true);
}

function syncVisionRequiredModelOptions(node, config, visionRequired) {
    const modelWidget = getWidget(node, config.modelWidget);
    if (!modelWidget?.options) {
        return;
    }

    if (!Array.isArray(node.__allLlmModelOptions)) {
        node.__allLlmModelOptions = [...(modelWidget.options.values || [])];
    }

    const allOptions = node.__allLlmModelOptions;
    const nextOptions = visionRequired
        ? allOptions.filter((value) => (
            isLocalModel(value, config)
            || isPlaceholderOption(value)
            || isVisionApiModel(value)
        ))
        : allOptions;

    modelWidget.options.values = nextOptions;
    if (!nextOptions.includes(modelWidget.value) && nextOptions.length > 0) {
        modelWidget.value = chooseReplacementModel(modelWidget.value, nextOptions, config);
        syncMmprojOptions(node, config);
    }

    node.setDirtyCanvas?.(true, true);
}

function syncNodeOptions(node, nodeName, config) {
    if (nodeName === QWEN_NODE_NAME) {
        const promptStyleWidget = getWidget(node, "prompt_style");
        syncVisionRequiredModelOptions(node, config, promptStyleWidget?.value === "Qwen-Image-Edit");
    }
    if (nodeName === WAN_NODE_NAME) {
        const taskWidget = getWidget(node, "task_type");
        syncVisionRequiredModelOptions(node, config, taskWidget?.value === "Image-to-Video");
    }
    syncMmprojOptions(node, config);
}

function attachMmprojFilter(nodeType, nodeData) {
    const config = TARGET_NODES[nodeData.name];
    if (!config) {
        return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = originalOnNodeCreated?.apply(this, arguments);

        const modelWidget = getWidget(this, config.modelWidget);
        if (modelWidget) {
            const originalCallback = modelWidget.callback;
            modelWidget.callback = (...args) => {
                const callbackResult = originalCallback?.apply(modelWidget, args);
                syncMmprojOptions(this, config);
                return callbackResult;
            };
        }

        if (nodeData.name === QWEN_NODE_NAME || nodeData.name === WAN_NODE_NAME) {
            const controllerWidgetName = nodeData.name === QWEN_NODE_NAME ? "prompt_style" : "task_type";
            const controllerWidget = getWidget(this, controllerWidgetName);
            if (controllerWidget) {
                const originalCallback = controllerWidget.callback;
                controllerWidget.callback = (...args) => {
                    const callbackResult = originalCallback?.apply(controllerWidget, args);
                    syncNodeOptions(this, nodeData.name, config);
                    return callbackResult;
                };
            }
        }

        syncNodeOptions(this, nodeData.name, config);
        return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        queueMicrotask(() => syncNodeOptions(this, nodeData.name, config));
        return result;
    };
}

app.registerExtension({
    name: "ComfyUI.MultiModalPromptNodes.MmprojFilter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        attachMmprojFilter(nodeType, nodeData);
    },
});

import { app } from "../../../scripts/app.js";

const SPECIAL_OPTIONS = ["(Auto-detect)", "(Not required)"];
const TARGET_NODES = {
    VisionLLMNode: { modelWidget: "model", localPrefix: "" },
    QwenImageEditPromptGenerator: { modelWidget: "llm_model", localPrefix: "Local: " },
    WanVideoPromptGenerator: { modelWidget: "llm_model", localPrefix: "Local: " },
};

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

        syncMmprojOptions(this, config);
        return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        queueMicrotask(() => syncMmprojOptions(this, config));
        return result;
    };
}

app.registerExtension({
    name: "ComfyUI.MultiModalPromptNodes.MmprojFilter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        attachMmprojFilter(nodeType, nodeData);
    },
});

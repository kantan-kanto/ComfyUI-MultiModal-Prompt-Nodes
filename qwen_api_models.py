QWEN_API_MAIN_MODELS = [
    "qwen3.7-plus",
    "qwen3.7-max",
    "qwen3.7-plus-2026-05-26",
    "qwen3.7-max-2026-05-20",
    "qwen3.6-plus",
    "qwen3.6-flash",
    "qwen3.6-plus-2026-04-02",
    "qwen3.6-flash-2026-04-16",
    "qwen3.6-35b-a3b",
]

QWEN_API_OFFLINE_SINCE_2026_05_13 = [
    "qwen-vl-max-latest",
    "qwen-vl-max-2025-08-13",
    "qwen-vl-max-2025-04-08",
    "qwen-max-latest",
]

QWEN_API_OFFLINE_SCHEDULED_2026_07_13 = [
    "qwen-vl-max",
    "qwen-vl-plus",
    "qwen-plus",
    "qwen-max",
    "qwen-turbo",
]

QWEN_API_LEGACY_MODELS = [
    "qwen-plus-latest",
]

QWEN_API_MODELS = (
    QWEN_API_MAIN_MODELS
    + [f"{model} (deprecated: announced offline since 2026-05-13)" for model in QWEN_API_OFFLINE_SINCE_2026_05_13]
    + [f"{model} (deprecated: offline scheduled 2026-07-13)" for model in QWEN_API_OFFLINE_SCHEDULED_2026_07_13]
    + [f"{model} (deprecated: legacy, prefer Qwen3.7)" for model in QWEN_API_LEGACY_MODELS]
)

QWEN_API_MODEL_IDS = set(
    QWEN_API_MAIN_MODELS
    + QWEN_API_OFFLINE_SINCE_2026_05_13
    + QWEN_API_OFFLINE_SCHEDULED_2026_07_13
    + QWEN_API_LEGACY_MODELS
)

QWEN_API_VISION_MODEL_IDS = {
    model for model in QWEN_API_MODEL_IDS
    if model.startswith(("qwen3.7-plus", "qwen3.6-", "qwen-vl-"))
}


def normalize_api_model_name(model):
    return model.split(" (", 1)[0]

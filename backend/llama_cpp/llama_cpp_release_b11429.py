# https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0
# Aka b11429

_LLAMA_RELEASE = "b11429"
_LLAMA_RELEASE_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{_LLAMA_RELEASE}/{{}}"
)

_RELEASE_ASSETS: dict[tuple[str, str, str], tuple[tuple[str, str], ...]] = {
    ("windows", "x64", "cpu"): (
        (
            "llama-b11429-bin-win-cpu-x64.zip",
            "1283323272b04cd07905816a597a0da810918102de958f4ff6f7bbaa70ed2efe",
        ),
    ),
    ("windows", "arm64", "cpu"): (
        (
            "llama-b11429-bin-win-cpu-arm64.zip",
            "ee0f631a9e58b146ff50714099d9cb498906af3143a773b580093a9214a8d1e5",
        ),
    ),
    ("windows", "x64", "cuda"): (
        (
            "llama-b11429-bin-win-cuda-13.4-x64.zip",
            "76ddc6eff2389570789ed608881efc6977751a722015d8e8c94f302224ff1a3a",
        ),
        (
            "cudart-llama-bin-win-cuda-13.4-x64.zip",
            "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
        ),
    ),
    ("windows", "arm64", "cuda"): (
        (
            "llama-b11429-bin-win-cuda-13.4-arm64.zip",
            "930cff325a0eb15e1232d5b5275b818c76c9c32a398788f18bb05273d3110c72",
        ),
        (
            "cudart-llama-bin-win-cuda-13.4-arm64.zip",
            "642dcde8805b3e3165ca710a5443b3b4044b27d96bd3ee3132473988c9bcb774",
        ),
    ),
    ("windows", "x64", "rocm"): (
        (
            "llama-b11429-bin-win-rocm-10.0-x64.zip",
            "289a8453bb222f0648f91eff3d55133c934119c841eb369b704aafae1d4719d2",
        ),
    ),
    ("macos", "x64", "cpu"): (
        (
            "llama-b11429-bin-macos-x64.tar.gz",
            "29ac3ea02be6bd143e824973f2cc5fa74bc4094393a9eaab0ff6814f19dd8522",
        ),
    ),
    ("macos", "arm64", "cpu"): (
        (
            "llama-b11429-bin-macos-arm64.tar.gz",
            "740288ec6887be94280a5dfa25b5e23a78285cab104519e6c7e218904ee82459",
        ),
    ),
    ("linux", "x64", "cpu"): (
        (
            "llama-b11429-bin-ubuntu-x64.tar.gz",
            "f6d25dde8f51133143d1453da4fd5f73b145127177612a283bf7995957af3392",
        ),
    ),
    ("linux", "arm64", "cpu"): (
        (
            "llama-b11429-bin-ubuntu-arm64.tar.gz",
            "ed44c0f79c3d02424f62bdc87e106318b0b89f49050e2e80c5f352723d14e00b",
        ),
    ),
    ("linux", "x64", "cuda"): (
        (
            "llama-b11429-bin-ubuntu-cuda-13.4-x64.tar.gz",
            "8082b7eaa74a714c9fecca19128f751c8e32da763ee8096b8ad1e824da7621d3",
        ),
        (
            "cudart-llama-b11429-bin-ubuntu-cuda-13.4-x64.tar.gz",
            "93d18648d815b2bd624d83d82f653e1db97afb478f02064305fe3cf570040a6d",
        ),
    ),
    ("linux", "arm64", "cuda"): (
        (
            "llama-b11429-bin-ubuntu-cuda-13.4-arm64.tar.gz",
            "9c76d072276c0faa7fc1b5cbf715b4186dd2e8d7f16acd20b67daab24c82bd22",
        ),
        (
            "cudart-llama-b11429-bin-ubuntu-cuda-13.4-arm64.tar.gz",
            "ad62e46cdc2e8636fa91e883b9e9ad779516f61cc478ece1c90e5dbc905a7d29",
        ),
    ),
    ("linux", "x64", "rocm"): (
        (
            "llama-b11429-bin-ubuntu-rocm-10.0-x64.tar.gz",
            "bd73e52146551f07a4c539bbe9da42f3b680c44ddc252fce61ab86d567a5932a",
        ),
    ),
}

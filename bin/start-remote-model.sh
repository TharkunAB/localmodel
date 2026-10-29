#!/usr/bin/env bash
# start-remote-model.sh -- standalone, consolidated rental-GPU launcher (2026-10-20).
# Requires Linux and Python >=3.10. DeepSeek uses a signed CUDA 13 engine.
# Other profiles require a CUDA development image. Keep model files on local storage.
# No separate updater/helper is required. The Python implementation is embedded.
# Reconstructed from the recorded profiles and the supplied Ornith add-on;
# this is not a claim of byte-for-byte preservation of the missing original.
#
# DeepSeek: official Strata source/control with signed native CUDA engine.
# Use --prepare-only to install/verify files and create the native pack.
# Serving uses prepared local files under an inherited outgoing-network block.
# Qwen/Flint/Ornith/MiniMax keep their supplied llama.cpp execution paths.
# Run --help for exact profile, memory, preview and migration controls.
set -Eeuo pipefail
command -v python3 >/dev/null 2>&1 || { echo 'ERROR: Python 3.10+ is required.' >&2; exit 1; }
exec python3 - "$@" <<'PYTHON_START_REMOTE_MODEL'
from __future__ import annotations
import argparse
import bisect
import csv
import concurrent.futures
import ctypes
import dataclasses
import fcntl
import hashlib
import json
import io
import math
import platform
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tarfile
import threading
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

VERSION = '2026-10-29-deepseek-strata-dspark-1'
GIB, MIB = 1024**3, 1024**2
SOURCE = 'https://github.com/ggml-org/llama.cpp.git'
STABLE = '620d2225cc436ab0b001e3200e590e3b286ef9ab'  # released v0.7.0
FLINT_RUNTIME = 'a37cc46104eb50900d5ab4a9ba02a0abdac039db'  # upstream merged Flint+Jinja hex escapes
DEEPSEEK_RUNTIME = 'b72050d03ae14bb3cb5138ecefab3f3f65157d31'  # draft PR 30817
# MiniMax profiles are opt-in; existing auto selection is unchanged.
# Checked 2026-10-24 against HF API, four remote GGUF headers, and upstream source.
MINIMAX_PROFILES = (
    'minimax-pro6000-dual-q2k', 'minimax-a40-quad-q2k',
    'minimax-pro6000-single-q2k', 'minimax-a40-dual-q2k',
    'minimax-a40-triple-q2k', 'minimax-h200-single-q2k', 'minimax-h200-dual-q2k',
)
MINIMAX_REPO = 'webmp3/Sakura-MiniMax-M3.1-Flash-Q2_K-GGUF'
MINIMAX_REV = '7c0300116b3a202425cf3b5da6f87df32a3d892f'
MINIMAX_SOURCE_REV = '54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e'
MINIMAX_RUNTIME = '970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc'
MINIMAX_SAKURA_PATCH_SHA256 = '666bf3ccf76cdeec9b0e6b40b248440cd6fd234e9c43fb10130e3ec3230de289'
# Combined patch additionally exposes target hidden states using upstream DFlash API.
MINIMAX_PATCH_SHA256 = 'e43c7c7ceb99e7128a2eeff735de67f513ea0a4cb0db42fc435485fd976b7494'
MINIMAX_DFLASH_REPO = 'incoai/MiniMax-M3.1-Flash-DFlash2'
MINIMAX_DFLASH_REV = 'ac07a7c1ecaae26a7ace595756124dca974b432d'
MINIMAX_DFLASH_MAX_BYTES = 4_350_000_000  # F16 conversion upper bound, not measured output
MINIMAX_DFLASH_FILES = {'config.json': {'size': 1252, 'sha256': '64c32db3c3bc5de64a048d949c5acd91352ae9b0f02ef636e3382c838f1d0815'}, 'model.safetensors': {'size': 4247776488, 'sha256': 'ad861f4ddd8f7204cda1f407837403e9095e1fb1f36d4c9f4188185261cc4fd1'}}
MINIMAX_TOKENIZER_FILES = {'config.json': {'size': 9122, 'sha256': 'df6a3668e2de5b8dc0e883e3767495ad17af7e4d69fabe9683088be015fda9ab'}, 'generation_config.json': {'size': 150, 'sha256': '272d19efac5060ff8c9f76c7ed67909a5da1be466284ee4f1f1bd2a810c6fad2'}, 'tokenizer.json': {'size': 9731500, 'sha256': 'bb1f1626cf01448f1e3b6036d0a061ffc66c91d9046aada14ea23a5441b5ad6e'}, 'tokenizer_config.json': {'size': 11288, 'sha256': '15caca9f74fbce9b86dc014e9324cbefa56703d81ccd6d56020faa1c41d2eae5'}, 'chat_template.jinja': {'size': 11633, 'sha256': '961e3f96b2c08d26555efdfeb3b384d5e81181bf0cfc04edbadc31c6e507b431'}, 'added_tokens.json': {'size': 1660, 'sha256': '63f277741fb061e615cb604286a845d971178ba4716732a59cf42b3752c0ce4e'}, 'special_tokens_map.json': {'size': 277, 'sha256': '401b624ec44507183d1dc65c74f7f3387699a77971aa19fa5a0cc57a3989ba97'}, 'merges.txt': {'size': 2414077, 'sha256': 'ca47ea60cf5bd48832586adc264f439d0ea4254176b24a95f85eaa9750e2b5f9'}, 'vocab.json': {'size': 4705413, 'sha256': 'b44c066b5dc34c800c4e3ecbd85f3e95ce3bfdbf8a5fe30223e005175103578a'}}
MINIMAX_BYTES = 155_983_334_400
# Exact native tensor sizes from pinned build manifest, reconciled with GGUF headers.
MINIMAX_DENSE_LAYER_BYTES = 138_708_992
MINIMAX_MOE_LAYER_BYTES = 2_703_903_232
MINIMAX_EXPERT_BYTES = 2_623_537_152  # routed gate + up + down, per MoE layer
MINIMAX_INPUT_BYTES = 403_329_024    # input embedding stays on CPU in this runtime
MINIMAX_OUTPUT_BYTES = 1_008_347_136 # output weight + output norm
MINIMAX_METADATA_BYTES = 33_047_040
MINIMAX_FILES = {'Sakura-MiniMax-M3.1-Flash-Q2_K-00001-of-00004.gguf': {'size': 47802425920, 'sha256': '9913da5047dbeff274866de7b93fa5e4b50de54d537fb264c2008486056da4a0', 'header_size': 8267328, 'header_sha256': 'cff2e99924c863ccf522cc59f12eb91363255de9dbce3033ee9f246310d2a4f8'}, 'Sakura-MiniMax-M3.1-Flash-Q2_K-00002-of-00004.gguf': {'size': 47805433376, 'sha256': '79f4b6a3c1aac7a093566dc3c928e0302e56199afd20f0df7792c5937ee675ab', 'header_size': 8264736, 'header_sha256': 'b66c289cf392e99d9a3cf30b39a3874f5e4834045564ebf3d3d3eb8a27004089'}, 'Sakura-MiniMax-M3.1-Flash-Q2_K-00003-of-00004.gguf': {'size': 47640433696, 'sha256': 'af83db53b63fae96359185e6597107860d7e774be0ea667b8dc2264535030f6a', 'header_size': 8265760, 'header_sha256': 'd3f6e1164e456b3924202319ae4f203bf129091b2260af2ac0d321290cd9a465'}, 'Sakura-MiniMax-M3.1-Flash-Q2_K-00004-of-00004.gguf': {'size': 12735041408, 'sha256': '7619f13d6a1c2f931bc2175242aea2919110a8e1627a1e333cf37867499d79b9', 'header_size': 8249216, 'header_sha256': 'bf967d4b44390f332fc7680e0a24265483c314288718e1dc0450b20929bf75fb'}}
MINIMAX_PATCHED_FILES = {'src/llama-arch.cpp': 'b16b19c83d91af8cf8ff03e433011afdee4c9e937d0333ada568441c57292cb2', 'src/llama-hparams.h': 'f249176375b42bc022623c71bcb9c9493ccfaa6bba0d5eefa9866f890a2ea868', 'src/llama-model.cpp': '45f6b23c3e6fdc8d50a675cfe183cd88a6d25b07eedeb3805456a779102339a2', 'src/models/minimax-m3.cpp': '3ffceb3ec185d0b752017423bf1cb37f46e925faa91e59ff196e48010e01df9f'}
MINIMAX_PATCH = r'''
diff --git a/src/llama-arch.cpp b/src/llama-arch.cpp
index a364174..bc626ea 100644
--- a/src/llama-arch.cpp
+++ b/src/llama-arch.cpp
@@ -1193,6 +1193,7 @@ const char * llm_arch_name(llm_arch arch) {
 }
 
 llm_arch llm_arch_from_string(const std::string & name) {
+    if (name == "minimaxm31_sakura") { return LLM_ARCH_MINIMAX_M3; }
     for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT
         if (kv.second == name) {
             return kv.first;
diff --git a/src/llama-hparams.h b/src/llama-hparams.h
index 719d708..efe70a2 100644
--- a/src/llama-hparams.h
+++ b/src/llama-hparams.h
@@ -307,6 +307,7 @@ struct llama_hparams {
     // DSA (deepseek sparse attention)
     uint32_t indexer_n_head    = 0;
     uint32_t indexer_head_size = 0;
+    uint32_t indexer_first_layer = 0;
     uint32_t indexer_top_k     = 0;
     uint32_t indexer_kpool     = 0; // k-pool size
     bool     indexer_kpool_select_tail = true;
diff --git a/src/llama-model.cpp b/src/llama-model.cpp
index 4cb0409..b684aea 100644
--- a/src/llama-model.cpp
+++ b/src/llama-model.cpp
@@ -2815,9 +2815,9 @@ llama_memory_i * llama_model::create_memory(const llama_memory_params & params,
             } break;
         case LLM_ARCH_MINIMAX_M3:
             {
-                // sparse (MSA) layers carry an indexer key cache, but leading dense layers do not
+                // Only sparse attention layers carry an indexer key cache.
                 llama_kv_cache::layer_filter_cb filter_idx =
-                    [&](int32_t il) { return (uint32_t) il >= hparams.n_layer_dense_lead; };
+                    [&](int32_t il) { return (uint32_t) il >= hparams.indexer_first_layer; };
 
                 res = new llama_kv_cache_msa(
                         *this,
diff --git a/src/models/minimax-m3.cpp b/src/models/minimax-m3.cpp
index 35e7b1a..69e5b7d 100644
--- a/src/models/minimax-m3.cpp
+++ b/src/models/minimax-m3.cpp
@@ -13,6 +13,9 @@
 void llama_model_minimax_m3::load_arch_hparams(llama_model_loader & ml) {
     ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
     ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
+    hparams.indexer_first_layer = hparams.n_layer_dense_lead;
+    ml.get_key("minimax-m3.attention.indexer.first_layer", hparams.indexer_first_layer, false);
+    if (hparams.indexer_first_layer > hparams.n_layer()) { throw std::runtime_error("Invalid first sparse layer"); }
     ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
     ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
     ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
@@ -79,7 +82,8 @@ void llama_model_minimax_m3::load_arch_tensors(llama_model_loader &) {
             layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
             layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
             layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
-
+        }
+        if ((uint32_t) i >= hparams.indexer_first_layer) {
             // indexer
             layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", i), {n_embd, hparams.indexer_n_head * hparams.indexer_head_size}, 0);
             layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", i), {n_embd, hparams.indexer_head_size}, 0);
@@ -233,6 +237,9 @@ llama_model_minimax_m3::graph::graph(const llama_model & model, const llm_graph_
     const bool fa_on       = cparams.flash_attn;
     const bool streams_ok  = cparams.n_seq_max == 1 || !cparams.kv_unified;
     const bool msa_enabled = fa_on && streams_ok;
+    if (hparams.indexer_first_layer == 0 && !msa_enabled) {
+        throw std::runtime_error("MiniMax M3.1 requires flash attention and supported stream layout");
+    }
 
     auto * inp_attn = build_attn_inp_kv_msa(msa_enabled);
 
@@ -301,6 +308,8 @@ llama_model_minimax_m3::graph::graph(const llama_model & model, const llm_graph_
     ggml_tensor * inp_out_ids = build_inp_out_ids();
 
     for (int il = 0; il < n_layer; ++il) {
+        // DFlash2 consumes the input of selected target layers.
+        res->t_layer_inp[il] = inpL;
         ggml_tensor * inpSA = inpL;
 
         // self-attention
@@ -331,7 +340,7 @@ llama_model_minimax_m3::graph::graph(const llama_model & model, const llm_graph_
             cb(Kcur, "Kcur", il);
             cb(Vcur, "Vcur", il);
 
-            const bool is_sparse = msa_enabled && il >= (int) hparams.n_layer_dense_lead;
+            const bool is_sparse = msa_enabled && il >= (int) hparams.indexer_first_layer;
 
             if (!is_sparse) {
                 cur = build_attn(inp_attn, model.layers[il].wo, NULL, model.layers[il].wo_s,
'''

HUB_VERSION = '1.0.1'
FLINT_REPO = 'RecursiveSI/flint-nvfp4-gguf'
FLINT_BYTES = 22_900_000_000   # publisher-reported, reconciled with file manifest before download
FLINT_NAME = 'flint-nvfp4.gguf'
QWEN_REPO = 'ggml-org/Qwen3.8-27B-GGUF'
QWEN_REV = 'a316ea527fde32c12487be7574e2db5ac2ad2a7e'
ORNITH_REPO = 'unsloth/Ornith-1.5-122B-GGUF'
ORNITH_REV = '3842d003341507c216295c6f28dfcd30349925c6'
ORNITH_OFFICIAL = 'ornith-ai/Ornith-1.5-122B'
ORNITH_OFFICIAL_REV = '346ec829dd63e107aad8771d74e885e901cec95d'
DS_REPO = 'unsloth/DeepSeek-V4.1-Flash-1017-GGUF'
DS_REV = 'f2e5a380fc945842d9783292315307eeab418684'
# Optional Unsloth Q2_K, pinned at the existing DS_REV; not DASLab GSQ-RCO.
DS_Q2_FILES = {'UD-Q2_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00001-of-00007.gguf': {'size': 5780927, 'sha256': '0c2a2031bd93ebd60aa6a4f3912775ba3990b5f39b24c6edf9ea029d2f387a28'}, 'UD-Q2_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00002-of-00007.gguf': {'size': 49290279808, 'sha256': '605972fece1b43277b9ff86a2cd79c9002ff424caefc507f20b6ccd47ae4ab39'}, 'UD-Q2_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00003-of-00007.gguf': {'size': 49745070208, 'sha256': '1cb58786245bd8c38478174eecc648b12691bb79c2bb2f446c135ab00263580e'}, 'UD-Q2_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00004-of-00007.gguf': {'size': 48680128800, 'sha256': '7b753295dd32534dfea52b2e9b5c4741b8007a955e38702ee37034271df7526e'}, 'UD-Q2_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00005-of-00007.gguf': {'size': 49970937696, 'sha256': '336be674937fe29683e7aeee7c6af32f896b864931a6c916907b78e0f3692de4'}, 'UD-Q2_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00006-of-00007.gguf': {'size': 1929325664, 'sha256': '192694fb09612e1200cce9ae35f3645ad39cf8aa43e94417070af9c4f4ecb43b'}, 'UD-Q2_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00007-of-00007.gguf': {'size': 135172021792, 'sha256': 'a653c5309b7c74a6c41580aaa8c2d2bbdae0342253bca2cd3fc9e69d05f3d0eb'}}
DS_Q2_BYTES = 334_793_544_895
DS_Q2_ENGRAM_BYTES = 135_172_021_792
GSQ_REPO = 'ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF'
GSQ_REV = 'cb02422c900df489cea5833e8a92171224059b3c'
GSQ_ENGRAM_NAME = 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf'
GSQ_FILES = {'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf': {'size': 39358830464, 'sha256': '6e3f0d5ff53d00428ec32c079db9e8255cb410ccaa435e5342aaa8a47fe1339d'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00002-of-00004.gguf': {'size': 39176113248, 'sha256': '34d68c99fdae9948f9f0af0dafe308e805ce0a78d2c48641b24cc57242fdc0af'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00003-of-00004.gguf': {'size': 39997334720, 'sha256': 'a08468c83955e0c4d3a9eb3f7b86c45b7bbac55d99c4640039a296a8a4b327ea'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00004-of-00004.gguf': {'size': 2188841824, 'sha256': 'b05f26a92325aa6845cebcb3ec1f93d2e65ebf89bdc3f805695960d2f0681832'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf': {'size': 39535776992, 'sha256': '4c65307992ab586c527ca1f6f64062aa96a33b23c02b7fed103f45f75e854798'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00002-of-00004.gguf': {'size': 39494617536, 'sha256': '933461a974fcccee741bf2f9798201e72ab4aa3f8a952be56f849896ca6ea30e'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00003-of-00004.gguf': {'size': 39423838496, 'sha256': 'bbf91fa346a85ebaf32dd4fd5c456b534d780d381e54f46ac90783a481373240'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00004-of-00004.gguf': {'size': 36771591232, 'sha256': 'e5b2f8781fab688cd4bc3bc7a69651b5dbb90ed94965657c6bead8bb4774f3cf'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf': {'size': 39075714624, 'sha256': 'd46efb67ccffd8fb5d7f18b423896381b9e06d8491e161c370ddf3b05b07e050'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00004.gguf': {'size': 39388449504, 'sha256': 'cec1b2c3da429dc768b8bb9bc4dc4403d81f74d34db660a6a34ee3e555bf2c70'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00004.gguf': {'size': 39246891616, 'sha256': '9d9c2140fcb134a0b9a74dedd0e5167e68077af6d08ce7f856bce112f0a35dd3'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00004.gguf': {'size': 20527837312, 'sha256': '9b790a2297ed3cdede379fc984f4a52bb95f3b23724c797af11d1fa6a69db059'}, 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf': {'size': 110595290816, 'sha256': 'eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a'}}
GSQ_RUNTIME = '0062e5930edf2f50c50254ba5080a6d37b024c6c'
GSQ_RUNTIME_SOURCE = 'https://github.com/smalinin/llama.cpp.git'
GSQ_PATCH_SHA256 = 'd7fd6f1a0240effc0962702b1e8cca9f12d21c5269b96eb146d3751b72d7afec'
GSQ_LOADER_SHA256 = '20e6aa0b8bd8926112cfe0a0f2cfe3f774ed9eb89138096fa498f80c5869bd8d'
GSQ_PROFILES = {'deepseek-gsq-a40-dual': {'kind': 'a40', 'count': 2, 'quant': 'IQ1_M', 'context': 16384, 'engram': 'cpu'}, 'deepseek-gsq-a40-triple': {'kind': 'a40', 'count': 3, 'quant': 'IQ2_XXS', 'context': 16384, 'engram': 'cpu'}, 'deepseek-gsq-a40-quad': {'kind': 'a40', 'count': 4, 'quant': 'IQ2_XS', 'context': 1048576, 'engram': 'cpu'}, 'deepseek-gsq-pro6000-single': {'kind': 'pro6000', 'count': 1, 'quant': 'IQ1_M', 'context': 16384, 'engram': 'cpu'}, 'deepseek-gsq-pro6000-dual': {'kind': 'pro6000', 'count': 2, 'quant': 'IQ2_XS', 'context': 262144, 'engram': 'cpu'}, 'deepseek-gsq-pro6000-quad': {'kind': 'pro6000', 'count': 4, 'quant': 'IQ2_XS', 'context': 524288, 'engram': 'gpu'}, 'deepseek-gsq-h200': {'kind': 'h200', 'count': 1, 'quant': 'IQ2_XXS', 'context': 32768, 'engram': 'cpu'}, 'deepseek-gsq-h200-dual': {'kind': 'h200', 'count': 2, 'quant': 'IQ2_XS', 'context': 262144, 'engram': 'gpu'}, 'deepseek-gsq-b200': {'kind': 'b200', 'count': 1, 'quant': 'IQ2_XS', 'context': 262144, 'engram': 'cpu'}, 'deepseek-gsq-b300': {'kind': 'b300', 'count': 1, 'quant': 'IQ2_XS', 'context': 524288, 'engram': 'gpu'}, 'deepseek-gsq-ampere-multi': {'kind': 'ampere48', 'count': 0, 'quant': 'IQ2_XS', 'context': 524288, 'engram': 'cpu'}}
GSQ_LAYOUT = {'IQ1_M': {'expert_layer_bytes': [2654208000, 2654208000, 2654208000, 2654208000, 2654208000, 2654208000, 2654208000, 3220439040, 2654208000, 3220439040, 3220439040, 2937323520, 2760376320, 2760376320, 2654208000, 2654208000, 2760376320, 2654208000, 3220439040, 2760376320, 2760376320, 2760376320, 2760376320, 3220439040, 2654208000, 3326607360, 3220439040, 2654208000, 2866544640, 3220439040, 3326607360, 2654208000, 3220439040, 2866544640, 2654208000, 2654208000, 2866544640, 2654208000, 3220439040, 2937323520], 'layer_bytes': [2776941528, 2885157848, 2798374360, 2776941528, 2776941528, 2776941528, 2776941528, 3343172568, 2798374360, 3343172568, 3343172568, 3060057048, 2883109848, 2883109848, 2906590680, 2776941528, 2883109848, 2776941528, 3343172568, 2883109848, 2899299800, 2883109848, 2883109848, 3343172568, 2787754968, 3449340888, 3343172568, 2776941528, 3000091608, 3343172568, 3449340888, 2776941528, 3353986008, 2989278168, 2776941528, 2776941528, 3000091608, 2776941528, 3343172568, 3060057048], 'input_bytes': 455065600, 'output_bytes': 455086080, 'metadata_bytes': 5833728}, 'IQ2_XXS': {'expert_layer_bytes': [2972712960, 2972712960, 2972712960, 2972712960, 2972712960, 2972712960, 2972712960, 3149660160, 3149660160, 2972712960, 3503554560, 3326607360, 3326607360, 2972712960, 2972712960, 3326607360, 2972712960, 3609722880, 3503554560, 3609722880, 2972712960, 2972712960, 3503554560, 3645112320, 3326607360, 3503554560, 3645112320, 3609722880, 3503554560, 3326607360, 2972712960, 3326607360, 3326607360, 3645112320, 3609722880, 3609722880, 3326607360, 3326607360, 3786670080, 3928227840], 'layer_bytes': [3095446488, 3203662808, 3116879320, 3095446488, 3095446488, 3095446488, 3095446488, 3272393688, 3293826520, 3095446488, 3626288088, 3449340888, 3449340888, 3095446488, 3225095640, 3449340888, 3095446488, 3732456408, 3626288088, 3732456408, 3111636440, 3095446488, 3626288088, 3767845848, 3460154328, 3626288088, 3767845848, 3732456408, 3637101528, 3449340888, 3095446488, 3449340888, 3460154328, 3767845848, 3732456408, 3732456408, 3460154328, 3449340888, 3909403608, 4050961368], 'input_bytes': 455065600, 'output_bytes': 455086080, 'metadata_bytes': 5833728}, 'IQ2_XS': {'expert_layer_bytes': [3503554560, 3503554560, 3503554560, 3503554560, 3503554560, 3503554560, 3503554560, 3503554560, 3786670080, 3928227840, 3786670080, 3645112320, 3928227840, 3503554560, 3645112320, 3503554560, 3786670080, 4211343360, 3645112320, 3503554560, 3503554560, 3786670080, 3786670080, 4352901120, 4069785600, 4352901120, 3928227840, 3928227840, 3645112320, 3645112320, 4069785600, 3786670080, 3786670080, 3503554560, 3645112320, 3503554560, 3503554560, 3503554560, 4069785600, 3786670080], 'layer_bytes': [3626288088, 3734504408, 3647720920, 3626288088, 3626288088, 3626288088, 3626288088, 3626288088, 3930836440, 4050961368, 3909403608, 3767845848, 4050961368, 3626288088, 3897495000, 3626288088, 3909403608, 4334076888, 3767845848, 3626288088, 3642478040, 3909403608, 3909403608, 4475634648, 4203332568, 4475634648, 4050961368, 4050961368, 3778659288, 3767845848, 4192519128, 3909403608, 3920217048, 3626288088, 3767845848, 3626288088, 3637101528, 3626288088, 4192519128, 3909403608], 'input_bytes': 455065600, 'output_bytes': 455086080, 'metadata_bytes': 5833728}}
GSQ_HEADERS = {'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf': {'size': 5255232, 'sha256': 'e03fbfcb8a24381b04f0913f7cadbe11b0d267bf2a0f08503835878e6f260aa0'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00002-of-00004.gguf': {'size': 3168, 'sha256': 'ab13ca9e4c10728806eba3001dba8da0ca755bf78c25726d2b9c798428733e6e'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00003-of-00004.gguf': {'size': 40384, 'sha256': '384d9ac952738c36b4bb9409cde56030be3a25078a5234257ea2e97444bfb5af'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00004-of-00004.gguf': {'size': 16992, 'sha256': '45c7df98c75c8bb31e065eca4c4603bca576604d41a7d59665324b4cce779127'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf': {'size': 5254912, 'sha256': '254ffb6539f7e0db2d27b288c1b6918f0025819f3b2154d7ba46d56f3a4860d0'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00004.gguf': {'size': 2784, 'sha256': 'deffd09184bf1f7dbab95d63da485fc3c29416b5afd008e241d6b8fda379bc61'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00004.gguf': {'size': 2656, 'sha256': '689a1ee72d725ed9786c66a4666f1dc5f0aac18c988a22729eca30e309c82fa1'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00004.gguf': {'size': 55424, 'sha256': 'ef8fa95238e80650840e71553ed84df6c37e7aba580ebf4efa8778c0bd4a6636'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf': {'size': 5254560, 'sha256': 'abffcae4bc3a4251322de6ab116fa69fb1ec6104ed785f188859189d3a0ec6b9'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00002-of-00004.gguf': {'size': 2496, 'sha256': '8c10cb4b21d85abe1db3dcbfcf18088495e60f13547ca1851fb1626fc7444472'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00003-of-00004.gguf': {'size': 2336, 'sha256': 'a7e03e5773092bffd98f0d507f54d6845150aa0276911f7b28341d70a6323262'}, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00004-of-00004.gguf': {'size': 56384, 'sha256': 'c6be1958e3aeea37f500596fbb10417a7fbe0369ee4fb8051f59719bc25d77f3'}, 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf': {'size': 416, 'sha256': '97a55e86b495fdc4ae2aea6d52a337578d2e514dc099676164cda48c635f40d1'}}
GSQ_ENGRAM_TABLE_BYTES = (55_296_888_192, 55_298_402_208)
GSQ_ENGRAM_TENSOR_BYTES = sum(GSQ_ENGRAM_TABLE_BYTES)
GSQ_PATCH = r'''--- a/src/llama-model-loader.cpp
+++ b/src/llama-model-loader.cpp
@@ -720,6 +720,107 @@
         llm_kv = LLM_KV(llm_arch_from_string(arch_name));
     }
 
+
+    // DASLab GSQ keeps the hash constants as tensors and the two tables in a sidecar.
+    const int gsq_file_key = gguf_find_key(metadata, "deepseek41.engram.file");
+    if (gsq_file_key >= 0) {
+        if (arch_name != "deepseek41" || fname.empty() ||
+                gguf_get_kv_type(metadata, gsq_file_key) != GGUF_TYPE_STRING ||
+                std::string(gguf_get_val_str(metadata, gsq_file_key)) != "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf") {
+            throw std::runtime_error("unsupported GSQ external Engram contract");
+        }
+        auto scalar_alias = [&](const char * from, const char * to) {
+            uint32_t value = 0;
+            const int key = gguf_find_key(metadata, from);
+            if (key < 0 || gguf_get_kv_type(metadata, key) != GGUF_TYPE_UINT32 || gguf_find_key(metadata, to) >= 0) {
+                throw std::runtime_error(format("missing or ambiguous GSQ metadata: %s", from));
+            }
+            value = gguf_get_val_u32(metadata, key);
+            gguf_set_val_u32(metadata, to, value);
+        };
+        scalar_alias("deepseek41.engram_n_heads", "deepseek41.engram.head_count");
+        scalar_alias("deepseek41.engram_head_dim", "deepseek41.engram.key_length");
+        scalar_alias("deepseek41.engram_max_ngram_size", "deepseek41.engram.max_ngram_size");
+        scalar_alias("deepseek41.engram_pad_token_id", "deepseek41.engram.pad_id");
+
+        auto hash_tensor = [&](const char * name, const char * key, ggml_type type, gguf_type meta_type) {
+            auto it = weights_map.find(name);
+            if (it == weights_map.end() || gguf_find_key(metadata, key) >= 0) {
+                throw std::runtime_error(format("missing or ambiguous GSQ hash tensor: %s", name));
+            }
+            const auto & weight = it->second;
+            const size_t size = ggml_nbytes(weight.tensor);
+            if (weight.tensor->type != type || !ggml_is_contiguous(weight.tensor) || size == 0 || size > 4*MiB) {
+                throw std::runtime_error(format("invalid GSQ hash tensor: %s", name));
+            }
+            std::vector<uint64_t> data((size + 7)/8);
+            auto & input = files.at(weight.idx);
+            input->seek(weight.offs, SEEK_SET);
+            input->read_raw(data.data(), size);
+            gguf_set_arr_data(metadata, key, meta_type, data.data(), ggml_nelements(weight.tensor));
+            n_bytes -= size;
+            n_elements -= ggml_nelements(weight.tensor);
+            weights_map.erase(it);
+        };
+        hash_tensor("engram_token_map.weight", "deepseek41.engram.token_map", GGML_TYPE_I32, GGUF_TYPE_INT32);
+        hash_tensor("engram_primes.weight", "deepseek41.engram.primes", GGML_TYPE_I64, GGUF_TYPE_UINT64);
+        hash_tensor("engram_offsets.weight", "deepseek41.engram.offsets", GGML_TYPE_I64, GGUF_TYPE_UINT64);
+        hash_tensor("engram_multipliers.weight", "deepseek41.engram.multipliers", GGML_TYPE_I64, GGUF_TYPE_UINT64);
+
+        const int source_key = gguf_find_key(metadata, "deepseek41.engram.source");
+        if (source_key < 0 || gguf_get_kv_type(metadata, source_key) != GGUF_TYPE_STRING) {
+            throw std::runtime_error("GSQ Engram source identity is missing");
+        }
+        const std::string source_identity = gguf_get_val_str(metadata, source_key);
+        const size_t slash = fname.find_last_of("/\\");
+        const std::string path = (slash == std::string::npos ? "" : fname.substr(0, slash + 1)) +
+            "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf";
+        ggml_context * ctx = nullptr;
+        const gguf_init_params params = { true, &ctx };
+        gguf_context_ptr sidecar { gguf_init_from_file(path.c_str(), params) };
+        ggml_context_ptr owned_ctx { ctx };
+        if (!sidecar || !owned_ctx) {
+            throw std::runtime_error("GSQ external Engram could not be read");
+        }
+        const int side_arch = gguf_find_key(sidecar.get(), "general.architecture");
+        const int side_source = gguf_find_key(sidecar.get(), "deepseek41.engram.source");
+        if (side_arch < 0 || side_source < 0 || gguf_get_kv_type(sidecar.get(), side_arch) != GGUF_TYPE_STRING ||
+                gguf_get_kv_type(sidecar.get(), side_source) != GGUF_TYPE_STRING ||
+                std::string(gguf_get_val_str(sidecar.get(), side_arch)) != "deepseek41-engram" ||
+                std::string(gguf_get_val_str(sidecar.get(), side_source)) != source_identity ||
+                gguf_get_n_tensors(sidecar.get()) != 2) {
+            throw std::runtime_error("GSQ external Engram architecture, source or tensor count differs");
+        }
+        std::vector<uint32_t> layer_ids;
+        std::vector<uint32_t> table_rows;
+        get_arr("deepseek41.engram.layer_ids", layer_ids);
+        get_arr("deepseek41.engram.num_embeddings", table_rows);
+        uint32_t head_dim = 0;
+        get_key("deepseek41.engram.key_length", head_dim);
+        if (layer_ids.size() != 2 || table_rows.size() != 2 || files.size() >= UINT16_MAX) {
+            throw std::runtime_error("GSQ external Engram layer layout differs");
+        }
+        const uint16_t file_index = (uint16_t) files.size();
+        files.emplace_back(new llama_file(path.c_str(), "rb", use_direct_io));
+        for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
+            const std::string name = cur->name;
+            size_t layer = 0;
+            while (layer < layer_ids.size() && name != "blk." + std::to_string(layer_ids[layer]) + ".engram_embd.weight") {
+                ++layer;
+            }
+            if (layer == layer_ids.size() || cur->type != GGML_TYPE_IQ4_NL || cur->ne[0] != head_dim ||
+                    cur->ne[1] != (int64_t) table_rows[layer] || cur->ne[2] != 1 || cur->ne[3] != 1 ||
+                    weights_map.count(name) != 0) {
+                throw std::runtime_error(format("invalid or duplicate GSQ Engram table: %s", name.c_str()));
+            }
+            weights_map.emplace(name, llama_tensor_weight(files.back().get(), file_index, sidecar.get(), cur));
+            n_elements += ggml_nelements(cur);
+            n_bytes += ggml_nbytes(cur);
+        }
+        contexts.emplace_back(std::move(owned_ctx));
+        LLAMA_LOG_INFO("%s: GSQ hash tensors imported; external IQ4_NL Engram indexed without copying\n", __func__);
+    }
+
     n_kv      = gguf_get_n_kv(metadata);
     n_tensors = weights_map.size();
 
'''
GSQ_BLOCK = ('GSQ-RCO under a legacy Unsloth profile is BLOCKED: PR #30817 does not load '
             'this external IQ4_NL Engram layout. Use a dedicated --profile deepseek-gsq-* '
             'to select the pinned smalinin runtime plus embedded GSQ loader adapter. '
             'The original Unsloth model/profile will not be silently replaced.')
GSQ_RUNTIME_AUDIT = {
    'checked_utc_date': '2026-10-24',
    'ggml-org/llama.cpp:PR30817': DEEPSEEK_RUNTIME,
    'smalinin/llama.cpp:my_build': '0062e5930edf2f50c50254ba5080a6d37b024c6c',
    'vcruz305/llama.cpp:runtime/deepseek41': '5210c7c5ed61dddaee6ed476623abf4b63093d16',
    'ikawrakow/ik_llama.cpp:main': '804ee29c28a5e892df6e84941cb78628ed4c266c',
    'Niko1221/Strata:main': '7ddc17a38fe7f31637e895ecfba58c2cd43d65ec',
    'Niko1221/Strata:rc/0.1.30': '8099fdb98095e10984a468ed20864ad324161e7f',
    'gopinath87607/Strata-DeepSeek:main': '2fafe8418c0ac532c8063be4c577ddcd250656f5',
    'mw00/project-maya:ds4-v4.1': '1b8f5ec2bc4d1e4ba18c3d538fa5d44470c7296c',
    'exact_gsq_loader_verified': False,
    'metadata_mismatches': ['external deepseek41-engram GGUF',
                            'engram_token_map/primes/offsets/multipliers I32/I64 tensors',
                            'candidate attention source-layer metadata'],
}
IQ2_ENGRAM_BYTES = 135_200_000_000
# Exact Unsloth 1017 Q6_K Engram shards, same tensor bytes but different GGUF split headers.
# These digests are published for the pinned DS_REV, not unverified model filenames.
Q6_ENGRAM_SHARDS = {
    'UD-IQ3_XXS': ('DeepSeek-V4.1-Flash-1017-UD-IQ3_XXS-00007-of-00007.gguf',
                   '9d21807ac1a40cd1635634c14135b5acca1c8b855dae228170361602c7de560b', 7),
    'UD-IQ3_S': ('DeepSeek-V4.1-Flash-1017-UD-IQ3_S-00008-of-00008.gguf',
                 'e70b4bf3ac5b26aa2addb048cff61a6be5490e99ba88cfbc29f7fb7279000141', 8),
}  # (basename, expected SHA-256, shard index == shard count)

IQ2_CPU_EXPERT_BYTES_PER_LAYER = 3_000_000_000  # conservative budget, NOT measured
FP4_BLOCK = (
    'NVIDIA Qwen3.8-Flash-Next-FP4-Local remains BLOCKED. NVIDIA still says its '
    'compressed NVFP4 scales require a community vLLM container. No verified, '
    'unpatched vllm/vllm-openai release is enabled here. No community image will '
    'be pulled and no substitute model will start. To deliberately run Qwen27B '
    'instead, use --profile qwen.'
)

class Error(RuntimeError):
    pass

CONTEXT_PRESETS = {'256K': 262144, '512K': 524288,
                   '768K': 786432, '1M': 1048576}


def parse_context(value: str) -> int:
    """Parse tokens directly, including exact binary 256K, 512K, 768K, 1M."""
    text = str(value).strip().upper()
    if text in CONTEXT_PRESETS:
        return CONTEXT_PRESETS[text]
    if re.fullmatch(r'[0-9]+', text):
        return int(text)
    # Other exact binary K/M values are useful for experimenting with limits.
    match = re.fullmatch(r'([1-9][0-9]*)([KM])', text)
    if match:
        return int(match.group(1)) * (1024 if match.group(2) == 'K' else 1048576)
    raise argparse.ArgumentTypeError('context must be a token count or 256K, 512K, 768K, 1M')

LOG = None

def say(text: str) -> None:
    line = '[start-remote-model] ' + text
    print(line, flush=True)
    if LOG is not None:
        LOG.write(line + '\n')
        LOG.flush()

def run(args: list[str], *, env=None, capture=False, timeout=None, quiet=False) -> str:
    if not quiet:
        # No secrets are passed on command lines by this launcher.
        say('$ ' + shlex.join([str(a) for a in args]))
    result = subprocess.run([str(a) for a in args], env=env, text=True,
                            capture_output=capture, timeout=timeout)
    if result.returncode:
        if capture and result.stderr:
            print(result.stderr, file=sys.stderr)
        raise Error(f'{args[0]} exited {result.returncode}; no automatic model fallback.')
    return result.stdout if capture else ''

def envint(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as e:
        raise Error(f'{name} must be an integer') from e

def enabled(*names: str) -> bool:
    return any(os.environ.get(n, '').lower() in ('1', 'yes', 'true') for n in names)

def atomic_json(path: Path, obj: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(obj, out, indent=2)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

@dataclasses.dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    total: int  # MiB, actual nvidia-smi value (not marketing capacity)
    free: int   # MiB, includes effects of existing GPU users
    cc: int = 0

def parse_inventory(text: str) -> list[GPU]:
    result = []
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        if len(row) != 5:
            raise Error('Unexpected nvidia-smi inventory format')
        i, ident, name, total, free = [x.strip() for x in row]
        if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', ident):
            raise Error('Expected full physical GPU UUIDs; MIG/vGPU is not supported by these profiles')
        result.append(GPU(int(i), ident.lower(), name, int(float(total)), int(float(free))))
    return result

def inventory() -> list[GPU]:
    if not shutil.which('nvidia-smi'):
        raise Error('nvidia-smi missing; use a GPU-enabled CUDA development pod')
    text = run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free',
                '--format=csv,noheader,nounits'], capture=True, quiet=True, timeout=30)
    return parse_inventory(text)

def cuda_inventory() -> list[tuple[str, int]]:
    """CUDA enumeration, unlike counting nvidia-smi rows, respects process visibility."""
    try:
        cuda = ctypes.CDLL('libcuda.so.1')
    except OSError as e:
        raise Error('Cannot load the NVIDIA CUDA driver (libcuda.so.1)') from e
    def check(code, name):
        if code:
            raise Error(f'{name} failed with CUDA status {code}')
    check(cuda.cuInit(0), 'cuInit')
    count = ctypes.c_int()
    check(cuda.cuDeviceGetCount(ctypes.byref(count)), 'cuDeviceGetCount')
    get_uuid = getattr(cuda, 'cuDeviceGetUuid_v2', None) or cuda.cuDeviceGetUuid
    result = []
    for i in range(count.value):
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        raw = (ctypes.c_ubyte * 16)()
        check(cuda.cuDeviceGet(ctypes.byref(device), i), 'cuDeviceGet')
        check(get_uuid(ctypes.byref(raw), device), 'cuDeviceGetUuid')
        check(cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, device), 'compute capability major')
        check(cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, device), 'compute capability minor')
        result.append(('gpu-' + str(uuid.UUID(bytes=bytes(raw))), major.value * 10 + minor.value))
    return result

def visible_gpus(all_gpus: list[GPU], visible: list[tuple[str, int]]) -> list[GPU]:
    lookup = {g.uuid: g for g in all_gpus}
    if len(set(i for i, _ in visible)) != len(visible):
        raise Error('Duplicate CUDA device identities')
    if any(i not in lookup for i, _ in visible):
        raise Error('CUDA/NVML UUID mismatch, possibly MIG; refusing to guess the allocation')
    return [dataclasses.replace(lookup[i], cc=cc) for i, cc in visible]

def is_a40(g: GPU) -> bool:
    return bool(re.fullmatch(r'(?:NVIDIA\s+)?A40', g.name, re.I)) and g.total >= 44000

def is_a6000(g: GPU) -> bool:
    # Explicitly exclude RTX 6000 Ada and RTX PRO 6000 Blackwell.
    return bool(re.fullmatch(r'(?:NVIDIA\s+)?RTX\s+A6000', g.name, re.I)) and g.total >= 44000

def is_ampere_48(g: GPU) -> bool:
    return is_a40(g) or is_a6000(g)

def flint_eligible(g: GPU) -> bool:
    # Discrete CUDA cards with at least 32GB; no claims of NVFP4 hardware acceleration on Ampere.
    return g.cc >= 80 and g.total >= 30000

def is_h200(g: GPU) -> bool:
    # Covers NVIDIA H200, H200 SXM, and explicitly named H200 NVL; a physical
    # ~141 GB card is required. Other Hopper GPUs are not silently selected.
    return (bool(re.search(r'\bH200\b', g.name, re.I))
            and g.total >= 130000 and g.cc == 90)

def is_b200(g: GPU) -> bool:
    # NVIDIA B200 SXM: compute capability 10.0, typically 183359 MiB shown by
    # nvidia-smi. Do not accidentally select GB200, B300 or a MIG slice.
    return (bool(re.match(r'^(?:NVIDIA\s+)?B200\b', g.name, re.I))
            and g.cc == 100 and g.total >= 170000)

def is_b300(g: GPU) -> bool:
    # A discrete B300 Blackwell Ultra, not a B200, GB300 or an RTX PRO 6000.
    # CC 10.3 is required for native CUDA compilation; 250,000 MiB guards
    # against undersized/MIG instances rather than trusting the name alone.
    return (bool(re.match(r'^(?:NVIDIA\s+)?B300\b', g.name, re.I))
            and g.cc == 103 and g.total >= 250000)

def pro6000(g: GPU) -> bool:
    return ('RTX PRO 6000' in g.name.upper() and 'BLACKWELL' in g.name.upper()
            and g.total >= 90000 and g.cc == 120)

def qwen_tier(g: GPU) -> str | None:
    name = g.name.upper()
    if '4090' in name and g.total >= 22000:
        return '4090'
    if '5090' in name and g.total >= 30000:
        return '5090'
    if is_ampere_48(g) or (
        re.search(r'\bL40S?\b', name) and g.total >= 44000):
        return '48gb'
    if 'RTX 6000 ADA' in name and g.total >= 44000:
        return '48gb'
    if pro6000(g):
        return '96gb'
    return None

def select(gpus: list[GPU], mode: str, gpu_count: int | None) -> tuple[str, list[GPU]]:
    if not gpus:
        raise Error('No CUDA-visible GPUs')
    ampere48 = [g for g in gpus if is_ampere_48(g)]
    pro_blackwell = [g for g in gpus if pro6000(g)]
    if mode == 'auto':
        if len(ampere48) >= 8:
            mode = 'deepseek'
        elif len(ampere48) == 2:
            mode = 'ornith'
        elif len(pro_blackwell) >= 4:
            mode = 'deepseek-pro6000-quad-q8'
        elif len(gpus) == 1 and is_b300(gpus[0]):
            mode = 'deepseek-b300-iq3'
        elif len(gpus) == 1 and is_b200(gpus[0]):
            mode = 'deepseek-b200-iq2'
        elif len(gpus) == 2 and all(is_h200(g) for g in gpus):
            mode = 'deepseek-h200-dual-iq3'
        elif len(gpus) == 1 and is_h200(gpus[0]):
            mode = 'deepseek-h200-iq2'
        elif any(pro6000(g) for g in gpus):
            mode = 'flash-next-fp4'
        elif (len(gpus) == 1 and flint_eligible(gpus[0])
              and qwen_tier(gpus[0]) in ('5090', '48gb')):
            mode = 'flint'
        elif 3 <= len(ampere48) <= 7 and len(gpus) == len(ampere48):
            # Smaller A40/A6000 pods previously selected Qwen on just one GPU.
            mode = 'flint'
        else:
            mode = 'qwen'
    if mode in GSQ_PROFILES:
        info=GSQ_PROFILES[mode];need=info['count']
        valid={'a40':lambda g:is_a40(g) and g.cc==86,'ampere48':lambda g:is_ampere_48(g) and g.cc==86,
               'pro6000':pro6000,'h200':is_h200,'b200':is_b200,'b300':is_b300}[info['kind']]
        count=len(gpus)
        if not all(valid(g) for g in gpus) or gpu_count not in (None,count) or (count!=need if need else not 8<=count<=40):
            raise Error(f'{mode} requires '+(str(need) if need else '8..40')+
                        ' CUDA-visible full-memory '+info['kind']+' GPUs on one host; no MIG/mixed hardware or implicit selection')
        return mode,gpus
    if mode.startswith('minimax-'):
        counts = {'minimax-pro6000-dual-q2k': 2, 'minimax-a40-quad-q2k': 4,
                  'minimax-pro6000-single-q2k': 1, 'minimax-a40-dual-q2k': 2,
                  'minimax-a40-triple-q2k': 3, 'minimax-h200-single-q2k': 1,
                  'minimax-h200-dual-q2k': 2}
        need = counts.get(mode)
        valid = (all(pro6000(g) for g in gpus) if 'pro6000' in mode
                 else all(is_h200(g) for g in gpus) if 'h200' in mode
                 else all(is_a40(g) and g.cc == 86 for g in gpus))
        if need is None or len(gpus) != need or gpu_count not in (None, need) or not valid:
            raise Error(f'{mode} requires exactly {need} CUDA-visible full-memory '
                        + ('RTX PRO 6000 Blackwell 96GB (sm_120)' if 'pro6000' in mode
                           else 'H200 141GB (sm_90)' if 'h200' in mode
                           else 'A40 48GB (sm_86)')
                        + ' on one host. Restrict CUDA_VISIBLE_DEVICES explicitly; Ada/A6000/MIG are excluded.')
        return mode, gpus  # CUDA order is also used by tensor-split
    if mode == 'flash-next-fp4':
        return mode, pro_blackwell
    if mode == 'deepseek-pro6000-quad-q8':
        if gpu_count not in (None, 4) or len(pro_blackwell) < 4:
            raise Error('Quad RTX PRO 6000 DeepSeek requires four physical, full-memory (96 GB) '
                        'Blackwell RTX PRO 6000 GPUs (sm_120) on one host. RTX 6000 Ada, '
                        'RTX A6000, MIG/vGPU and smaller allocations are excluded.')
        if len(pro_blackwell) > 4 and gpu_count is None:
            raise Error('More than four RTX PRO 6000 cards visible. Pass --gpu-count 4 '
                        'or restrict CUDA_VISIBLE_DEVICES to select four deliberately.')
        return mode, sorted(pro_blackwell, key=lambda g: g.index)[:4]
    if mode == 'deepseek-b200-iq2':
        if gpu_count not in (None, 1) or len(gpus) != 1 or not is_b200(gpus[0]):
            raise Error('The B200 IQ2 profile requires exactly one CUDA-visible, full-memory '
                        'NVIDIA B200 (sm_100); GB200/B300/MIG are not accepted.')
        return mode, gpus
    if mode == 'deepseek-b300-iq3':
        if gpu_count not in (None, 1) or len(gpus) != 1 or not is_b300(gpus[0]):
            raise Error('The experimental B300 IQ3 profile requires exactly one '
                        'CUDA-visible, full-memory NVIDIA B300 (sm_103). '
                        'Use CUDA_VISIBLE_DEVICES if other GPUs are present.')
        return mode, gpus
    if mode == 'deepseek-h200-dual-iq3':
        eligible = [g for g in gpus if is_h200(g)]
        if gpu_count not in (None, 2) or len(eligible) != 2:
            raise Error('The H200 dual-IQ3 profile requires exactly two CUDA-visible full-memory '
                        'H200 SXM GPUs (141 GB each), on the same host. Use CUDA_VISIBLE_DEVICES '
                        'if other GPUs are present; separate pods cannot be pooled.')
        return mode, sorted(eligible, key=lambda g: g.index)
    if mode == 'deepseek-h200-iq2':
        eligible = [g for g in gpus if is_h200(g)]
        if gpu_count not in (None, 1) or len(eligible) != 1:
            raise Error('The experimental H200 IQ2 profile requires exactly one CUDA-visible H200; '
                        'multi-H200 serving is not configured or validated.')
        return mode, eligible
    if mode == 'flint':
        if gpu_count not in (None, 1):
            raise Error('Flint profile uses one GPU/replica; --gpu-count must be 1')
        eligible = [g for g in gpus if flint_eligible(g)]
        if not eligible:
            raise Error('Flint NVFP4 requires a CUDA-visible GPU with >=30,000 MiB VRAM; '
                        'no automatic CPU offload or pooling of smaller cards')
        picked = max(eligible, key=lambda g: (g.free, g.total, -g.index))
        return 'flint', [picked]
    if mode in ('ornith', 'deepseek'):
        need = 2 if mode == 'ornith' else 8
        use = gpu_count if gpu_count is not None else (2 if mode == 'ornith' else len(ampere48))
        if (mode == 'ornith' and use != 2) or use < need or use > len(ampere48):
            raise Error(f'{mode} requires {need if mode == "ornith" else "at least 8"} local A40/RTX A6000 cards; '
                        f'visible supported cards={len(ampere48)}, requested={use}. Separate pods cannot be pooled.')
        picked = sorted(ampere48, key=lambda g: (-g.free, g.index))[:use]
        return mode, sorted(picked, key=lambda g: g.index)
    if mode != 'qwen':
        raise Error('Unknown profile ' + mode)
    if gpu_count not in (None, 1):
        raise Error('The Qwen27B profile uses one GPU; no implicit multi-GPU sharding')
    candidates = [g for g in gpus if qwen_tier(g)]
    if not candidates:
        raise Error('No supported rental-GPU profile. Your 12 GB 4070 Super should keep its '
                    'measured Bonsai localmodel presets; this script does not replace them.')
    return 'qwen', [max(candidates, key=lambda g: (g.free, g.total, -g.index))]

@dataclasses.dataclass
class Plan:
    profile: str
    model_id: str
    repo: str
    revision: str
    quant: str
    context: int
    kv: str
    spec: str
    depth: int
    runtime: str
    runtime_ref: str
    weight_bytes: int
    table_bytes: int
    draft_bytes: int
    reserve_bytes: int
    root: Path
    gpus: list[GPU]
    effort: str
    cache_ram: int
    port: int
    ubatch: int
    batch: int
    model_dir: Path | None = None
    cpu_moe_layers: int = 0
    engine: str = 'llama.cpp'
    minimax_split: str | None = None
    gsq_split: str | None = None
    gsq_engram_device: str | None = None

    def public(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out['root'] = str(self.root)
        out['model_dir'] = str(self.model_dir) if self.model_dir else None
        out['endpoint'] = f'http://127.0.0.1:{self.port}/v1'
        out['experimental_runtime'] = self.profile.startswith(('deepseek', 'minimax-'))
        if self.engine == 'strata':
            for key in ('kv', 'cache_ram', 'batch', 'ubatch', 'cpu_moe_layers', 'gsq_split', 'gsq_engram_device', 'minimax_split', 'reserve_bytes'):
                out.pop(key, None)
            out.update(experimental_runtime=False, memory_estimates_not_gpu_validation=True,
                       runtime_version='0.1.52', control_version='0.1.53',
                       native_template_and_parser='deepseek41', hard_ram_limit=None)
            if self.spec == 'dspark':
                out.update(draft_source=STRATA_DSPARK, dspark_block_limit=self.depth)
            return out
        if self.profile.startswith('minimax-'):
            out.update(runtime_patch_sha256=MINIMAX_PATCH_SHA256,
                       source_revision=MINIMAX_SOURCE_REV, architecture='minimaxm31_sakura',
                       sparse_attention_layers=60, modalities='text-only',
                       host_ram_floor_bytes=minimax_host_requirement(self),
                       cpu_routed_expert_bytes=self.cpu_moe_layers * MINIMAX_EXPERT_BYTES,
                       gpu_budgets_bytes=minimax_gpu_budgets(self), kv_bytes=kv_bytes(self))
            if self.spec == 'dflash':
                out['draft_source'] = dict(repo=MINIMAX_DFLASH_REPO, revision=MINIMAX_DFLASH_REV,
                                           output='local F16 GGUF conversion', gpu='CUDA0',
                                           full_model_conversion_and_gpu_inference_tested=False)
        else:
            out.pop('minimax_split', None)
        if self.profile in GSQ_PROFILES:
            out.update(source='gsq-rco',runtime_source=GSQ_RUNTIME_SOURCE,runtime_patch_sha256=GSQ_PATCH_SHA256,
                       architecture='deepseek41',expert_layers=40,experts_per_layer=384,active_experts=6,
                       gpu_budgets_bytes=gsq_gpu_budgets(self),host_ram_floor_bytes=gsq_host_requirement(self),
                       cpu_routed_expert_bytes=gsq_cpu_expert_bytes(self),
                       engram_placement=self.gsq_engram_device,engram_bytes=GSQ_ENGRAM_TENSOR_BYTES,
                       prompt_prefix_cache_default=False,gpu_inference_tested=False,
                       pinned_files=gsq_files(self.quant))
        else:
            out.pop('gsq_split',None);out.pop('gsq_engram_device',None)
        if self.profile.startswith('deepseek') and self.quant == 'UD-Q2_K_XL':
            out['prompt_prefix_cache_default'] = False
            out['client_cache_prompt_override_policy'] = 'keep false; pinned runtime Engram prefix-reuse issue #30944'
        out['memory_estimates_not_gpu_validation'] = True
        return out

def make_plan(args, mode: str, gpus: list[GPU]) -> Plan:
    if mode in GSQ_PROFILES:
        return make_gsq_plan(args,mode,gpus)
    if mode == 'flash-next-fp4':
        raise Error(FP4_BLOCK)
    base = Path(args.root).expanduser().absolute()
    prefix = {'qwen': 'QWEN', 'ornith': 'ORNITH', 'flint': 'FLINT', 'deepseek': 'DS', 'deepseek-h200-iq2': 'DS_H200',
              'deepseek-h200-dual-iq3': 'DS_H200_DUAL', 'deepseek-b300-iq3': 'DS_B300',
              'deepseek-pro6000-quad-q8': 'DS_PRO6000_QUAD',
              'deepseek-b200-iq2': 'DS_B200', **{m: 'MINIMAX' for m in MINIMAX_PROFILES}}[mode]
    if mode == 'qwen':
        tier = qwen_tier(gpus[0])
        quant = 'Q4_K_M' if tier in ('4090', '5090') else 'Q8_0'
        context = 131072 if tier == '4090' else 262144
        kv = 'q4_0' if tier in ('4090', '5090') else 'q8_0'
        spec, depth, effort = 'mtp', 3, 'xhigh'
        ident, repo, revision = 'qwen3.8-27b', QWEN_REPO, QWEN_REV
        runtime, runtime_ref = STABLE, 'refs/tags/v0.7.0'
        table, reserve = 0, int((1.5 if tier == '4090' else 3.0) * GIB)
    elif mode == 'flint':
        # Publisher's GGUF is quantization-aware trained NVFP4, ~22.9 GB.
        # Full documented native ctx=262,144. On 5090/32GB, f16 KV+buffers
        # makes full 262K tight; 192K is the safer default without unvalidated KV quant.
        small = gpus[0].total < 39000
        quant, context, kv = 'NVFP4', (196608 if small else 262144), 'f16'
        spec, depth, effort = 'none', 0, 'default'
        ident, repo, revision = 'recursive-flint-nvfp4', FLINT_REPO, os.environ.get('FLINT_REVISION', 'main')
        runtime, runtime_ref = FLINT_RUNTIME, 'upstream-merged-PR-30928'
        table, reserve = 0, (3 if small else 4) * GIB
    elif mode.startswith('minimax-'):
        context = (262144 if mode == 'minimax-h200-dual-q2k' else 16384
                   if mode in ('minimax-a40-quad-q2k', 'minimax-a40-triple-q2k') else 32768)
        quant, kv = 'Q2_K', 'f16'
        spec, depth, effort = 'none', 0, 'high'
        ident, repo, revision = 'minimax-m31-flash-' + mode.removeprefix('minimax-'), MINIMAX_REPO, MINIMAX_REV
        runtime, runtime_ref = MINIMAX_RUNTIME, 'upstream-commit+Sakura-all-sparse+DFlash-target-features'
        table, reserve = 0, len(gpus) * 6 * GIB
    elif mode == 'ornith':
        quant, context, kv = 'UD-Q4_K_XL', 262144, 'f16'
        spec, depth, effort = 'none', 0, 'default'
        ident, repo, revision = 'ornith-1.5-122b', ORNITH_REPO, ORNITH_REV
        runtime, runtime_ref = STABLE, 'refs/tags/v0.7.0'
        table, reserve = 0, 6 * GIB
    elif mode == 'deepseek-pro6000-quad-q8':
        # Unsloth 1017 UD-Q8_K_XL: ~305.6 GB of computational weights,
        # ~202.8 GB of Engram in host RAM/SSD. Q8 retains native MXFP4 experts,
        # BF16 non-expert weights. Four 96GB SM120 GPUs via layer splitting.
        # 8 GiB per GPU reserve is only a planning allowance, not a measured peak.
        quant, context, kv = 'UD-Q8_K_XL', 262144, 'f16'
        spec, depth, effort = 'none', 0, '100'
        ident, repo, revision = 'deepseek-v4.1-1017-quad-pro6000-q8', DS_REPO, DS_REV
        runtime, runtime_ref = DEEPSEEK_RUNTIME, 'refs/pull/30817/head'
        table, reserve = 202_800_000_000, len(gpus) * 8 * GIB
    elif mode == 'deepseek-b200-iq2':
        # B200 full card: ~179 GiB reported by nvidia-smi. Unsloth's 2-bit
        # variant has ~165.3 decimal GB compute weights (153.96 GiB), with
        # a ~135.2 GB Q5 Engram table mapped to CPU/SSD. Actual context and
        # temporary CUDA allocations remain untested: fail closed on OOM.
        # UD-IQ1_M fits much more easily but sacrifices benchmark fidelity.
        quant, context, kv = 'UD-IQ2_XXS', 262144, 'f16'
        spec, depth, effort = 'none', 0, '100'
        ident, repo, revision = 'deepseek-v4.1-1017-b200-iq2', DS_REPO, DS_REV
        runtime, runtime_ref = DEEPSEEK_RUNTIME, 'refs/pull/30817/head'
        table, reserve = 135_200_000_000, 11 * GIB
    elif mode == 'deepseek-b300-iq3':
        # Unsloth IQ3_XXS: 394.7 GB total, including 161.3 GB Q6_K Engram.
        # The backbone (~233.5 GB) stays in HBM. The table stays CPU/mmap.
        quant, context, kv = 'UD-IQ3_XXS', 1048576, 'f16'
        spec, depth, effort = 'none', 0, '100'
        ident, repo, revision = 'deepseek-v4.1-flash-1017-b300-iq3', DS_REPO, DS_REV
        runtime, runtime_ref = DEEPSEEK_RUNTIME, 'refs/pull/30817/head'
        # At 512K the user's full GPU allocation was 226072 MiB for IQ3_XXS,
        # versus ~217.5 GiB of compute weights. A fixed 8 GiB reserve plus
        # a separate bounded cache allowance more closely follows actual
        # B300 compressed-attention allocations than 8 GiB per 256K.
        # This is a conservative *planning* adjustment, not an exact guarantee.
        table, reserve = 161_300_000_000, 8 * GIB
    elif mode == 'deepseek-h200-dual-iq3':
        # Unsloth IQ3_XXS: 394.7 GB overall, including 161.3 GB CPU Engram.
        # A quality-first UD-IQ3_S (423.6 GB overall) is optional at shorter context.
        # Keep all transformer layers and their FP16 KV on the 2 H200 GPUs.
        quant, context, kv = 'UD-IQ3_XXS', 262144, 'f16'
        spec, depth, effort = 'none', 0, '100'
        ident, repo, revision = 'deepseek-v4.1-1017-dual-h200-iq3', DS_REPO, DS_REV
        runtime, runtime_ref = DEEPSEEK_RUNTIME, 'refs/pull/30817/head'
        table, reserve = 161_300_000_000, 2 * 8 * GIB
    elif mode == 'deepseek-h200-iq2':
        quant, context, kv = 'UD-IQ2_XXS', 262144, 'f16'
        spec, depth, effort = 'none', 0, '100'
        ident, repo, revision = 'deepseek-v4.1-1017-h200-iq2', DS_REPO, DS_REV
        runtime, runtime_ref = DEEPSEEK_RUNTIME, 'refs/pull/30817/head'
        table, reserve = IQ2_ENGRAM_BYTES, 9 * GIB
    else:
        quant, context, kv = 'UD-Q8_K_XL', 262144, 'f16'
        spec, depth, effort = 'none', 0, '100'
        ident, repo, revision = 'deepseek-v4.1-flash-1017', DS_REPO, DS_REV
        runtime, runtime_ref = DEEPSEEK_RUNTIME, 'refs/pull/30817/head'
        table, reserve = 202_800_000_000, len(gpus) * 4 * GIB
    if args.variant is not None:
        if mode not in ('deepseek-b300-iq3', 'deepseek-h200-dual-iq3'):
            raise Error('--variant iq3_s/iq3_xxs is only supported on B300 and dual H200 profiles')
        selected_variant = {'iq3_s': 'UD-IQ3_S', 'iq3_xxs': 'UD-IQ3_XXS'}[args.variant]
        if args.quant is not None and args.quant != selected_variant:
            raise Error('--variant and --quant disagree; choose one quantization')
        quant = selected_variant
    else:
        quant = args.quant or os.environ.get(prefix + '_QUANT', quant)
    if mode == 'deepseek-b200-iq2' and quant == 'UD-IQ1_M':
        # Unsloth's 1-bit fallback: ~139.8 GB compute weights, 110.6 GB Engram.
        table = 110_600_000_000
    context = args.context if args.context is not None else parse_context(os.environ.get(prefix + '_CONTEXT', str(context)))
    kv = args.kv or os.environ.get(prefix + '_KV', kv)
    spec = args.spec or os.environ.get(prefix + '_SPEC', spec)
    if args.dspark is not None:
        if not mode.startswith('deepseek'):
            raise Error('--dspark is only supported on DeepSeek 1017 profiles')
        if args.spec and args.spec not in ('none', 'dspark'):
            raise Error('--dspark conflicts with --spec; choose one speculative decoder')
        if args.draft_depth is not None:
            raise Error('--dspark N already specifies depth; do not also pass --draft-depth')
        if args.dspark < 0 or args.dspark > 5:
            raise Error('--dspark must be 0..5; 0 disables it, 3 recommended')
        spec = 'dspark' if args.dspark else 'none'
        depth = args.dspark
    else:
        depth = args.draft_depth if args.draft_depth is not None else envint(prefix + '_DRAFT_DEPTH', depth)
    if args.no_spec:
        if ((args.spec is not None and args.spec != 'none')
                or (args.dspark is not None and args.dspark > 0)):
            raise Error('--no-mtp/--no-dflash conflicts with explicitly enabled speculation')
        spec = 'none'
    effort = args.reasoning_effort or os.environ.get(prefix + '_REASONING_EFFORT', effort)
    maximum = 1048576 if mode.startswith(('deepseek', 'minimax-')) else 262144
    if not 4096 <= context <= maximum:
        raise Error(f'{mode}: context must be 4096..{maximum}; no automatic YaRN extension')
    allowed_quants = {'qwen': ('Q4_K_M', 'Q8_0'), 'flint': ('NVFP4',), 'ornith': ('UD-Q4_K_XL',),
                      'deepseek': ('UD-Q8_K_XL', 'UD-Q4_K_XL', 'UD-Q2_K_XL'),
                      'deepseek-h200-iq2': ('UD-IQ2_XXS',),
                      'deepseek-h200-dual-iq3': ('UD-IQ3_XXS', 'UD-IQ3_S', 'UD-Q2_K_XL'),
                      'deepseek-pro6000-quad-q8': ('UD-Q8_K_XL', 'UD-Q4_K_XL', 'UD-Q2_K_XL'),
                      'deepseek-b300-iq3': ('UD-IQ3_XXS', 'UD-IQ3_S', 'UD-Q2_K_XL'),
                      'deepseek-b200-iq2': ('UD-IQ2_XXS', 'UD-IQ1_M')}
    if mode.startswith('minimax-'):
        allowed_quants[mode] = ('Q2_K',)
        if kv != 'f16' or spec not in ('none', 'dflash'):
            raise Error('MiniMax requires f16 KV and --spec none/dflash; DSpark/MTP are not enabled')
        if spec == 'dflash':
            depth = 7 if depth == 0 else depth
            if not 1 <= depth <= 7:
                raise Error('MiniMax DFlash2 draft-depth must be 1..7 (trained block size 8)')
        elif args.draft_depth not in (None, 0):
            raise Error('MiniMax draft-depth requires --spec dflash')
        if effort not in ('low', 'medium', 'high', 'xhigh', 'max'):
            raise Error('MiniMax reasoning effort must be low/medium/high/xhigh/max')
    if quant not in allowed_quants[mode]:
        raise Error(f'Unsupported quant for {mode}: {quant}; allowed: {allowed_quants[mode]}')
    if kv not in ('f16', 'q8_0', 'q4_0') or (mode.startswith('deepseek') and kv != 'f16'):
        raise Error('Supported KV: f16/q8_0/q4_0; the DeepSeek baseline deliberately uses f16')
    if mode == 'flint' and gpus[0].total < 39000 and context > 196608 and kv == 'f16':
        raise Error('Flint on 32GB GPU beyond 192K with f16 KV leaves unsafe VRAM headroom; '
                    'try --ctxt 256K --kv q8_0 (unvalidated accuracy/throughput)')
    if ((mode == 'qwen' and spec not in ('none', 'mtp', 'dflash'))
            or (mode in ('ornith', 'flint') and spec != 'none')
            or (mode.startswith('deepseek') and spec not in ('none', 'dspark'))):
        raise Error('Qwen uses MTP/DFlash; DeepSeek 1017 uses DSpark; Flint/Ornith use ordinary decoding')
    if spec == 'dspark' and depth > 5:
        raise Error('DeepSeek 1017 DSpark supports at most 5 drafts; 3 recommended')
    if spec == 'dspark' and depth == 0:
        # --spec dspark can also be used directly, although --dspark N is preferred.
        depth = 3
    if spec != 'none' and not 1 <= depth <= 16:
        raise Error('Draft depth must be 1..16')
    if spec == 'none':
        depth = 0
    if mode == 'qwen' and effort not in ('xhigh', 'medium', 'low', 'default'):
        raise Error('Qwen3.8 effort must be xhigh, medium, low or default')
    if mode in ('ornith', 'flint') and effort != 'default':
        raise Error('Flint/Ornith have no validated reasoning-effort override; use the native defaults')
    if mode.startswith('deepseek'):
        effort = {'low':'50', 'medium':'62', 'high':'75', 'max':'100', 'xhigh':'100'}.get(effort, effort)
        if not effort.isdigit() or not 1 <= int(effort) <= 100:
            raise Error('DeepSeek reasoning effort must be 1..100 (or low/medium/high/max)')
    if mode.startswith('minimax-'):
        weights = MINIMAX_BYTES
    elif mode == 'qwen':
        weights = 19_000_000_000 if quant == 'Q4_K_M' else 28_600_000_000
    elif mode == 'ornith':
        weights = 79_700_000_000
    elif mode == 'flint':
        weights = FLINT_BYTES
    elif mode == 'deepseek-pro6000-quad-q8':
        weights = 508_400_000_000 if quant == 'UD-Q8_K_XL' else 500_700_000_000
    elif mode == 'deepseek-h200-dual-iq3':
        weights = 423_600_000_000 if quant == 'UD-IQ3_S' else 394_700_000_000
    elif mode == 'deepseek-h200-iq2':
        weights = 300_500_000_000
    elif mode == 'deepseek-b300-iq3':
        weights = 423_600_000_000 if quant == 'UD-IQ3_S' else 394_700_000_000
    elif mode == 'deepseek-b200-iq2':
        weights = 250_400_000_000 if quant == 'UD-IQ1_M' else 300_500_000_000
    else:
        weights = 508_400_000_000 if quant == 'UD-Q8_K_XL' else 500_700_000_000
    if mode.startswith('deepseek') and quant == 'UD-Q2_K_XL':
        weights, table = DS_Q2_BYTES, DS_Q2_ENGRAM_BYTES
        ident += '-q2k'  # separate manifest/model directory; never replaces an IQ3/Q8 cache
    draft = {'none': 0, 'mtp': 1_680_000_000, 'dflash': 1_090_000_000,
             'dspark': 7_980_000_000}[spec]
    if mode.startswith('minimax-') and spec == 'dflash':
        draft = MINIMAX_DFLASH_MAX_BYTES
    if mode == 'qwen' and spec == 'dflash':
        reserve += GIB  # a planning allowance, not a measured draft-state budget
    root = Path(os.environ.get(prefix + '_ROOT', str(base / ident))).expanduser().absolute() if mode != 'qwen' else base / ident
    model_dir = Path(args.model_dir).expanduser().absolute() if args.model_dir else None
    cpu_moe = (args.cpu_moe_layers if args.cpu_moe_layers is not None else 18) if mode == 'deepseek-h200-iq2' else 0
    if mode == 'deepseek-h200-iq2' and not 1 <= cpu_moe <= 40:
        raise Error('H200 IQ2 requires --cpu-moe-layers in 1..40 (default 18)')
    if mode.startswith('minimax-'):
        cpu_moe = args.cpu_moe_layers if args.cpu_moe_layers is not None else 0
        if not 0 <= cpu_moe <= 57:
            raise Error('MiniMax --cpu-moe-layers must be 0..57; counts routed-expert layers after the three dense MLPs')
        if args.ubatch > 128 or args.batch > 512:
            raise Error('MiniMax planning reserve only covers --ubatch <=128 and --batch <=512')
    elif args.cpu_moe_layers is not None and mode != 'deepseek-h200-iq2':
        raise Error('--cpu-moe-layers is only for H200 IQ2 or MiniMax profiles')
    plan = Plan(mode, ident, repo, revision, quant, context, kv, spec, depth, runtime,
                runtime_ref, weights, table, draft, reserve, root, gpus, effort,
                args.cache_ram, args.port, args.ubatch, args.batch, model_dir, cpu_moe,
                'llama.cpp')
    if mode.startswith('minimax-'):
        plan.minimax_split = args.tensor_split or minimax_balanced_split(plan)
        minimax_gpu_budgets(plan)
    return plan

def kv_bytes(plan: Plan) -> int:
    if plan.profile in GSQ_PROFILES:
        return len(plan.gpus)*(6*1024*((plan.context+255)//256*256)+256*MIB)
    if plan.profile.startswith('minimax-'):
        # The pinned index cache allocates BOTH K and V, though MSA reads only K.
        # Main: 60*2*4*128*2; index: 60*2*1*128*2 bytes/token.
        # Sparse selection does not shorten the retained KV cache.
        return 60 * (2 * 4 * 128 * 2 + 2 * 128 * 2) * ((plan.context + 255) // 256 * 256)
    if plan.profile == 'flint':
        # Official vLLM figure: 14 global-attention layers, 4 KV heads, head_dim 128.
        # FP16 cache = 14 * 2 * 4 * 128 * 2 = 28,672 bytes/token = 28 KiB.
        # SWA cache: 35 layers with 513-token window, kept compact by default.
        full_kv = 14 * 2 * 4 * 128 * plan.context
        window_kv = 35 * 2 * 4 * 128 * 513
        scale = {'f16': 2, 'q8_0': 34/32, 'q4_0': 18/32}[plan.kv]
        return math.ceil((full_kv + window_kv) * scale)
    if plan.profile == 'deepseek-b300-iq3' and plan.quant != 'UD-Q2_K_XL':
        # User observed 220.8 GiB total allocations on B300 with IQ3_XXS at
        # 512K, only ~3.3 GiB above its estimated GPU weights. We still budget
        # a 2-4 GiB cache allowance, plus the separate 8 GiB runtime reserve.
        # Memory growth with fully populated 1M prompts is NOT proven.
        return max(2, math.ceil(plan.context / 262144)) * GIB
    if plan.profile.startswith('deepseek'):
        # Other hardware retains the original conservative unmeasured budget.
        return math.ceil(plan.context / 262144) * 8 * GIB
    layers, heads = (12, 2) if plan.profile == 'ornith' else (16, 4)
    values = 2 * layers * heads * 256 * plan.context
    return {'f16': values * 2, 'q8_0': values // 32 * 34, 'q4_0': values // 32 * 18}[plan.kv]

def gpu_requirement(plan: Plan, weights: int | None = None, draft: int | None = None) -> int:
    if plan.profile in GSQ_PROFILES:
        if weights not in (None,plan.weight_bytes) or draft not in (None,0):
            raise Error('GSQ checkpoint/draft size differs from the exact pinned five-file layout')
        return sum(gsq_gpu_budgets(plan))
    if plan.profile.startswith('minimax-'):
        if weights not in (None, MINIMAX_BYTES) or (draft is not None and not 0 <= draft <= plan.draft_bytes):
            raise Error('MiniMax exact checkpoint size changed; refusing to extrapolate fit')
        return sum(minimax_gpu_budgets(plan))
    w = plan.weight_bytes if weights is None else weights
    d = plan.draft_bytes if draft is None else draft
    cpu_moe_saving = (plan.cpu_moe_layers * IQ2_CPU_EXPERT_BYTES_PER_LAYER
                      if plan.profile == 'deepseek-h200-iq2' else 0)
    return w - plan.table_bytes - cpu_moe_saving + d + kv_bytes(plan) + plan.reserve_bytes

def gsq_files(quant: str) -> dict[str, Any]:
    return {n: meta for n, meta in GSQ_FILES.items() if n == GSQ_ENGRAM_NAME or f'-{quant}-' in n}


def gsq_cpu_expert_bytes(plan: Plan) -> int:
    return sum(GSQ_LAYOUT[plan.quant]['expert_layer_bytes'][:plan.cpu_moe_layers])


def gsq_host_requirement(plan: Plan) -> int:
    layout = GSQ_LAYOUT[plan.quant]
    return (gsq_cpu_expert_bytes(plan) + layout['input_bytes'] + layout['metadata_bytes']
            + 12*GIB + plan.cache_ram*MIB
            + (GSQ_ENGRAM_TENSOR_BYTES if plan.gsq_engram_device == 'ram' else 0))


def gsq_device_reserve(plan: Plan, index: int) -> int:
    # At most four device-local 512-wide KV sources and eight 128-wide index sources.
    # All are f16/K-only. Bound each at full context, plus compact raw/state storage.
    kv = 6*1024*((plan.context+255)//256*256) + 256*MIB
    # Bound a complete f32 index score tile, even though candidate filtering can reduce it.
    scores = plan.context * 32 * plan.ubatch * 4
    tables = (sum(n for i, n in enumerate(GSQ_ENGRAM_TABLE_BYTES)
                  if i % len(plan.gpus) == index) if plan.gsq_engram_device == 'gpu' else 0)
    return 6*GIB + kv + scores + tables


def gsq_units(plan: Plan) -> list[int]:
    layout=GSQ_LAYOUT[plan.quant]
    return [n-(layout['expert_layer_bytes'][i] if i < plan.cpu_moe_layers else 0)
            for i,n in enumerate(layout['layer_bytes'])] + [layout['output_bytes']]


def gsq_balanced_split(plan: Plan) -> str | None:
    if len(plan.gpus)==1:return None
    units=gsq_units(plan);prefix=[0]
    for n in units:prefix.append(prefix[-1]+n)
    states={0:(0.0,[])}
    for index,gpu in enumerate(plan.gpus):
        following={};remaining=len(plan.gpus)-index-1
        for start,(score,cuts) in states.items():
            for end in range(start+1,len(units)-remaining+1):
                budget=prefix[end]-prefix[start]+gsq_device_reserve(plan,index)
                candidate=max(score,budget/max(1,gpu.free*MIB))
                if end not in following or candidate<following[end][0]:following[end]=(candidate,cuts+[end])
        states=following
    cuts=states[len(units)][1]
    positions=[0.0]+[end-0.5 for end in cuts[:-1]]+[41.0]
    return ','.join(f'{b-a:g}' for a,b in zip(positions,positions[1:]))


def gsq_gpu_budgets(plan: Plan) -> list[int]:
    units=gsq_units(plan)
    if len(plan.gpus)==1:
        if plan.gsq_split is not None:raise Error('GSQ tensor-split needs multiple GPUs')
        return [sum(units)+gsq_device_reserve(plan,0)]
    try:
        raw=[float(x) for x in (plan.gsq_split or '').split(',')]
        if len(raw)!=len(plan.gpus) or not all(math.isfinite(x) and 0<x<=1e6 for x in raw):raise ValueError()
        f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
        total=0.0;splits=[]
        for x in raw:total=f32(total+f32(x));splits.append(total)
        splits=[f32(x/total) for x in splits]
    except (ValueError,OverflowError,struct.error) as exc:
        raise Error('GSQ tensor-split requires one positive finite weight <=1e6 per GPU') from exc
    budgets=[gsq_device_reserve(plan,i) for i in range(len(plan.gpus))]
    for il,n in enumerate(units):
        i=bisect.bisect_right(splits,f32(il/41))
        if i>=len(budgets):raise Error('Invalid GSQ layer placement')
        budgets[i]+=n
    return budgets


def make_gsq_plan(args, mode: str, gpus: list[GPU]) -> Plan:
    defaults=GSQ_PROFILES[mode]
    quant=args.quant or defaults['quant']
    if quant not in GSQ_LAYOUT:raise Error('GSQ uses IQ1_M/IQ2_XXS/IQ2_XS; Q2_K and Unsloth UD-* are different releases')
    if args.variant is not None or args.reuse_engram:
        raise Error('GSQ does not use Unsloth --variant or Q6 --reuse-engram; its pinned shared IQ4_NL file is reused directly')
    if args.spec not in (None,'none') or args.dspark not in (None,0) or args.draft_depth not in (None,0):
        raise Error('This GSQ target runtime supports ordinary decoding only; MiniMax DFlash and Unsloth DSpark are separate profiles')
    if args.kv not in (None,'f16'):raise Error('GSQ requires f16 KV in this experimental runtime')
    if args.ubatch>128 or args.batch>512:raise Error('GSQ reserve covers ubatch <=128 and batch <=512')
    context=args.context if args.context is not None else defaults['context']
    if not 4096<=context<=1048576:raise Error('GSQ context must be 4096..1048576; no automatic extension')
    cpu=args.cpu_moe_layers if args.cpu_moe_layers is not None else 0
    if not 0<=cpu<=40:raise Error('GSQ cpu-moe-layers must be 0..40 physical MoE layers; no implicit offload')
    placement=args.gsq_engram or defaults['engram']
    if args.engram_ram:
        if placement=='gpu':raise Error('--engram-ram conflicts with GPU Engram; choose --gsq-engram cpu or omit --engram-ram')
        placement='ram'
    effort={'low':'50','medium':'62','high':'75','max':'100','xhigh':'100'}.get(args.reasoning_effort,args.reasoning_effort) or '50'
    if not effort.isdigit() or not 1<=int(effort)<=100:raise Error('GSQ reasoning effort must be 1..100 (publisher measured at 50)')
    ident=mode+'-'+quant.lower()
    root=Path(os.environ.get('GSQ_ROOT',str(Path(args.root).expanduser().absolute()/ident))).expanduser().absolute()
    plan=Plan(profile=mode,model_id=ident,repo=GSQ_REPO,revision=GSQ_REV,quant=quant,context=context,
              kv='f16',spec='none',depth=0,runtime=GSQ_RUNTIME,runtime_ref='smalinin-my_build+GSQ-external-Engram-adapter',
              weight_bytes=sum(f['size'] for f in gsq_files(quant).values()),
              table_bytes=GSQ_FILES[GSQ_ENGRAM_NAME]['size'],draft_bytes=0,reserve_bytes=0,root=root,gpus=gpus,
              effort=effort,cache_ram=args.cache_ram,port=args.port,ubatch=args.ubatch,batch=args.batch,
              model_dir=Path(args.model_dir).expanduser().absolute() if args.model_dir else
                        Path(args.root).expanduser().absolute()/'gsq-models'/GSQ_REV,cpu_moe_layers=cpu,
              gsq_engram_device=placement)
    plan.gsq_split=args.tensor_split or gsq_balanced_split(plan)
    plan.reserve_bytes=sum(6*GIB+plan.context*32*plan.ubatch*4 for _ in gpus)
    gsq_gpu_budgets(plan)
    return plan


def verify_gsq_runtime(src: Path) -> None:
    changed=set(run(['git','-C',str(src),'diff','--name-only','HEAD'],capture=True,quiet=True).splitlines())
    staged=run(['git','-C',str(src),'diff','--cached','--name-only'],capture=True,quiet=True).strip()
    if staged or changed!={'src/llama-model-loader.cpp'}:
        raise Error('GSQ runtime has unrecognized edits or lacks the pinned loader adapter')
    if file_sha256(src/'src/llama-model-loader.cpp')!=GSQ_LOADER_SHA256:
        raise Error('GSQ loader differs from pinned external-Engram adapter')


def minimax_host_requirement(plan: Plan) -> int:
    # Resident experts + CPU input embedding + metadata + prompt cache.
    # 12 GiB covers CPU workspaces, loader buffers and additional headroom.
    # Full-model mmap virtual address space is NOT another resident model copy.
    return (plan.cpu_moe_layers * MINIMAX_EXPERT_BYTES + MINIMAX_INPUT_BYTES
            + MINIMAX_METADATA_BYTES + plan.cache_ram * MIB + 12 * GIB)

def minimax_units(plan: Plan) -> list[int]:
    per_layer_kv = kv_bytes(plan) // 60
    return [((MINIMAX_DENSE_LAYER_BYTES if il < 3 else MINIMAX_MOE_LAYER_BYTES)
             - (MINIMAX_EXPERT_BYTES if 3 <= il < 3 + plan.cpu_moe_layers else 0)
             + per_layer_kv) for il in range(60)] + [MINIMAX_OUTPUT_BYTES]

def minimax_device_reserve(plan: Plan, index: int) -> int:
    # Draft is explicitly confined to CUDA0. Allow its complete F16 weights
    # plus another 2 GiB for draft KV, extracted features and execution buffers.
    return 6 * GIB + ((MINIMAX_DFLASH_MAX_BYTES + 2 * GIB)
                      if plan.spec == 'dflash' and index == 0 else 0)

def minimax_balanced_split(plan: Plan) -> str | None:
    if len(plan.gpus) == 1:
        return None
    # Upstream splits by LAYER COUNT. Balance resident bytes including
    # asymmetric CPU experts, contiguous layers, output weight and KV.
    units = minimax_units(plan)
    prefix = [0]
    for value in units:
        prefix.append(prefix[-1] + value)
    states = {0: (0.0, [])}
    for index, gpu in enumerate(plan.gpus):
        following = {}
        remaining = len(plan.gpus) - index - 1
        for start, (score, cuts) in states.items():
            for end in range(start + 1, len(units) - remaining + 1):
                budget = prefix[end] - prefix[start] + minimax_device_reserve(plan, index)
                candidate = max(score, budget / max(1, gpu.free * MIB))
                if end not in following or candidate < following[end][0]:
                    following[end] = (candidate, cuts + [end])
        states = following
    cuts = states[len(units)][1]
    # Place boundaries between layer positions, avoiding float rounding
    # moving an entire 2.5 GiB layer in upstream C++ upper_bound().
    positions = [0.0] + [end - 0.5 for end in cuts[:-1]] + [61.0]
    return ','.join(f'{b-a:g}' for a, b in zip(positions, positions[1:]))

def minimax_gpu_budgets(plan: Plan) -> list[int]:
    units = minimax_units(plan)
    if len(plan.gpus) == 1:
        if plan.minimax_split is not None:
            raise Error('--tensor-split needs multiple GPUs')
        return [sum(units) + minimax_device_reserve(plan, 0)]
    try:
        raw = [float(x) for x in (plan.minimax_split or '').split(',')]
        if len(raw) != len(plan.gpus) or not all(math.isfinite(x) and 0 < x <= 1e6 for x in raw):
            raise ValueError()
        f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
        total, splits = 0.0, []
        for value in raw:
            total = f32(total + f32(value))
            splits.append(total)
        splits = [f32(x / total) for x in splits]
    except (ValueError, OverflowError, struct.error) as exc:
        raise Error('MiniMax tensor-split requires one positive finite weight <=1e6 per GPU') from exc
    budgets = [minimax_device_reserve(plan, i) for i in range(len(plan.gpus))]
    for il, value in enumerate(units):
        device = bisect.bisect_right(splits, f32(il / 61))
        if device >= len(budgets):
            raise Error('Invalid MiniMax layer placement')
        budgets[device] += value
    return budgets


def cgroup_reclaimable_file_bytes(stats: dict[str, str]) -> int:
    """Estimate reclaimable clean file cache, not anonymous RAM or tmpfs.

    Both active_file and inactive_file can be reclaimed by the kernel. The
    previous check included only inactive_file, underestimating available RAM
    after a large model was read into the page cache. Avoid counting shmem,
    dirty/writeback data, or unevictable pages as free space.
    """
    def stat(name: str, fallback: str | None = None) -> int:
        try:
            return max(0, int(stats.get(name, stats.get(fallback, '0') if fallback else '0')))
        except ValueError:
            return 0

    file_lru = stat('inactive_file', 'total_inactive_file') + stat('active_file', 'total_active_file')
    # cgroup v2 uses `file`; legacy v1 uses `total_cache`. The LRU sum is a
    # fallback for kernels that do not export a usable file/cache counter.
    cached = stat('file', 'total_cache') if ('file' in stats or 'total_cache' in stats) else file_lru
    clean = max(0, cached - stat('shmem', 'total_shmem')
                - stat('file_dirty', 'total_dirty')
                - stat('file_writeback', 'total_writeback'))
    return min(file_lru, clean)


def available_ram(*, meminfo: Path | None = None,
                  cgroup_roots: list[tuple[Path, str, str, str]] | None = None) -> int:
    """Host MemAvailable bounded by cgroup headroom after reclaiming clean files.

    This is an estimate, not a guarantee of allocation or RAM residency.
    A cache-heavy cgroup can reclaim both active and inactive clean file pages.
    """
    if meminfo is None:
        meminfo = Path('/proc/meminfo')
    entries = dict(line.split(':', 1) for line in meminfo.read_text().splitlines())
    available = int(entries['MemAvailable'].split()[0]) * 1024
    candidates = cgroup_roots
    if candidates is None:
        candidates = [(Path('/sys/fs/cgroup'), 'memory.max', 'memory.current', 'inactive_file'),
                      (Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes', 'memory.usage_in_bytes', 'total_inactive_file')]
        # Include a non-root cgroup when the filesystem is not namespaced.
        try:
            for line in Path('/proc/self/cgroup').read_text().splitlines():
                _, controllers, rel = line.split(':', 2)
                if not controllers:
                    candidates.append((Path('/sys/fs/cgroup') / rel.lstrip('/'), 'memory.max', 'memory.current', 'inactive_file'))
                elif 'memory' in controllers.split(','):
                    candidates.append((Path('/sys/fs/cgroup/memory') / rel.lstrip('/'), 'memory.limit_in_bytes', 'memory.usage_in_bytes', 'total_inactive_file'))
        except OSError:
            pass
    for root, limit_name, used_name, _inactive in candidates:
        try:
            raw_limit = (root / limit_name).read_text().strip()
            if raw_limit == 'max':
                continue  # no cgroup hard limit; host MemAvailable still applies
            limit, used = int(raw_limit), int((root / used_name).read_text())
            stats = dict(line.split() for line in (root / 'memory.stat').read_text().splitlines())
            reclaimable = cgroup_reclaimable_file_bytes(stats)
            available = min(available, max(0, limit - used + reclaimable))
        except (OSError, ValueError):
            pass
    return available


def existing_parent(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path

def check_storage(path: Path, allow_network: bool) -> None:
    if not shutil.which('findmnt'):
        say('WARNING: findmnt unavailable; verify that model storage is local SSD/NVMe.')
        return
    fs = run(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(existing_parent(path))],
             capture=True, quiet=True, timeout=10).strip().lower()
    if fs.startswith(('nfs', 'cifs', 'fuse', '9p', 'ceph', 'lustre')) and not allow_network:
        raise Error(f'{path}: {fs} is not accepted for mmap model storage. Use local /opt/qwen '
                    'or explicitly --allow-network-storage. This is particularly important for Engram random reads.')

def check_port(port: int) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('127.0.0.1', port))
    except OSError as e:
        raise Error(f'Port {port} is busy. Nothing was killed; choose --port or stop your old server explicitly.') from e

def preflight(plan: Plan, args, weights: int | None = None, draft: int | None = None) -> None:
    need = gpu_requirement(plan, weights, draft)
    each = need / len(plan.gpus)
    if plan.profile in GSQ_PROFILES:
        budgets=gsq_gpu_budgets(plan)
        bad=[g for g,b in zip(plan.gpus,budgets) if g.free*MIB<b]
        say('GSQ per-device budgets (GiB, includes KV, index-score bound and reserve): '+
            ', '.join(f'{b/GIB:.2f}' for b in budgets)+'; tensor-split='+str(plan.gsq_split))
    elif plan.profile.startswith('minimax-'):
        budgets = minimax_gpu_budgets(plan)
        bad = [g for g, budget in zip(plan.gpus, budgets) if g.free * MIB < budget]
        say('MiniMax per-device budgets (GiB, including 6 GiB reserve each): '
            + ', '.join(f'{b/GIB:.2f}' for b in budgets)
            + '; tensor-split=' + str(plan.minimax_split))
    else:
        bad = [g for g in plan.gpus if g.free * MIB < each]
    say(f'Planning GPU budget: {need/GIB:.2f} GiB total, {each/GIB:.2f} GiB/device; '
        'average only; see per-device budgets above for MiniMax, including reserved buffers.')
    if bad:
        if plan.profile in GSQ_PROFILES:
            raise Error('GSQ per-device VRAM budget exceeded. Explicitly choose IQ1_M/IQ2_XXS/IQ2_XS, '
                        '--cpu-moe-layers, --gsq-engram cpu, a shorter --ctxt or smaller --ubatch; '
                        'no automatic precision/context reduction or expert offload')
        if plan.profile.startswith('minimax-'):
            raise Error('MiniMax per-device VRAM budget exceeded. Explicit --cpu-moe-layers, '
                        'a smaller --ctxt, or different --tensor-split is required; '
                        'no automatic CPU offload/context reduction. 96GB experiments: '
                        'single PRO: --cpu-moe-layers 26 --ctxt 16K --cache-ram 0; '
                        'dual A40: --cpu-moe-layers 30 --ctxt 16K --cache-ram 0. '
                        'Host RAM must also pass.')
        detail = ', '.join(f'GPU {g.index}: {g.free/1024:.2f} GiB free' for g in bad)
        raise Error(f'Insufficient free VRAM: {detail}. Reduce context/KV precision explicitly or free memory; '
                    'the launcher will not silently shrink context or move compute weights to the CPU.')
    ram = available_ram()
    minimum = (96 if plan.profile in ('deepseek-h200-iq2', 'deepseek-h200-dual-iq3')
               else 96 if plan.profile == 'deepseek-pro6000-quad-q8'
               else 64 if plan.profile.startswith('deepseek')
               else 32 if plan.profile == 'ornith' else 16) * GIB
    minimum = max(minimum, (args.cache_ram + 8192) * MIB)
    if plan.profile.startswith('minimax-'):
        minimum = minimax_host_requirement(plan)
        say(f'MiniMax: {plan.cpu_moe_layers}/57 routed-expert layers on CPU '
            f'({plan.cpu_moe_layers*MINIMAX_EXPERT_BYTES/GIB:.2f} GiB); '
            f'f16 main+index KV={kv_bytes(plan)/GIB:.2f} GiB. '
            'CPU experts run on CPU, not GPU streaming. Keep their working set in RAM; '
            'SSD paging is not treated as usable capacity. No throughput/GPU-success claim.')
    if plan.profile in GSQ_PROFILES:
        minimum=gsq_host_requirement(plan)
        say(f'GSQ {plan.quant}: {plan.cpu_moe_layers}/40 MoE layers run on CPU '
            f'({gsq_cpu_expert_bytes(plan)/GIB:.2f} GiB); Engram={plan.gsq_engram_device}. '
            'Shared experts/attention stay on GPU. GPU peaks and speed are untested.')
        if plan.gsq_engram_device=='cpu' and ram<minimum+GSQ_ENGRAM_TENSOR_BYTES:
            say('GSQ Engram remains SSD-mapped; limited host cache can add random-read latency. '
                '--engram-ram requires the complete 103 GiB table plus the host working set.')
    say(f'Host memory available (cgroup-aware, reclaimable file cache included): {ram/GIB:.1f} GiB; safety floor: {minimum/GIB:.1f} GiB.')
    if ram < minimum:
        raise Error('Insufficient available host RAM for this launcher policy')
    if plan.profile in ('deepseek', 'deepseek-h200-iq2', 'deepseek-h200-dual-iq3', 'deepseek-b300-iq3', 'deepseek-b200-iq2', 'deepseek-pro6000-quad-q8') and args.engram_ram:
        # Full table (~189 GiB) plus host model/runtime overhead, conversation cache and breathing room.
        minimum_ram = (plan.table_bytes + plan.cpu_moe_layers*IQ2_CPU_EXPERT_BYTES_PER_LAYER
                       + max(32 * GIB, (args.cache_ram + 16384) * MIB))
        say(f'Engram RAM mode: require {minimum_ram/GIB:.1f} GiB currently available host RAM')
        if ram < minimum_ram:
            raise Error(f'Not enough available host RAM to keep Engram resident: {ram/GIB:.1f} GiB free, need at least {minimum_ram/GIB:.1f} GiB')
    if plan.profile == 'deepseek' and not args.engram_ram and ram < 256 * GIB:
        say('WARNING: Engram is about 189 GiB. With less than 256 GiB available host RAM, '
            'demand paging may materially affect latency; fitting in VRAM is not a speed guarantee.')
    if plan.profile == 'flint':
        say('Flint: official Recursive NVFP4 checkpoint, full documented 262K context '
            '(192K f16 default on RTX 5090). The Blackwell vLLM 485 tok/s '
            'measurement is NOT an A40/A6000 speed claim. Keep SWA compact; '
            'inference uses pinned upstream llama.cpp. Full-context performance '
            'requires a separate long-prompt smoke test.')
    if plan.profile == 'deepseek-pro6000-quad-q8' and plan.quant != 'UD-Q2_K_XL':
        say('4× RTX PRO 6000 Blackwell: all non-Engram weights and FP16 KV requested on GPUs, '
            'Engram (~202.8 GB) mapped on CPU/SSD. No CPU-MoE fallback and no automatic context reduction.')
        if ram < 240 * GIB and not args.engram_ram:
            say('WARNING: less than 240 GiB available host RAM for ~202.8 GB Engram; '
                'SSD-backed random lookups may slow decoding, even with sufficient GPU VRAM.')
        if plan.quant == 'UD-Q4_K_XL':
            say('Q4 fallback reduces non-Engram weights only ~7.7 GB vs Q8; it has slightly '
                'lower reference agreement. Q8 is normally preferable if it fits.')
    if plan.profile == 'deepseek-b200-iq2':
        say('B200: all computational weights requested on the GPU; only Engram stays '
            'on CPU/SSD. No CPU-MoE fallback; 256K IQ2 has tight HBM headroom.')
        if plan.quant == 'UD-IQ1_M':
            say('WARNING: UD-IQ1_M improves context/VRAM headroom but Unsloth reports '
                'only 88.2% 17K reference next-token agreement (versus 91.2% for IQ2). '
                'These are not SWE coding scores.')
        if ram < 160 * GIB and not args.engram_ram:
            say('WARNING: Engram table is 110.6–135.2 GB. Limited RAM for Linux page '
                'cache can cause heavy SSD random reads and poor decoding speed.')
    if plan.profile == 'deepseek-b300-iq3' and plan.quant != 'UD-Q2_K_XL':
        say('B300 IQ3: compute weights on HBM, ~161.3 GB Q6 Engram on CPU/SSD. '
            '1M context is the default; 512K loading has been measured on B300, '
            'but 1M and IQ3_S+DSpark remain unvalidated.')
        if plan.quant == 'UD-IQ3_S':
            say('IQ3_S uses the same Q6_K Engram tensor as IQ3_XXS, but the GGUF split '
                'headers differ. --reuse-engram can safely clone the verified 161 GB '
                'table using filesystem reflinks (if supported), then verify the new '
                'shard against Unsloth SHA-256. The old model is never modified.')
        if plan.spec == 'dspark':
            say('DSpark is experimental. Check acceptance and tokens/s; 3 drafts are '
                'recommended by the publisher, but speedup at 1M context is unknown.')
        if not args.engram_ram and ram < 200 * GIB:
            say('WARNING: less than 200 GiB host memory available. Engram may '
                'demand-page from SSD and hurt token throughput.')
    if plan.profile == 'deepseek-h200-dual-iq3' and plan.quant != 'UD-Q2_K_XL':
        say('2×H200 IQ3: every transformer layer on GPU, CPU-resident Engram only; '
            'no CPU expert offload. GPU fit is not guaranteed by the aggregate memory figure.')
        if plan.quant == 'UD-IQ3_S':
            say('WARNING: UD-IQ3_S has ~29 GB more weights than IQ3_XXS. At 256K or larger '
                'contexts CUDA buffers and per-device placement may OOM. Try --context 131072 first.')
        if ram < 200 * GIB and not args.engram_ram:
            say('WARNING: Engram Q6 table is ~161.3 GB. Below ~200 GiB free host memory, '
                'SSD page faults can severely degrade decode throughput.')
    if plan.profile.startswith('deepseek') and plan.quant == 'UD-Q2_K_XL':
        say('Unsloth UD-Q2_K_XL: 199.62 GB backbone on GPU; 135.17 GB Q5_K Engram '
            'on CPU/SSD; no CPU expert fallback. Smaller weights leave space for context/DSpark, '
            'but quant kernels, memory bandwidth and draft acceptance determine actual performance.')
        say('Q2_K uses the original experimental PR runtime. Full-model loading, per-device '
            'placement, quality and throughput have not been validated on this hardware.')
        say('Q2_K disables prompt-prefix reuse by default for the reported Engram bug #30944. '
            'Clients must keep cache_prompt=false; re-prefill costs time on each turn.')
        if ram < 160 * GIB and not args.engram_ram:
            say('WARNING: less than 160 GiB available host RAM for Q5_K Engram; '
                'random SSD page faults may dominate latency.')
    if plan.profile == 'deepseek-h200-iq2':
        say(f'H200 IQ2: {plan.cpu_moe_layers} first MoE layers on CPU. '
            'VRAM saving is an unmeasured estimate; verify actual startup allocations.')
        if ram < 160 * GIB:
            say('WARNING: host Engram paging can dominate latency below ~160 GiB available RAM.')
    check_storage(plan.model_dir or plan.root, args.allow_network_storage)
    if not args.prepare_only:
        check_port(plan.port)










def install_dependencies(args, mode: str) -> None:
    # Installing CUDA/drivers inside a running pod is intentionally not attempted.
    if not shutil.which('nvcc'):
        raise Error('nvcc is missing. Select a CUDA DEVELOPMENT / PyTorch DEVELOPMENT image, '
                    'not a runtime-only image. This script will not change NVIDIA drivers.')
    if args.install_deps:
        if os.geteuid() != 0 or not shutil.which('apt-get'):
            raise Error('--install-deps requires root on Debian/Ubuntu')
        env = dict(os.environ, DEBIAN_FRONTEND='noninteractive')
        run(['apt-get', 'update'], env=env)
        run(['apt-get', 'install', '-y', '--no-install-recommends', 'build-essential', 'cmake',
             'git', 'python3-venv', 'pkg-config', 'libcurl4-openssl-dev', 'ca-certificates', 'util-linux'], env=env)
    missing = [x for x in ('cmake', 'git', 'c++', 'make', 'nvcc') if not shutil.which(x)]
    if missing:
        raise Error('Missing build tools: ' + ', '.join(missing) + '. Use --install-deps.')


def check_cuda_toolkit(gpus: list[GPU]) -> None:
    blackwell_cc = {g.cc for g in gpus} & {100, 103, 120}
    if not blackwell_cc:
        return
    # NVIDIA CUDA 12.8 introduced sm_100 and sm_120. CUDA 12.9 added
    # sm_103 (B300): NVIDIA CUDA Toolkit 12.9 release notes, section 1.2.1.
    # Check nvcc, not nvidia-smi's maximum driver-supported CUDA version.
    minimum = (12, 9) if 103 in blackwell_cc else (12, 8)
    raw = run(['nvcc', '--version'], capture=True, quiet=True, timeout=20)
    match = re.search(r'release\s+(\d+)\.(\d+)', raw)
    observed = tuple(map(int, match.groups())) if match else None
    if observed is None or observed < minimum:
        shown = '.'.join(map(str, observed)) if observed else 'unknown'
        raise Error(f'SM architecture {sorted(blackwell_cc)} needs nvcc {minimum[0]}.{minimum[1]}+; '
                    f'found nvcc {shown} ({shutil.which("nvcc") or "not on PATH"}). '
                    'Use a CUDA 12.9+/13.x DEVELOPMENT image for B300, or select a newer '
                    'installed toolkit; nvidia-smi reports driver capability, not nvcc.')
    say(f'CUDA nvcc {observed[0]}.{observed[1]} supports the selected Blackwell architecture '
        f'(toolkit: {shutil.which("nvcc")})')


# Build archives live on a persistent volume; only a vetted llama-server executable
# plus build identity is included. Never model weights, Hugging Face tokens, SSH keys,
# configuration dumps, user files, or even other files in the build directory.
# The archive is untrusted if someone else can write to /workspace; use a pinned
# SHA-256 provided out-of-band for authenticated restoration of sensitive workloads.
BUILD_ARCHIVE_FORMAT = 'llama-server-pinned-runtime-v1'
BUILD_ARCHIVE_MAX_BINARY = 4 * GIB


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(16 * MIB):
            digest.update(chunk)
    return digest.hexdigest()


def build_archive_identity(plan: Plan, arch: str) -> dict[str, str]:
    raw = run(['nvcc', '--version'], capture=True, quiet=True, timeout=20)
    found = re.search(r'release\s+(\d+)\.(\d+)', raw)
    if found is None:
        raise Error('Cannot establish CUDA compiler version for portable build archive')
    toolkit = found.group(1) + '.' + found.group(2)
    libc, libc_version = platform.libc_ver()
    return dict(format=BUILD_ARCHIVE_FORMAT, runtime=plan.runtime,
                arch=arch, toolkit=toolkit, system=sys.platform,
                machine=platform.machine(), libc=libc, libc_version=libc_version,
                build='Release;GGML_CUDA=ON;BUILD_SHARED_LIBS=OFF',
                **({'runtime_patch_sha256': MINIMAX_PATCH_SHA256} if plan.profile.startswith('minimax-') else {}),
                **({'runtime_patch_sha256':GSQ_PATCH_SHA256,'source':GSQ_RUNTIME_SOURCE} if plan.profile in GSQ_PROFILES else {}))


def build_archive_path(identity: dict[str, str], args) -> Path:
    # Sanitise even the deliberately assembled fields before using them in a filename.
    key = ('llama-server-' + identity['runtime'][:16] + '-sm' + identity['arch']
           .replace(';','_') + '-cuda' + identity['toolkit'] + '-'
           + identity['system'] + '-' + identity['machine'] + '-'
           + identity['libc'] + identity['libc_version'])
    if identity.get('runtime_patch_sha256'):
        key += '-patch' + identity['runtime_patch_sha256'][:16]
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', key):
        raise Error('Unsafe build cache key')
    return Path(args.archive_dir).expanduser().absolute() / (key + '.tar.gz')


def restore_build_archive(path: Path, binary: Path, identity: dict[str, str],
                          expected_archive_sha: str | None = None) -> bool:
    if not path.is_file():
        return False
    if expected_archive_sha:
        actual = file_sha256(path)
        if actual != expected_archive_sha:
            raise Error(f'Persistent build archive checksum mismatch: {path}. '
                        'Do not execute this archive; use --no-build-archive to compile from pinned source.')
    # Never tar.extract() an executable from a potentially untrusted volume.
    # Stream only two regular, exactly named files, into a controlled temporary file.
    meta = None
    found = []
    tmp = None
    binary.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(path, 'r|gz') as archive:
            for member in archive:
                if (not member.isfile() or member.name not in ('build.json', 'llama-server')
                    or member.name in found):
                    raise Error(f'Unexpected entry in build archive: {member.name}')
                found.append(member.name)
                if member.name == 'build.json':
                    if len(found) != 1 or member.size > 8192:
                        raise Error('Invalid archive metadata layout')
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise Error('Missing build archive metadata')
                    meta = json.loads(stream.read())
                    if not isinstance(meta, dict) or any(meta.get(k) != v for k,v in identity.items()):
                        say('Build archive is for a different runtime/toolkit/ABI; compiling instead.')
                        return False
                    if not re.fullmatch(r'[a-f0-9]{64}', str(meta.get('binary_sha256',''))):
                        raise Error('Build archive lacks a valid binary SHA-256')
                else:
                    if meta is None or member.size <= 0 or member.size > BUILD_ARCHIVE_MAX_BINARY:
                        raise Error('Build archive binary has invalid size or is missing metadata')
                    if member.size != meta.get('binary_bytes'):
                        raise Error('Build archive binary size mismatch')
                    fd, name = tempfile.mkstemp(prefix='.llama-server-cache-', dir=binary.parent)
                    tmp = Path(name)
                    digest = hashlib.sha256()
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise Error('Cannot read cached llama-server executable')
                    with os.fdopen(fd,'wb') as output:
                        remaining = member.size
                        while remaining:
                            chunk = stream.read(min(16*MIB, remaining))
                            if not chunk:
                                raise Error('Truncated cached llama-server executable')
                            output.write(chunk)
                            digest.update(chunk)
                            remaining -= len(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    if digest.hexdigest() != meta['binary_sha256']:
                        raise Error('Cached llama-server binary SHA-256 mismatch')
            if found != ['build.json','llama-server'] or tmp is None:
                raise Error('Persistent archive missing expected entries')
        tmp.chmod(0o755)  # local executable, NEVER chmod /workspace
        os.replace(tmp, binary)
        say(f'Restored pinned llama-server executable from {path} (no compiler run).')
        return True
    except (tarfile.TarError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
        raise Error(f'Invalid build archive {path}: {exc}. '
                    'Compile from pinned source with --no-build-archive instead.') from exc
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def write_build_archive(path: Path, binary: Path, identity: dict[str, str]) -> None:
    if path.exists():
        return  # preserve the first successfully qualified archive
    if not binary.is_file() or not 0 < binary.stat().st_size <= BUILD_ARCHIVE_MAX_BINARY:
        raise Error('Cannot archive missing/oversized llama-server executable')
    meta = dict(identity, binary_sha256=file_sha256(binary), binary_bytes=binary.stat().st_size)
    encoded = (json.dumps(meta, sort_keys=True) + '\n').encode('utf-8')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The persistent mount may disallow chmod/chown. Create the archive
        # readable across pods from the outset; it contains no secrets.
        tmp = path.parent / ('.build-cache-' + uuid.uuid4().hex + '.tar.gz')
        prior_umask = os.umask(0o022)
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        finally:
            os.umask(prior_umask)
        try:
            with os.fdopen(fd,'wb') as output:
                with tarfile.open(fileobj=output, mode='w:gz', compresslevel=1) as archive:
                    metadata_info = tarfile.TarInfo('build.json')
                    metadata_info.size, metadata_info.mode = len(encoded), 0o644
                    metadata_info.mtime = 0
                    archive.addfile(metadata_info, io.BytesIO(encoded))
                    binary_info = tarfile.TarInfo('llama-server')
                    binary_info.size, binary_info.mode = meta['binary_bytes'], 0o755
                    binary_info.mtime = 0
                    with binary.open('rb') as source:
                        archive.addfile(binary_info, source)
                output.flush()
                os.fsync(output.fileno())
            if not path.exists():
                os.replace(tmp, path)
                say(f'Saved executable-only build archive: {path} ({path.stat().st_size/GIB:.2f} GiB, '
                    f'archive SHA-256 {file_sha256(path)}).')
                say('Security: to authenticate restoration on another pod, provide '
                    'QWEN_BUILD_ARCHIVE_SHA256 with that checksum from a trusted source.')
        finally:
            if tmp.exists():
                tmp.unlink()
    except (OSError, tarfile.TarError) as exc:
        say(f'WARNING: Could not persist executable cache in {path.parent}: {exc}; '
            'inference can continue, but a new pod may have to rebuild.')


def verify_minimax_runtime(src: Path) -> None:
    changed = set(run(['git', '-C', str(src), 'diff', '--name-only', 'HEAD'], capture=True, quiet=True).splitlines())
    staged = run(['git', '-C', str(src), 'diff', '--cached', '--name-only'], capture=True, quiet=True).strip()
    if staged or changed != set(MINIMAX_PATCHED_FILES):
        raise Error('MiniMax runtime has unexpected tracked changes or lacks the exact all-sparse patch')
    for name, digest in MINIMAX_PATCHED_FILES.items():
        if file_sha256(src / name) != digest:
            raise Error('MiniMax source differs from pinned all-60-layer patch: ' + name)


def build(plan: Plan, args, env: dict[str, str]) -> tuple[Path, str]:
    if plan.profile.startswith('deepseek'):
        raise Error('DeepSeek uses the pinned native Strata runtime; llama.cpp builds are disabled')
    # Dedicated directories keep DeepSeek's draft PR separate from stable Qwen/Ornith.
    root = Path(args.root).expanduser().absolute() / 'runtimes'
    root.mkdir(parents=True, exist_ok=True)
    arch = ';'.join(str(n) for n in sorted({g.cc for g in plan.gpus}))
    check_cuda_toolkit(plan.gpus)
    if not arch or '0' in arch.split(';'):
        raise Error('Unknown CUDA architecture')
    minimax = plan.profile.startswith('minimax-')
    gsq = plan.profile in GSQ_PROFILES
    runtime_source = GSQ_RUNTIME_SOURCE if gsq else SOURCE
    runtime_key = plan.runtime + ('-minimax-' + MINIMAX_PATCH_SHA256[:16] if minimax else '-gsq-'+GSQ_PATCH_SHA256[:16] if gsq else '')
    src = root / runtime_key
    build_dir = src / ('build-sm' + arch.replace(';', '-'))
    binary = build_dir / 'bin/llama-server'
    with (root / (runtime_key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not src.exists():
            src.mkdir()
            run(['git', '-C', str(src), 'init'])
            run(['git', '-C', str(src), 'remote', 'add', 'origin', runtime_source])
        if run(['git', '-C', str(src), 'remote', 'get-url', 'origin'], capture=True).strip() != runtime_source:
            raise Error('Unexpected runtime Git remote')
        have = subprocess.run(['git', '-C', str(src), 'cat-file', '-e', plan.runtime + '^{commit}'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not have:
            run(['git', '-C', str(src), 'fetch', '--depth', '1', 'origin', plan.runtime])
        dirty = run(['git', '-C', str(src), 'status', '--porcelain', '--untracked-files=no'], capture=True).strip()
        if dirty:
            if not (minimax or gsq):
                raise Error('Tracked runtime sources have local edits; refusing to overwrite or compile them')
            (verify_gsq_runtime if gsq else verify_minimax_runtime)(src)
        else:
            run(['git', '-C', str(src), 'checkout', '--detach', plan.runtime])
        if run(['git', '-C', str(src), 'rev-parse', 'HEAD'], capture=True).strip() != plan.runtime:
            raise Error('Runtime commit mismatch')
        if minimax:
            payload = MINIMAX_PATCH.lstrip('\n').encode()
            if hashlib.sha256(payload).hexdigest() != MINIMAX_PATCH_SHA256:
                raise Error('Embedded MiniMax runtime patch checksum mismatch')
            if not dirty:
                patch_file = root / ('minimax-' + MINIMAX_PATCH_SHA256 + '.patch')
                patch_file.write_bytes(payload)
                run(['git', '-C', str(src), 'apply', '--check', str(patch_file)])
                run(['git', '-C', str(src), 'apply', str(patch_file)])
            verify_minimax_runtime(src)
        if gsq:
            payload=GSQ_PATCH.encode()
            if hashlib.sha256(payload).hexdigest()!=GSQ_PATCH_SHA256:
                raise Error('Embedded GSQ loader adapter checksum differs')
            if not dirty:
                patch_file=root/('gsq-'+GSQ_PATCH_SHA256+'.patch');patch_file.write_bytes(payload)
                run(['git','-C',str(src),'apply','--check',str(patch_file)])
                run(['git','-C',str(src),'apply',str(patch_file)])
            verify_gsq_runtime(src)
        archive_path = None
        archive_identity = None
        if not args.no_build_archive:
            archive_identity = build_archive_identity(plan, arch)
            archive_path = build_archive_path(archive_identity, args)
            if not binary.is_file():
                if archive_path.is_file() and not args.archive_sha256:
                    say('SECURITY NOTE: restoring executable bytes from the persistent workspace. '
                        'For sensitive data, set --archive-sha256 to the checksum recorded when the archive was made.')
                restore_build_archive(archive_path, binary, archive_identity, args.archive_sha256)
        if not binary.is_file():
            run(['cmake', '-S', str(src), '-B', str(build_dir), '-DGGML_CUDA=ON',
                 '-DCMAKE_CUDA_ARCHITECTURES=' + arch, '-DCMAKE_BUILD_TYPE=Release',
                 '-DBUILD_SHARED_LIBS=OFF'], env=env)
            run(['cmake', '--build', str(build_dir), '--config', 'Release', '-j', str(args.build_jobs),
                 '--target', 'llama-server'], env=env)
        help_text = run([str(binary), '--help'], env=env, capture=True, timeout=120)
        needed = ['--split-mode', '--tensor-split', '--cache-type-k', '--fit', '--no-context-shift',
                  '--cache-ram', '--cache-prompt', '--jinja', '--reasoning-format', '--offline', '--no-webui']
        if plan.profile in ('deepseek', 'deepseek-h200-iq2', 'deepseek-h200-dual-iq3', 'deepseek-b300-iq3', 'deepseek-b200-iq2', 'deepseek-pro6000-quad-q8'):
            needed += ['--load-mode', '--lazy-mode', '--override-tensor', '--chat-template-kwargs']
            if plan.profile == 'deepseek-h200-iq2':
                needed.append('--n-cpu-moe')
            # A version label is not evidence that the architecture implementation is present.
            if not (src / 'src/models/deepseek41.cpp').is_file():
                raise Error('Pinned runtime is missing DeepSeek 4.1 implementation')
        if gsq:
            needed += ['--load-mode','--lazy-mode','--override-tensor','--n-cpu-moe','--no-cache-prompt']
            verify_gsq_runtime(src)
        if minimax:
            needed += ['--load-mode', '--lazy-mode', '--n-cpu-moe', '--moe-cache-mib', '--chat-template-kwargs', '--spec-draft-device']
            verify_minimax_runtime(src)
        if plan.profile == 'flint':
            if not (src / 'src/models/flint.cpp').is_file():
                raise Error('Pinned runtime does not contain upstream Flint implementation')
        if plan.profile == 'ornith':
            needed += ['--chat-template-file']
        if plan.spec != 'none':
            needed += ['--spec-type', '--spec-draft-model', '--spec-draft-n-max', '--spec-draft-ngl',
                       '--spec-draft-n-ubatch', '--spec-draft-sampling', 'draft-' + plan.spec]
        missing = [s for s in needed if s not in help_text]
        if missing:
            raise Error('Runtime lacks required options: ' + ', '.join(missing))
        if archive_path is not None and archive_identity is not None:
            write_build_archive(archive_path, binary, archive_identity)
        (plan.root / 'runtime-help.txt').write_text(help_text)
        (plan.root / 'runtime-version.txt').write_text(run([str(binary), '--version'], env=env, capture=True))
    return binary, help_text

def hub_python(args, env: dict[str, str]) -> Path:
    base = Path(args.root).expanduser().absolute()
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.hub-setup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        venv = base / ('download-venv-' + HUB_VERSION)
        python = venv / 'bin/python'
        if not python.exists():
            run([sys.executable, '-m', 'venv', str(venv)])
        marker = venv / '.ready'
        if not marker.exists():
            run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                 'huggingface_hub==' + HUB_VERSION], env=env)
            (venv / 'freeze.txt').write_text(run([str(python), '-m', 'pip', 'freeze'], capture=True, env=env))
            marker.write_text(HUB_VERSION + '\n')
    return python

def draft_name(plan: Plan) -> str | None:
    if plan.profile.startswith('minimax-'):
        return None  # separately pinned Inco AI sidecar, prepared after main shards
    if plan.spec == 'none':
        return None
    if plan.spec == 'dspark':
        return 'dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf'
    return f'{plan.spec}-Qwen3.8-27B-Q4_0.gguf'

def safe_path(name: str) -> None:
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or not name or str(p) != name:
        raise Error('Unsafe repository path: ' + name)

def shard_sequence(names: list[str]) -> None:
    if len(names) == 1 and not re.search(r'-\d{5}-of-\d{5}\.gguf$', names[0]):
        return
    parsed = [re.fullmatch(r'(.+)-(\d{5})-of-(\d{5})\.gguf', n) for n in names]
    if any(x is None for x in parsed):
        raise Error('Mixed split/non-split or malformed GGUF files')
    assert all(x is not None for x in parsed)
    totals = {int(x[3]) for x in parsed}
    prefixes = {x[1] for x in parsed}
    if len(totals) != 1 or len(prefixes) != 1 or totals != {len(names)} or {int(x[2]) for x in parsed} != set(range(1, len(names)+1)):
        raise Error('Incomplete, duplicate or inconsistent GGUF shard set')

def validate_manifest(plan: Plan, data: dict[str, Any]) -> None:
    if (data.get('repo'), data.get('revision'), data.get('quant'), data.get('spec')) != (
        plan.repo, plan.revision, plan.quant, plan.spec):
        raise Error('Cached manifest does not match the pinned model/profile')
    files = data.get('files', [])
    main = [f for f in files if f.get('kind') == 'model']
    drafts = [f for f in files if f.get('kind') == 'draft']
    if not main or len({f['path'] for f in files}) != len(files):
        raise Error('Empty/duplicate model file manifest')
    if plan.profile == 'flint':
        if [f['path'] for f in main] != [FLINT_NAME]:
            raise Error('Flint checkpoint must contain exactly flint-nvfp4.gguf')
        pinned = json.loads((plan.root / 'flint-source-pin.json').read_text())
        if (main[0].get('sha256') != pinned['sha256'] or
            main[0]['size'] != pinned['size']):
            raise Error('Flint model LFS hash/size differs from the first trusted publisher revision; refusing to launch')
    elif plan.profile.startswith('minimax-'):
        pinned = {name: (f['size'], f['sha256']) for name, f in MINIMAX_FILES.items()}
        observed = {f['path']: (f['size'], f.get('sha256')) for f in main}
        if observed != pinned or len(main) != 4:
            raise Error('MiniMax must use the exact four pinned all-sparse Q2_K shards with published SHA-256')
        shard_sequence([f['path'] for f in main])
    elif plan.profile in GSQ_PROFILES:
        expected_files={n:(f['size'],f['sha256']) for n,f in gsq_files(plan.quant).items()}
        if {f['path']:(f['size'],f.get('sha256')) for f in main}!=expected_files or len(main)!=5:
            raise Error('GSQ requires exactly four pinned backbone shards and the shared IQ4_NL Engram with exact SHA-256')
        shard_sequence([f['path'] for f in main if f['path']!=GSQ_ENGRAM_NAME])
    elif plan.profile.startswith('deepseek') and plan.quant == 'UD-Q2_K_XL':
        expected_files = {name: (f['size'], f['sha256']) for name, f in DS_Q2_FILES.items()}
        if {f['path']: (f['size'], f.get('sha256')) for f in main} != expected_files or len(main) != 7:
            raise Error('DeepSeek Q2_K requires the exact seven pinned Unsloth shards including Q5_K Engram')
        shard_sequence([f['path'] for f in main])
    elif plan.profile == 'qwen':
        expected = f'Qwen3.8-27B-{plan.quant}.gguf'
        if [f['path'] for f in main] != [expected]:
            raise Error('Unexpected Qwen target file')
    else:
        prefix = 'Ornith-1.5-122B' if plan.profile == 'ornith' else 'DeepSeek-V4.1-Flash-1017'
        pattern = re.escape(plan.quant + '/' + prefix + '-' + plan.quant) + r'-\d{5}-of-\d{5}\.gguf'
        if any(not re.fullmatch(pattern, f['path']) for f in main):
            raise Error('Unexpected files in the chosen quantization folder')
        shard_sequence([f['path'] for f in main])
    want_drafts = [] if plan.spec == 'none' or plan.profile.startswith('minimax-') else [draft_name(plan)]
    if [f['path'] for f in drafts] != want_drafts:
        raise Error('Missing or mismatched speculative decoder')
    if plan.spec == 'dspark' and abs(sum(f['size'] for f in drafts) - plan.draft_bytes) > 100_000_000:
        raise Error('DSpark checkpoint size is inconsistent with pinned 1017 drafter')
    if len(files) != len(main) + len(drafts):
        raise Error('Unexpected manifest file kind')
    official = data.get('official_files', [])
    if plan.profile == 'ornith':
        if data.get('official_repo') != ORNITH_OFFICIAL or data.get('official_revision') != ORNITH_OFFICIAL_REV:
            raise Error('Official Ornith metadata revision mismatch')
        if {f['path'] for f in official} != {'chat_template.jinja', 'config.json'} or len(official) != 2:
            raise Error('Missing official Ornith template/config')
    elif official:
        raise Error('Unexpected official metadata files')
    for item in files + official:
        safe_path(item['path'])
        if not isinstance(item.get('size'), int) or item['size'] <= 0:
            raise Error('Invalid file size in manifest')
        digest = item.get('sha256') or item.get('git_sha1')
        length = 64 if item.get('sha256') else 40
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{' + str(length) + '}', digest):
            raise Error('Missing/invalid published checksum')
    if any(not f.get('sha256') for f in files):
        raise Error('Every GGUF must have a published SHA-256')
    size = sum(f['size'] for f in main)
    if abs(size - plan.weight_bytes) > max(2_000_000_000, plan.weight_bytes * 0.025):
        raise Error('Pinned checkpoint size differs materially from the researched profile; do not guess its fit')
    if sum(f['size'] for f in official) > MIB:
        raise Error('Unexpectedly large official template/config')

MANIFEST_CODE = r'''
import json,sys
from huggingface_hub import HfApi
repo,rev,quant,spec,profile,official,official_rev=sys.argv[1:]
api=HfApi()
a=api.model_info(repo,revision=rev,files_metadata=True)
assert a.sha == rev, 'Hub model revision mismatch'
def item(f,kind):
    return dict(path=f.rfilename,size=f.size,sha256=f.lfs.sha256 if f.lfs else None,git_sha1=f.blob_id,kind=kind)
files=[]
for f in a.siblings:
    n=f.rfilename
    target=(n=='Qwen3.8-27B-'+quant+'.gguf') if profile=='qwen' else ((n=='flint-nvfp4.gguf') if profile=='flint' else (n.startswith(quant+'/') and n.endswith('.gguf')))
    if profile.startswith('minimax-'):
        target=n.startswith('Sakura-MiniMax-M3.1-Flash-Q2_K-') and n.endswith('.gguf')
    if profile.startswith('deepseek-gsq-'):
        target=(n=='DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf' or
                (n.startswith('DeepSeek-V4.1-Flash-1017-GSQ-RCO-'+quant+'-') and n.endswith('.gguf')))
    if target: files.append(item(f,'model'))
    if profile=='qwen' and spec!='none' and n==spec+'-Qwen3.8-27B-Q4_0.gguf': files.append(item(f,'draft'))
    if profile.startswith('deepseek') and spec=='dspark' and n=='dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf': files.append(item(f,'draft'))
out=dict(repo=repo,revision=rev,quant=quant,spec=spec,files=sorted(files,key=lambda f:f['path']),official_files=[])
if profile=='ornith':
    b=api.model_info(official,revision=official_rev,files_metadata=True)
    assert b.sha==official_rev, 'Hub official metadata revision mismatch'
    out.update(official_repo=official,official_revision=official_rev,
       official_files=[item(f,'metadata') for f in b.siblings if f.rfilename in ('config.json','chat_template.jinja')])
print(json.dumps(out))
'''

FLINT_RESOLVE_CODE = r'''
import json,sys
from huggingface_hub import HfApi
repo,revision=sys.argv[1:]
a=HfApi().model_info(repo,revision=revision,files_metadata=True)
assert len(a.sha)==40 and all(c in '0123456789abcdef' for c in a.sha), 'Invalid upstream source commit'
files=[f for f in a.siblings if f.rfilename=='flint-nvfp4.gguf']
assert len(files)==1 and files[0].lfs and len(files[0].lfs.sha256)==64, 'Official Flint GGUF missing LFS SHA-256'
assert 21_000_000_000 < files[0].size < 25_000_000_000, 'Unexpected Flint size'
print(json.dumps(dict(repo=repo,requested_revision=revision,sha=a.sha,filename=files[0].rfilename,sha256=files[0].lfs.sha256,size=files[0].size)))
'''

def pin_flint_revision(python: Path, plan: Plan, env: dict[str,str]) -> None:
    # TOFU: retrieve the publisher's commit once; later pods may opt into an
    # explicitly pinned FLINT_REVISION. We never execute downloaded HF code.
    marker = plan.root / 'flint-source-pin.json'
    override = os.environ.get('FLINT_REVISION')
    if marker.is_file():
        pin = json.loads(marker.read_text())
    else:
        info=run([str(python), '-c', FLINT_RESOLVE_CODE, FLINT_REPO, override or 'main'],
                 env=env,capture=True,quiet=True)
        pin=json.loads(info)
        if pin.get('repo') != FLINT_REPO:
            raise Error('Unexpected Flint source repository')
        atomic_json(marker,pin)
        say('Pinned Recursive Flint checkpoint revision ' + pin['sha'])
    sha=pin.get('sha')
    if (pin.get('repo') != FLINT_REPO or pin.get('filename') != FLINT_NAME or
        not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{40}',sha) or
        not isinstance(pin.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',pin['sha256']) or
        not isinstance(pin.get('size'),int) or not 21_000_000_000 < pin['size'] < 25_000_000_000):
        raise Error('Invalid local Flint source pin')
    if override and override not in (sha,'main'):
        raise Error('FLINT_REVISION conflicts with existing source pin; use a fresh model root')
    plan.revision=sha

def manifest(python: Path, plan: Plan, env: dict[str, str]) -> tuple[Path, dict[str, Any]]:
    if plan.profile == 'flint':
        pin_flint_revision(python,plan,env)
    path = plan.root / f'manifest-{plan.quant}-{plan.spec}-{plan.revision[:12]}.json'
    if path.is_file():
        data = json.loads(path.read_text())
    else:
        text = run([str(python), '-c', MANIFEST_CODE, plan.repo, plan.revision, plan.quant,
                    plan.spec, plan.profile, ORNITH_OFFICIAL, ORNITH_OFFICIAL_REV], env=env, capture=True, quiet=True)
        data = json.loads(text)
    validate_manifest(plan, data)
    atomic_json(path, data)
    return path, data

def model_folder(plan: Plan) -> Path:
    return plan.model_dir or plan.root / 'models' / plan.revision

def disk_needed(folder: Path, files: list[dict[str, Any]]) -> int:
    partial_root = folder / '.cache/huggingface/download'
    partials = list(partial_root.rglob('*.incomplete')) if partial_root.exists() else []
    result = 0
    for item in files:
        path = folder / item['path']
        if path.is_file() and path.stat().st_size == item['size']:
            continue
        partial = max((p.stat().st_size for p in partials if item['sha256'] in p.name), default=0)
        result += max(0, item['size'] - min(partial, item['size']))
    return result

def verify_file(path: Path, item: dict[str, Any]) -> None:
    if not path.is_file() or path.stat().st_size != item['size']:
        raise Error('Missing file or size mismatch: ' + str(path))
    if item.get('sha256'):
        digest, expected = hashlib.sha256(), item['sha256']
    else:
        digest, expected = hashlib.sha1(), item['git_sha1']
        digest.update(f'blob {item["size"]}\0'.encode())
    with path.open('rb') as src:
        while block := src.read(16*MIB):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise Error('Published checksum mismatch: ' + str(path) + '. File retained for investigation, not used.')

DOWNLOAD_CODE = r'''
import json,sys
from huggingface_hub import snapshot_download
m=json.load(open(sys.argv[1]))
# Exact allowlist; do not download all quants, BF16s, vision encoders, or drafters.
workers=int(sys.argv[4]); assert 1<=workers<=32
snapshot_download(m['repo'],revision=m['revision'],local_dir=sys.argv[2],
    allow_patterns=[f['path'] for f in m['files']],max_workers=workers)
if m['official_files']:
    snapshot_download(m['official_repo'],revision=m['official_revision'],local_dir=sys.argv[3],
        allow_patterns=[f['path'] for f in m['official_files']],max_workers=workers)
'''

def patch_gguf_split_fields(path: Path, source_idx: int, target_idx: int) -> None:
    """Only re-stamp Unsloth's Q6_K table split.no and split.count (UINT16).

    Its published 350-byte set_split.py uses the same GGUF header offsets. Check
    both current values before changing anything, and always work on a clone.
    """
    with path.open('r+b') as stream:
        head = stream.read(16 * MIB)
        for key, current, updated in (
            (b'split.no', source_idx - 1, target_idx - 1),
            (b'split.count', source_idx, target_idx),
        ):
            marker = struct.pack('<Q', len(key)) + key + struct.pack('<I', 2)
            pos = head.find(marker)
            if pos < 0 or head.find(marker, pos + 1) >= 0:
                raise Error('Missing/ambiguous GGUF ' + key.decode() + ' header in shared Engram')
            offset = pos + len(key) + 12
            observed = struct.unpack_from('<H', head, offset)[0]
            if observed != current:
                raise Error(f'Unexpected GGUF {key.decode()}={observed}, expected {current}')
            stream.seek(offset)
            stream.write(struct.pack('<H', updated))
        stream.flush()
        os.fsync(stream.fileno())


def reuse_verified_engram_reflink(plan: Plan, folder: Path, data: dict, args) -> bool:
    """Share Q6_K table blocks across IQ3 variants without mutating the old model.

    Reflinks are mandatory: never silently copy 161GB or hardlink and mutate
    the original. All source/destination hashes must match the pinned repo.
    Unsupported filesystems cause a clear failure, with the old model intact.
    """
    if not args.reuse_engram:
        return False
    if plan.profile not in ('deepseek-b300-iq3', 'deepseek-h200-dual-iq3') or plan.quant not in Q6_ENGRAM_SHARDS:
        raise Error('--reuse-engram is only for DeepSeek IQ3_S / IQ3_XXS B300 or dual-H200 profiles')
    if plan.repo != DS_REPO or plan.revision != DS_REV:
        raise Error('--reuse-engram requires the exact pinned Unsloth DeepSeek 1017 checkpoint')
    target_name, target_hash, target_index = Q6_ENGRAM_SHARDS[plan.quant]
    target_rel = f'{plan.quant}/{target_name}'
    target_items = [f for f in data['files'] if f['path'] == target_rel and f['kind'] == 'model']
    if len(target_items) != 1 or target_items[0].get('sha256') != target_hash:
        raise Error('Pinned Q6_K target table hash differs from published checkpoint')
    target_item = target_items[0]
    target = folder / target_rel
    if target.is_file() and target.stat().st_size == target_item['size']:
        return False  # already downloaded or prepared; standard verification follows
    if target.exists() or target.is_symlink():
        raise Error('Existing incomplete target Engram; refusing to overwrite: ' + str(target))
    source_quant = 'UD-IQ3_XXS' if plan.quant == 'UD-IQ3_S' else 'UD-IQ3_S'
    source_name, source_hash, source_index = Q6_ENGRAM_SHARDS[source_quant]
    source = folder / source_quant / source_name
    if not source.is_file() or source.is_symlink():
        raise Error('--reuse-engram requires a pre-existing local verified shard: ' + str(source))
    if source.stat().st_size != target_item['size']:
        raise Error('Q6_K source and target GGUF shard sizes differ; cannot reflink')
    # Never trust a matching filename alone. Hash the whole original source.
    verify_file(source, dict(path=str(source), size=source.stat().st_size, sha256=source_hash))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('.reflink-' + uuid.uuid4().hex + '.gguf')
    try:
        cp = subprocess.run(['cp', '--reflink=always', '--', str(source), str(temporary)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cp.returncode:
            raise Error('Filesystem cannot reflink the shared 161GB Engram table: ' +
                        cp.stderr.strip() + '. No data modified. Use a CoW-capable local '
                        'filesystem or provision extra disk; no ordinary copy fallback.')
        patch_gguf_split_fields(temporary, source_index, target_index)
        # This detects any other metadata difference or a mismatch in table bytes.
        verify_file(temporary, target_item)
        os.replace(temporary, target)
        say(f'Reused verified Q6_K Engram from {source_quant} via reflink; '
            're-stamped split header and verified exact published target SHA-256')
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def download(python: Path, plan: Plan, path: Path, data: dict, args, env: dict) -> tuple[Path, Path | None, Path | None]:
    folder = model_folder(plan)
    official = plan.root / 'official' / ORNITH_OFFICIAL_REV
    folder.mkdir(parents=True, exist_ok=True)
    reuse_verified_engram_reflink(plan, folder, data, args)
    missing_bytes = disk_needed(folder, data['files'])
    required = missing_bytes + 20*GIB
    free = shutil.disk_usage(folder).free
    if free < required:
        detail = (f'Missing checkpoint bytes: {missing_bytes/GIB:.1f} GiB; '
                  f'20.0 GiB disk reserve; available: {free/GIB:.1f} GiB; '
                  f'additional capacity needed: {(required-free)/GIB:.1f} GiB. ')
        if args.reuse_engram and plan.quant in Q6_ENGRAM_SHARDS:
            detail += ('Shared Engram verification succeeded, but your filesystem/quota may still charge '
                       'the reflinked shard separately. Compare df -h and du on both variant directories; '
                       'no model files are deleted automatically. ')
        raise Error('Insufficient checkpoint disk space. ' + detail +
                    'Expand storage or explicitly retire a no-longer-needed checkpoint.')
    cache_path = plan.root / 'verified-files.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    groups = [(folder, data['files']), (official, data['official_files'])]
    # Reuse pre-existing exact-size files only after independent checksum validation.
    missing = any(not (base / f['path']).is_file() or (base / f['path']).stat().st_size != f['size']
                  for base, files in groups for f in files)
    if missing:
        run([str(python), '-c', DOWNLOAD_CODE, str(path), str(folder), str(official), str(args.download_workers)], env=env, quiet=True)
    todo = []
    for base, files in groups:
        for item in files:
            file = base / item['path']
            st = file.stat()
            fingerprint = [item.get('sha256') or item['git_sha1'], st.st_size, st.st_mtime_ns, st.st_ctime_ns]
            if args.reverify or cache.get(str(file)) != fingerprint:
                todo.append((file, item, fingerprint))
    # SHA hashing releases the GIL. Bound the number of simultaneous disk readers.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.verify_workers) as pool:
        futures = {pool.submit(verify_file, file, item): (file, fp) for file, item, fp in todo}
        for f in concurrent.futures.as_completed(futures):
            f.result()
            file, fingerprint = futures[f]
            say('Verified ' + file.name)
            cache[str(file)] = fingerprint
            atomic_json(cache_path, cache)
    template = None
    if plan.profile == 'ornith':
        cfg = json.loads((official / 'config.json').read_text())
        text = cfg.get('text_config', cfg)
        if (cfg.get('model_type') != 'qwen3_5_moe' or text['layer_types'].count('full_attention') != 12
            or text['num_key_value_heads'] != 2 or text['head_dim'] != 256):
            raise Error('Ornith architecture no longer matches the documented KV budget')
        template = official / 'chat_template.jinja'
    if plan.profile.startswith('minimax-'):
        for name, pinned in MINIMAX_FILES.items():
            with (folder / name).open('rb') as stream:
                header = stream.read(pinned['header_size'])
            if hashlib.sha256(header).hexdigest() != pinned['header_sha256']:
                raise Error('MiniMax GGUF header differs from verified all-60-layer layout: ' + name)
    if plan.profile in GSQ_PROFILES:
        for name in gsq_files(plan.quant):
            expected=GSQ_HEADERS[name]
            with (folder/name).open('rb') as stream:header=stream.read(expected['size'])
            if hashlib.sha256(header).hexdigest()!=expected['sha256']:
                raise Error('GSQ header differs from researched tensor layout: '+name)
    target = folder / sorted(f['path'] for f in data['files'] if f['kind'] == 'model')[0]
    draft = folder / draft_name(plan) if draft_name(plan) else None
    return target, draft, template







def read_gguf_metadata(path: Path) -> dict[str, Any]:
    # Bounded header-only parser; never evaluates model-provided code.
    with path.open('rb') as header_stream:
        data = header_stream.read(16 * MIB)
    stream = io.BytesIO(data)
    formats = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
    def read(fmt):
        width = struct.calcsize('<' + fmt)
        block = stream.read(width)
        if len(block) != width:
            raise Error('Truncated/oversized GGUF metadata')
        return struct.unpack('<' + fmt, block)[0]
    def string(keep=True):
        length = read('Q')
        if length > len(data) - stream.tell():
            raise Error('Invalid GGUF metadata string length')
        block = stream.read(length)
        return block.decode() if keep else None
    def value(kind, keep):
        if kind == 8:
            return string(keep)
        if kind == 9:
            subtype, count = read('I'), read('Q')
            if count > 2_000_000:
                raise Error('Oversized GGUF metadata array')
            if not keep and subtype in formats:
                target = stream.tell() + count * struct.calcsize('<' + formats[subtype])
                if target > len(data):
                    raise Error('Truncated GGUF array')
                stream.seek(target)
                return None
            items = [value(subtype, keep) for _ in range(count)]
            return items if keep else None
        if kind not in formats:
            raise Error('Unknown GGUF metadata type')
        return read(formats[kind])
    if stream.read(4) != b'GGUF' or read('I') != 3:
        raise Error('Expected GGUF v3')
    read('Q')  # tensor count
    count = read('Q')
    if count > 10000:
        raise Error('Oversized GGUF metadata table')
    result = {}
    for _ in range(count):
        key, kind = string(), read('I')
        keep = key == 'general.architecture' or key.startswith('dflash.')
        result[key] = value(kind, keep)
    return result


def validate_minimax_dflash(path: Path) -> None:
    if not 4_000_000_000 <= path.stat().st_size <= MINIMAX_DFLASH_MAX_BYTES:
        raise Error('MiniMax DFlash2 converted size is outside the researched bound')
    meta = read_gguf_metadata(path)
    expected = {'general.architecture':'dflash', 'dflash.block_count':5,
                'dflash.embedding_length':6144, 'dflash.attention.head_count':64,
                'dflash.attention.head_count_kv':8, 'dflash.target_layers':[6,17,27,38,48,59],
                'dflash.block_size':8, 'dflash.conv_kernel_size':2,
                'dflash.conv_group_size':16, 'dflash.selector_rank':256,
                'dflash.selector_top_k':32, 'dflash.attention.sliding_window':2048}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise Error('Converted MiniMax DFlash2 metadata does not match the pinned trained draft')


def prepare_minimax_dflash(python: Path, plan: Plan, binary: Path, args, env: dict) -> Path:
    # Only called after explicit experimental acceptance. The trained Inco AI
    # draft is converted with the exact upstream converter; no remote Python is loaded.
    base = Path(args.root).expanduser().absolute() / 'minimax-dflash2' / MINIMAX_DFLASH_REV
    check_storage(base, args.allow_network_storage)
    base.mkdir(parents=True, exist_ok=True)
    output = base / 'MiniMax-M3.1-Flash-DFlash2-F16.gguf'
    receipt = base / 'converted.json'
    identity = dict(source_repo=MINIMAX_DFLASH_REPO, source_revision=MINIMAX_DFLASH_REV,
                    target_revision=MINIMAX_SOURCE_REV, converter_commit=MINIMAX_RUNTIME,
                    runtime_patch_sha256=MINIMAX_PATCH_SHA256, outtype='f16')
    with (base / '.convert.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists() or receipt.exists():
            if not output.is_file() or not receipt.is_file():
                raise Error('Incomplete MiniMax DFlash conversion retained; inspect/remove the partial artifact explicitly')
            saved = json.loads(receipt.read_text())
            if saved.get('identity') != identity or file_sha256(output) != saved.get('sha256'):
                raise Error('MiniMax DFlash2 local conversion provenance/checksum mismatch')
            validate_minimax_dflash(output)
            return output
        if shutil.disk_usage(base).free < 16 * GIB:
            raise Error('MiniMax DFlash2 preparation needs 16 GiB free for source, F16 output and converter dependencies')
        sources, target = base / 'source', base / 'target-tokenizer'
        # Snapshot allowlists include ONLY pinned safetensors/config and tokenizer metadata.
        code = ("import json,sys; from huggingface_hub import snapshot_download; "
                "snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3],"
                "allow_patterns=json.loads(sys.argv[4]))")
        for repo, revision, folder, files in (
            (MINIMAX_DFLASH_REPO, MINIMAX_DFLASH_REV, sources, MINIMAX_DFLASH_FILES),
            ('MiniMaxAI/MiniMax-M3.1-Flash', MINIMAX_SOURCE_REV, target, MINIMAX_TOKENIZER_FILES),
        ):
            run([str(python), '-c', code, repo, revision, str(folder), json.dumps(list(files))],
                env=env, quiet=True)
            for name, item in files.items():
                verify_file(folder / name, item)
        config = json.loads((sources / 'config.json').read_text())
        if (config.get('architectures') != ['DFlash2DraftModel'] or
            config.get('dflash_config', {}).get('target_layer_ids') != [5,16,26,37,47,58]):
            raise Error('Unexpected MiniMax DFlash2 source configuration')
        venv = base / 'convert-venv'
        converter_python = venv / 'bin/python'
        if not converter_python.is_file():
            run([sys.executable, '-m', 'venv', str(venv)])
        ready = venv / '.ready'
        if not ready.exists():
            run([str(converter_python), '-m', 'pip', 'install', '--disable-pip-version-check',
                 '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.11.0+cpu'], env=env)
            run([str(converter_python), '-m', 'pip', 'install', '--disable-pip-version-check',
                 'numpy==2.2.6', 'transformers==4.57.6', 'sentencepiece==0.2.1',
                 'safetensors==0.6.2', 'protobuf==4.25.8'], env=env)
            # gguf is imported from this exact checkout by convert_hf_to_gguf.py.
            (venv / 'freeze.txt').write_text(run([str(converter_python), '-m', 'pip', 'freeze'], env=env, capture=True))
            ready.write_text(MINIMAX_RUNTIME + '\n')
        src = binary.parents[2]  # <source>/build-sm.../bin/llama-server
        verify_minimax_runtime(src)
        temporary = output.with_suffix('.partial.gguf')
        convert_env = dict(env, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        convert_env.pop('NO_LOCAL_GGUF', None)
        convert_env.pop('PYTHONPATH', None)
        run([str(converter_python), str(src / 'convert_hf_to_gguf.py'), str(sources),
             '--target-model-dir', str(target), '--outtype', 'f16', '--outfile', str(temporary)], env=convert_env)
        validate_minimax_dflash(temporary)
        digest = file_sha256(temporary)
        os.replace(temporary, output)
        atomic_json(receipt, dict(identity=identity, sha256=digest, size=output.stat().st_size,
                                 qualification='Converted locally from pinned BF16 source; GPU acceptance/speed unvalidated'))
        say('Prepared MiniMax DFlash2 F16: ' + str(output) + '; SHA-256 ' + digest)
        return output



def command_for(plan: Plan, binary: Path, model: Path, draft: Path | None, template: Path | None, args) -> list[str]:
    if plan.profile.startswith('deepseek'):
        raise Error('DeepSeek uses strata_settings; llama.cpp serving is disabled')
    cmd = [str(binary), '--model', str(model), '--alias', plan.model_id,
           '--n-gpu-layers', 'all', '--split-mode', 'layer' if len(plan.gpus)>1 else 'none',
           '--ctx-size', str(plan.context), '--parallel', '1', '--batch-size', str(plan.batch),
           '--ubatch-size', str(plan.ubatch), '--cache-type-k', plan.kv, '--cache-type-v', plan.kv,
           '--flash-attn', 'on', '--fit', 'off', '--no-context-shift', '--offline',
           '--no-cache-prompt' if plan.profile in GSQ_PROFILES or (plan.profile.startswith('deepseek') and plan.quant == 'UD-Q2_K_XL') else '--cache-prompt',
           '--cache-ram', str(plan.cache_ram), '--jinja',
           *([] if plan.profile == 'flint' else ['--reasoning-format', 'deepseek']), '--host', '127.0.0.1', '--port', str(plan.port),
           '--no-webui', '--metrics', '--timeout', '3600', '--min-p', '0',
           '--presence-penalty', '0', '--repeat-penalty', '1']
    if len(plan.gpus)>1:
        # The first CUDA card can carry extra non-layer allocations. Quad PRO6000
        # uses a slight, unvalidated placement bias rather than an aggressive
        # split. Explicit --tensor-split always takes precedence.
        default_split = ('0.94,1,1,1.06' if plan.profile == 'deepseek-pro6000-quad-q8'
                         else ','.join('1' for _ in plan.gpus))
        split = plan.gsq_split if plan.profile in GSQ_PROFILES else plan.minimax_split if plan.profile.startswith('minimax-') else (args.tensor_split or default_split)
        cmd += ['--tensor-split', split]
    if plan.profile in GSQ_PROFILES:
        cmd += ['--load-mode','mmap','--lazy-mode','on' if plan.gsq_engram_device=='cpu' else 'off',
                '--temp','1.0','--top-p','1.0','--top-k','0',
                '--chat-template-kwargs',json.dumps({'reasoning_effort':int(plan.effort)},separators=(',',':'))]
        for i,il in enumerate((1,14)):
            device='CUDA'+str(i%len(plan.gpus)) if plan.gsq_engram_device=='gpu' else 'CPU'
            cmd += ['--override-tensor',r'^blk\.'+str(il)+r'\.engram_embd\.weight$='+device]
        if plan.cpu_moe_layers:cmd += ['--n-cpu-moe',str(plan.cpu_moe_layers),'--threads',str(args.cpu_threads)]
    elif plan.profile in ('deepseek', 'deepseek-h200-iq2', 'deepseek-h200-dual-iq3', 'deepseek-b300-iq3', 'deepseek-b200-iq2', 'deepseek-pro6000-quad-q8'):
        cmd += ['--load-mode', 'mmap', '--lazy-mode', 'off' if args.engram_ram else 'on',
                '--override-tensor', 'per_layer_token_embd.weight=CPU',
                '--temp', '1.0', '--top-p', '1.0', '--top-k', '0',
                '--chat-template-kwargs', json.dumps({'reasoning_effort': int(plan.effort)}, separators=(',', ':'))]
        if plan.profile == 'deepseek-h200-iq2':
            cmd += ['--n-cpu-moe', str(plan.cpu_moe_layers),
                    '--threads', str(args.cpu_threads)]
    elif plan.profile.startswith('minimax-'):
        cmd += ['--load-mode', 'mmap', '--lazy-mode', 'off', '--moe-cache-mib', '0',
                '--temp', '1.0', '--top-p', '0.95', '--top-k', '0',
                '--chat-template-kwargs', json.dumps({'reasoning_effort': plan.effort}, separators=(',', ':'))]
        if plan.cpu_moe_layers:
            # Runtime N counts physical blocks; the first THREE have dense MLPs.
            cmd += ['--n-cpu-moe', str(plan.cpu_moe_layers + 3),
                    '--threads', str(args.cpu_threads)]
    elif plan.profile == 'flint':
        cmd += ['--temp', '1.0', '--top-p', '0.95', '--top-k', '20']
    elif plan.profile == 'ornith':
        if template is None:
            raise Error('Ornith requires its official template')
        cmd += ['--chat-template-file', str(template), '--temp', '0.6', '--top-p', '0.95', '--top-k', '20']
    else:
        cmd += ['--temp', '1.0', '--top-p', '0.95', '--top-k', '20',
                '--chat-template-kwargs', json.dumps({'enable_thinking': True, 'preserve_thinking': True,
                   **({} if plan.effort == 'default' else {'reasoning_effort': plan.effort})}, separators=(',', ':'))]
    if plan.spec != 'none':
        if plan.profile.startswith('minimax-'):
            cmd += ['--spec-draft-device', 'CUDA0']
        if draft is None:
            raise Error('Missing speculative draft path')
        cmd += ['--spec-type', 'draft-' + plan.spec, '--spec-draft-model', str(draft),
                '--spec-draft-ngl', 'all', '--spec-draft-n-max', str(plan.depth),
                '--spec-draft-n-ubatch', str(min(plan.ubatch, 128)),
                '--spec-draft-sampling', args.draft_sampling]
    else:
        cmd += ['--spec-type', 'none']
    return cmd


def ssh_setup(args) -> None:
    """Optional existing-key SSH bootstrap. Never replace config/keys or enable passwords."""
    if args.no_ssh:
        return
    port = args.ssh_port
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            say(f'Port {port} already has a listener; leaving the existing SSH service/config untouched.')
            return
    except OSError:
        pass
    if not args.ssh_only and not args.setup_ssh:
        say('SSH is not listening on the configured port. Not changing it automatically; use --setup-ssh if needed.')
        return
    if os.geteuid() != 0:
        raise Error('Starting sshd requires root; existing services/keys are untouched')
    sshd = shutil.which('sshd') or ('/usr/sbin/sshd' if Path('/usr/sbin/sshd').is_file() else None)
    if sshd is None:
        raise Error('sshd missing. Use a pod with SSH already installed; this launcher will not '
                    'install/start a service whose package defaults may enable passwords.')
    home = Path.home() / '.ssh'
    keys = home / 'authorized_keys'
    supplied = os.environ.get('SSH_PUBLIC_KEY') or os.environ.get('PUBLIC_KEY')
    if supplied:
        lines = [s.strip() for s in supplied.splitlines() if s.strip()]
        for key in lines:
            if not re.match(r'^(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp\d+) [A-Za-z0-9+/=]+(?: .*)?$', key):
                raise Error('SSH_PUBLIC_KEY/PUBLIC_KEY must contain public-key lines, not private keys or commands')
            with tempfile.NamedTemporaryFile(mode='w') as tmp:
                tmp.write(key+'\n'); tmp.flush()
                run(['ssh-keygen', '-l', '-f', tmp.name], capture=True, quiet=True)
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        old = keys.read_text() if keys.exists() else ''
        additions = [line for line in lines if line not in old.splitlines()]
        if additions:
            with keys.open('a') as out:
                out.write(('\n' if old and not old.endswith('\n') else '') + '\n'.join(additions) + '\n')
        keys.chmod(0o600)
    if not keys.exists() or not keys.read_text().strip():
        raise Error('No authorized SSH public key; not starting an inaccessible SSH daemon')
    Path('/run/sshd').mkdir(exist_ok=True)
    run(['ssh-keygen', '-A'])
    opts = ['-p', str(port), '-o', 'PasswordAuthentication=no', '-o', 'KbdInteractiveAuthentication=no',
            '-o', 'PermitRootLogin=prohibit-password', '-o', 'AllowTcpForwarding=yes']
    run([sshd, '-t', *opts])
    run([sshd, *opts])
    say(f'Key-only sshd started on port {port}; host config and existing authorized keys retained.')


def serve(command: list[str], plan: Plan, env: dict[str, str]) -> int:
    clean = {k:v for k,v in env.items() if k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN')}
    say('$ ' + shlex.join(command))
    child = None
    try:
        with (plan.root / 'server.log').open('a', buffering=1) as log:
            log.write('\n=== ' + time.strftime('%Y-%m-%d %H:%M:%S') + ' ===\n')
            child = subprocess.Popen(command, env=clean, text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, bufsize=1, start_new_session=True)
            assert child.stdout is not None
            for line in child.stdout:
                sys.stdout.write(line); sys.stdout.flush(); log.write(line)
            return child.wait()
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def smoke(port: int, max_tokens: int) -> int:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base_url = f'http://127.0.0.1:{port}'
    def request(path, payload=None):
        return opener.open(urllib.request.Request(base_url+path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type':'application/json'}), timeout=900)
    try:
        with request('/health') as response:
            json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        # Some OpenAI-compatible runtimes omit a /health route.

    with request('/v1/models') as response:
        models=json.load(response)['data']
    if not models:
        raise Error('No loaded model advertised')
    ident=models[0]['id']
    common={'model':ident, 'max_tokens':max_tokens, 'temperature':0.6,
            'chat_template_kwargs':{'enable_thinking':False}}
    say('Smoke-testing '+ident+'; formatting/tool integration only, not a coding benchmark.')
    # Streaming support matters to OpenCode; check both final output and SSE framing.
    content=[]; done=False
    with request('/v1/chat/completions', {**common, 'stream':True,
        'messages':[{'role':'user','content':'Reply with exactly READY.'}]}) as response:
        for raw in response:
            line=raw.decode().strip()
            if not line.startswith('data:'):
                continue
            text=line[5:].strip()
            if text=='[DONE]':
                done=True; break
            value=json.loads(text)
            if 'error' in value:
                raise Error('Streaming server error: '+str(value['error']))
            for choice in value.get('choices',[]):
                content.append(choice.get('delta',{}).get('content') or '')
    if not done or 'READY' not in ''.join(content):
        raise Error('Streaming completion failed or exhausted the smoke token budget')
    tools=[{'type':'function','function':{'name':'echo','description':'Echo a string.',
        'parameters':{'type':'object','properties':{'text':{'type':'string'}},'required':['text'],'additionalProperties':False}}}]
    messages=[{'role':'user','content':"Call echo exactly once with text 'ok', then acknowledge the returned result."}]
    with request('/v1/chat/completions', {**common,'stream':False,'tools':tools,'tool_choice':'required','messages':messages}) as r:
        message=json.load(r)['choices'][0]['message']
    calls=message.get('tool_calls') or []
    if len(calls)!=1 or calls[0]['function']['name']!='echo' or json.loads(calls[0]['function']['arguments'])!={'text':'ok'}:
        raise Error('Structured tool-call smoke test failed')
    assistant={k:message[k] for k in ('content','reasoning_content','tool_calls') if k in message}
    messages += [{'role':'assistant',**assistant},{'role':'tool','tool_call_id':calls[0]['id'],'content':'ok'}]
    with request('/v1/chat/completions', {**common,'stream':False,'tools':tools,'tool_choice':'none','messages':messages}) as r:
        message=json.load(r)['choices'][0]['message']
    if not message.get('content') or message.get('tool_calls'):
        raise Error('Tool-result continuation smoke test failed')
    say('PASS: streaming completion, structured tool call, tool-result continuation. No external tool executed.')
    return 0


# Full-context stress smoke test for a *running* llama.cpp HTTP server.
# All prompts are synthetic. No user files, weights, tools or credentials are read.

class SmokeHTTP:
    def __init__(self, port: int, timeout: float):
        self.base = f'http://127.0.0.1:{port}'
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.timeout = timeout

    def open(self, path: str, payload=None):
        req = urllib.request.Request(self.base + path,
            data=None if payload is None else json.dumps(payload, separators=(',', ':')).encode('utf-8'),
            headers={'Content-Type':'application/json'})
        try:
            return self.opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read(512).decode('utf-8', 'replace')
            raise Error(f'HTTP {exc.code} {path}: {detail}') from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise Error(f'Server unreachable at {path}: {exc}') from exc

    def json(self, path: str, payload=None) -> dict:
        with self.open(path, payload) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise Error(f'Expected a JSON object from {path}')
        if result.get('error'):
            raise Error(f'{path}: {str(result["error"])[:350]}')
        return result

    def count_tokens(self, text: str) -> int:
        result = self.json('/tokenize', {'content': text, 'add_special': False,
                                         'parse_special': False, 'with_pieces': False})
        tokens = result.get('tokens')
        if not isinstance(tokens, list):
            raise Error('The server /tokenize endpoint did not return token IDs. '
                        'Refusing to claim a full-context test based on character counts.')
        return len(tokens)

    def stream_chat(self, payload: dict) -> dict:
        started = time.monotonic()
        first_token = None
        content, reasoning = [], []
        finish_reason = None
        done = False
        usage = None
        with self.open('/v1/chat/completions', {**payload, 'stream': True,
              'stream_options': {'include_usage': True}}) as response:
            for raw in response:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    done = True
                    break
                if not data:
                    continue
                chunk = json.loads(data)
                if chunk.get('error'):
                    raise Error('Streaming inference error: ' + str(chunk['error'])[:350])
                if isinstance(chunk.get('usage'), dict):
                    usage = chunk['usage']
                for choice in chunk.get('choices', []):
                    delta = choice.get('delta') or {}
                    if delta.get('content'):
                        if first_token is None:
                            first_token = time.monotonic()
                        content.append(delta['content'])
                    if delta.get('reasoning_content'):
                        if first_token is None:
                            first_token = time.monotonic()
                        reasoning.append(delta['reasoning_content'])
                    if choice.get('finish_reason') is not None:
                        finish_reason = choice['finish_reason']
        ended = time.monotonic()
        if not done:
            raise Error('Stream ended without [DONE], possibly a server crash or connection drop')
        if first_token is None:
            raise Error('Stream completed without a generated token')
        return {'content': ''.join(content), 'reasoning': ''.join(reasoning),
                'ttft_s': round(first_token - started, 2),
                'elapsed_s': round(ended - started, 2),
                'finish_reason': finish_reason,
                'usage': usage or {}}


def synthetic_stress_line(i: int) -> str:
    # Deliberately nonidentical source-like input, no code from the user's project.
    h = hashlib.blake2s(str(i).encode('ascii'), digest_size=8).hexdigest()
    a, b = 1009 + (i * 37) % 7919, (i * 67 + 19) % 99991
    return (f'module Synthetic.Run_{i:07d} where\n'
            f'-- synthetic checksum {h}; fragment {i:07d} of an artificial repository\n'
            f'compute_{i:07d} x = (x * {a} + {b}) `mod` 1000003\n')


def synthetic_stress_prompt(http: SmokeHTTP, n_ctx: int, fraction: float,
                            run_id: str) -> tuple[str, int, dict]:
    target = int(n_ctx * fraction)
    code_start = 'SMOKE_START_' + run_id
    code_mid = 'SMOKE_MIDDLE_' + run_id
    code_end = 'SMOKE_END_' + run_id
    prefix = ('SYNTHETIC GPU STRESS TEST. Every record below is generated test data. '
              'No real files or secrets are included.\n'
              f'EARLY_MARKER={code_start}\nBEGIN ARTIFICIAL REPOSITORY\n')
    suffix = (f'END ARTIFICIAL REPOSITORY\nEND_MARKER={code_end}\n'
              'FINAL INSTRUCTION: On your first line write exactly '
              f'{code_end}. Then output at least 90 lines, numbered 001 to 090, '
              'each of the form CHECK_001=OK, CHECK_002=OK, and so on. '
              'Do not call tools.\n')
    sample = ''.join(synthetic_stress_line(i) for i in range(96))
    sample_tokens = http.count_tokens(sample)
    if sample_tokens < 96:
        raise Error('Unexpectedly small tokenizer sample; refusing unsafe fill estimation')
    per_line = sample_tokens / 96
    fixed = http.count_tokens(prefix + suffix + f'MIDDLE_MARKER={code_mid}\n')
    n = max(1, min(250000, int((target - fixed) / per_line)))
    final = None
    count = None
    for attempt in range(4):
        mid = n // 2
        sections = []
        for i in range(n):
            if i == mid:
                sections.append(f'MIDDLE_MARKER={code_mid}\n')
            sections.append(synthetic_stress_line(i))
        final = prefix + ''.join(sections) + suffix
        say(f'Stress prompt: tokenizing {len(final)/MIB:.1f} MiB synthetic text '
            f'(round {attempt+1}, {n:,} records).')
        count = http.count_tokens(final)
        difference = target - count
        if abs(difference) < max(64, int(0.004 * n_ctx)):
            break
        n = max(1, min(250000, n + int(difference / per_line)))
    assert final is not None and count is not None
    if count > n_ctx - min(2048, max(128, int(0.02 * n_ctx))):
        raise Error(f'Synthetic prompt would risk context overflow ({count:,}/{n_ctx:,} tokens)')
    if count < int(n_ctx * (fraction - 0.015)) or count > int(n_ctx * (fraction + 0.015)):
        raise Error(f'Synthetic prompt calibration failed: {count:,} tokens vs target {target:,}. '
                    'Refusing to report a false full-context PASS.')
    return final, count, {'start':code_start, 'middle':code_mid, 'end':code_end}


class SmokeGPUMonitor:
    """Read-only GPU/RAM snapshots, sampled while inference blocks on HTTP."""
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.stop = threading.Event()
        self.thread = None
        self.samples = []
        self.errors = []
        self.phase = 'startup'
        self.started = time.monotonic()
        self.next_heartbeat = self.started + 30
        self.indices = None

    def _once(self):
        result = subprocess.run(['nvidia-smi',
           '--query-gpu=index,name,memory.used,memory.free,temperature.gpu',
           '--format=csv,noheader,nounits'], capture_output=True,text=True,timeout=10)
        if result.returncode:
            raise Error('nvidia-smi returned nonzero: ' + result.stderr[:150])
        gpus=[]
        for raw in result.stdout.splitlines():
            fields=[s.strip() for s in raw.split(',')]
            if len(fields)<5:
                continue
            try:
                i=int(fields[0]); used=float(fields[2]); free=float(fields[3]); temp=float(fields[4])
                gpus.append({'index':i,'name':fields[1],'used_mib':used,'free_mib':free,'temp_c':temp})
            except ValueError:
                continue
        if not gpus:
            raise Error('nvidia-smi returned no readable GPUs')
        if self.indices is None:
            # Model weights are already resident. Disregard idle secondary cards.
            self.indices = {gpu['index'] for gpu in gpus if gpu['used_mib'] >= 1024}
            if not self.indices:
                raise Error('No model-resident GPUs detected (>=1 GiB allocated)')
        selected = [gpu for gpu in gpus if gpu['index'] in self.indices]
        available = None
        try:
            for line in Path('/proc/meminfo').read_text().splitlines():
                if line.startswith('MemAvailable:'):
                    available=round(int(line.split()[1])/1024/1024,2)
                    break
        except OSError:
            pass
        cgroup = {}
        try:
            root=Path('/sys/fs/cgroup')
            for fname in ('memory.current', 'memory.max'):
                raw=(root/fname).read_text().strip()
                if raw.isdigit():
                    cgroup[fname.split('.')[1]+'_gib']=round(int(raw)/GIB,3)
            raw=(root/'memory.events').read_text()
            for event in ('oom','oom_kill'):
                found=re.search(r'^'+event+r'\s+(\d+)\s*$',raw,re.M)
                if found:
                    cgroup[event]=int(found.group(1))
        except OSError:
            pass
        sample={'t_s':round(time.monotonic()-self.started,1),'phase':self.phase,
                'gpus':selected,'host_available_gib':available,'cgroup':cgroup}
        self.samples.append(sample)
        if time.monotonic()>=self.next_heartbeat:
            self.next_heartbeat=time.monotonic()+30
            remaining=', '.join(f"{g['index']}:{g['free_mib']/1024:.1f}GiB" for g in selected)
            say(f'Stress heartbeat {sample["t_s"]:.0f}s [{self.phase}]; '
                f'free GPU HBM {remaining}.')
            callback=getattr(self,'heartbeat_callback',None)
            if callback:
                callback()

    def _run(self):
        while not self.stop.is_set():
            try:
                self._once()
            except (OSError,Error,subprocess.TimeoutExpired) as exc:
                if len(self.errors)<5:
                    self.errors.append(str(exc))
            self.stop.wait(self.seconds)

    def __enter__(self):
        self._once()  # Fail fast if GPU telemetry isn't available.
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=self.seconds+2)

    def summary(self):
        by_gpu={}
        for sample in self.samples:
            for gpu in sample['gpus']:
                row=by_gpu.setdefault(str(gpu['index']),{'name':gpu['name'],
                    'min_free_mib':gpu['free_mib'],'max_used_mib':gpu['used_mib'],'max_temp_c':gpu['temp_c']})
                row['min_free_mib']=min(row['min_free_mib'],gpu['free_mib'])
                row['max_used_mib']=max(row['max_used_mib'],gpu['used_mib'])
                row['max_temp_c']=max(row['max_temp_c'],gpu['temp_c'])
        oom_baseline=(self.samples[0]['cgroup'].get('oom_kill') if self.samples else None)
        oom_current=(self.samples[-1]['cgroup'].get('oom_kill') if self.samples else None)
        oom_change=(oom_current-oom_baseline if oom_current is not None and oom_baseline is not None else None)
        return {'samples':len(self.samples),'gpus':by_gpu,'errors':list(self.errors),
                'cgroup_oom_kill_delta':oom_change,
                'peak_cgroup_memory_gib':max((s['cgroup']['current_gib'] for s in self.samples
                      if 'current_gib' in s['cgroup']),default=None),
                'cgroup_memory_limit_gib':next((s['cgroup']['max_gib'] for s in self.samples
                      if 'max_gib' in s['cgroup']),None),
                'min_host_available_gib':min((s['host_available_gib'] for s in self.samples
                     if s['host_available_gib'] is not None),default=None)}


def smoke_full_context(args) -> int:
    """Strict, near-full, synthetic, multi-round live inference qualification."""
    h = SmokeHTTP(args.port,args.smoke_timeout_seconds)
    props = h.json('/props')
    settings = props.get('default_generation_settings') or {}
    n_ctx = settings.get('n_ctx') or props.get('n_ctx')
    try:
        n_ctx = int(n_ctx)
    except (TypeError,ValueError) as exc:
        raise Error('Server /props did not advertise its actual n_ctx; cannot certify long context') from exc
    if n_ctx < 8192:
        raise Error('The server context is too small for a long-context stress test')
    if args.context is not None and args.context != n_ctx:
        raise Error(f'Wrong server running: --ctxt {args.context:,} vs actual server {n_ctx:,} tokens')
    models=h.json('/v1/models').get('data',[])
    if not models:
        raise Error('No loaded model found for the full-context smoke test')
    model=models[0]['id']
    run_id=uuid.uuid4().hex[:12].upper()
    begun=time.time()
    report={'version':VERSION,'test':'synthetic_full_context_stress','run_id':run_id,
            'model':model,'context_tokens':n_ctx,'target_fraction':args.smoke_fill,
            'started_unix':begun,'stages':[], 'status':'RUNNING',
            'synthetic_only':True,'prompt_content_saved':False}
    destination = Path(args.smoke_report) if args.smoke_report else None
    if destination is None:
        for folder in (Path('/workspace/archive'),Path('/tmp')):
            try:
                folder.mkdir(parents=True,exist_ok=True)
                if os.access(folder,os.W_OK):
                    destination=folder / ('start-remote-model-smoke-'+time.strftime('%Y%m%d-%H%M%S')+'.json')
                    break
            except OSError:
                continue
    if destination is None:
        raise Error('No writable directory found for a full-context test report')
    say(f'Full-context soak: model={model}, server context={n_ctx:,}, '
        f'target={args.smoke_fill*100:.1f}% with 1:1 synthetic request; report={destination}')

    report_lock=threading.RLock()
    def save():
        # This only writes timings, aggregate counts and GPU readings, never prompts,
        # model responses, keys or user files. Every stage is persisted before inference.
        with report_lock:
            atomic_json(destination,report)

    def record(stage,status,**data):
        entry={'stage':stage,'status':status,'at_s':round(time.time()-begun,2),**data}
        report['stages'].append(entry)
        save()
        return entry

    monitor=None
    stage='prepare'
    previous_alarm=signal.getsignal(signal.SIGALRM)
    def total_timeout(_signum,_frame):
        raise Error(f'Smoke test exceeded the {args.smoke_max_minutes:g}-minute wall-clock limit')
    signal.signal(signal.SIGALRM,total_timeout)
    signal.setitimer(signal.ITIMER_REAL,60*args.smoke_max_minutes)
    try:
        save()
        monitor=SmokeGPUMonitor(args.smoke_gpu_sample_seconds)
        def save_heartbeat():
            with report_lock:
                report['gpu']=monitor.summary()
                report['last_heartbeat_unix']=time.time()
                save()
        monitor.heartbeat_callback=save_heartbeat
        with monitor:
            stage='construct'
            monitor.phase=stage
            report['current_stage']=stage
            save()
            text, n_raw, codes=synthetic_stress_prompt(h,n_ctx,args.smoke_fill,run_id)
            record(stage,'PASS',raw_tokens=n_raw,raw_fraction=round(n_raw/n_ctx,4),
                   payload_mib=round(len(text.encode('utf-8'))/MIB,2))
            common={'model':model,'max_tokens':args.smoke_decode_tokens,'temperature':0,
                    'chat_template_kwargs':{'enable_thinking':False}}
            stage='long_prefill_and_decode'
            monitor.phase=stage
            report['current_stage']=stage
            save()
            say(f'Phase 1: actual near-full chat prefill ({n_raw:,} raw tokens) '
                f'and {args.smoke_decode_tokens} requested decode tokens.')
            first=h.stream_chat({**common,'messages':[{'role':'user','content':text}]})
            if codes['end'] not in first['content']:
                raise Error('Long-context response did not echo END marker; long-context retrieval failed')
            usage=first.get('usage') or {}
            counted=usage.get('prompt_tokens')
            if not isinstance(counted,int):
                raise Error('Server omitted usage.prompt_tokens on the streamed request. '
                            'Cannot certify that the near-full context was actually evaluated.')
            if counted < int(n_ctx * (args.smoke_fill - 0.02)):
                raise Error(f'Server reported only {counted:,} prompt tokens; '
                            'the supposed near-full prefill was not actually evaluated')
            # Count generated output using the model tokenizer when usage is absent.
            n_out=usage.get('completion_tokens')
            if not isinstance(n_out,int):
                n_out=h.count_tokens(first['content']+first['reasoning'])
            if n_out < min(64,max(16,args.smoke_decode_tokens//4)):
                raise Error(f'Only {n_out} output tokens generated; '
                            'insufficient sustained decoding for a full-context stress pass')
            record(stage,'PASS',prompt_tokens_reported=counted,
                   raw_tokens=n_raw,decode_tokens=n_out,ttft_s=first['ttft_s'],
                   elapsed_s=first['elapsed_s'],
                   observed_prefill_tps=(round(counted / first['ttft_s'],1)
                                         if first['ttft_s']>0 else None),
                   observed_decode_tps=(round(n_out / (first['elapsed_s']-first['ttft_s']),1)
                                         if first['elapsed_s']>first['ttft_s'] else None),
                   finish_reason=first['finish_reason'])

            stage='full_context_tool_call'
            monitor.phase=stage
            report['current_stage']=stage
            save()
            say('Phase 2: tool-call round trip while retaining the near-full prompt.')
            tool_code='TOOL_'+run_id
            # Do not add the potentially long first model response to the context,
            # leaving the same prefix available for the model's prompt cache.
            messages=[{'role':'user','content':text},
                      {'role':'assistant','content':codes['end']},
                      {'role':'user','content':f'Call the echo tool exactly once with text {tool_code}.'}]
            tools=[{'type':'function','function':{'name':'echo',
                     'description':'Return the caller-supplied text unchanged.',
                     'parameters':{'type':'object','properties':{'text':{'type':'string'}},
                                   'required':['text'],'additionalProperties':False}}}]
            started=time.monotonic()
            tool_response=h.json('/v1/chat/completions',{
                **common,'max_tokens':max(256,args.smoke_decode_tokens),
                'messages':messages,'tools':tools,'tool_choice':'required','stream':False})
            choices=tool_response.get('choices') or []
            if not choices:
                raise Error('Long-context tool request returned no choices')
            assistant=choices[0]['message']
            calls=assistant.get('tool_calls') or []
            if len(calls)!=1 or calls[0].get('function',{}).get('name')!='echo':
                raise Error('Full-context model failed to produce exactly one echo tool call')
            try:
                tool_args=json.loads(calls[0]['function']['arguments'])
            except (TypeError,ValueError) as exc:
                raise Error('Full-context model returned malformed tool JSON') from exc
            if tool_args!={'text':tool_code}:
                raise Error('Full-context tool call had incorrect arguments')
            report_usage=tool_response.get('usage') or {}
            if not isinstance(report_usage.get('prompt_tokens'),int):
                raise Error('Server omitted usage.prompt_tokens for the long-context tool request')
            if report_usage['prompt_tokens'] < int(n_ctx*(args.smoke_fill-.02)):
                raise Error('Tool call was not evaluated at the expected near-full-context occupancy')
            record(stage,'PASS',elapsed_s=round(time.monotonic()-started,2),
                   prompt_tokens_reported=report_usage.get('prompt_tokens'),
                   cached_prompt_tokens=(report_usage.get('prompt_tokens_details') or {}).get('cached_tokens'))

            stage='full_context_tool_continuation'
            monitor.phase=stage
            report['current_stage']=stage
            save()
            say('Phase 3: continue after a tool result, with the large conversation still attached.')
            retained={k:assistant[k] for k in ('content','reasoning_content','tool_calls') if k in assistant}
            messages.extend([{'role':'assistant',**retained},
                             {'role':'tool','tool_call_id':calls[0]['id'],'content':tool_code}])
            started=time.monotonic()
            follow=h.json('/v1/chat/completions',{**common,'messages':messages,
                     'max_tokens':max(128,args.smoke_decode_tokens),'tools':tools,
                     'tool_choice':'none','stream':False})
            options=follow.get('choices') or []
            if not options:
                raise Error('Tool-result continuation returned no choices')
            answer=options[0]['message']
            if not answer.get('content') or answer.get('tool_calls'):
                raise Error('Model could not continue normally after full-context tool result')
            follow_count=(follow.get('usage') or {}).get('prompt_tokens')
            if not isinstance(follow_count,int) or follow_count < int(n_ctx*(args.smoke_fill-.02)):
                raise Error('Tool continuation lacked evidence of near-full-context processing')
            record(stage,'PASS',elapsed_s=round(time.monotonic()-started,2),
                   prompt_tokens_reported=follow_count)

            stage='soak'
            monitor.phase=stage
            report['current_stage']=stage
            save()
            if args.smoke_soak_minutes:
                say(f'Phase 4: {args.smoke_soak_minutes:g}-minute near-full-context soak '
                    '(repeated requests with changed short suffixes).')
                until=time.monotonic()+60*args.smoke_soak_minutes
                rounds=0
                longest_ttft=0.0
                while time.monotonic()<until:
                    rounds+=1
                    result=h.stream_chat({**common, 'max_tokens':min(128,args.smoke_decode_tokens),
                        'messages':[{'role':'user','content':text},
                        {'role':'assistant','content':codes['end']},
                        {'role':'user','content':f'Soak round {rounds}: answer only ALIVE_{run_id}.'}]})
                    if not result['content'].strip():
                        raise Error(f'Soak round {rounds} produced no answer')
                    soak_count=(result.get('usage') or {}).get('prompt_tokens')
                    if not isinstance(soak_count,int) or soak_count < int(n_ctx*(args.smoke_fill-.02)):
                        raise Error(f'Soak round {rounds} lacked evidence of processing the near-full context')
                    longest_ttft=max(longest_ttft,result['ttft_s'])
                    if rounds%3==0:
                        say(f'Soak: {rounds} near-full-context requests, '
                            f'latest TTFT {result["ttft_s"]:.1f}s.')
                    if time.monotonic()<until:
                        time.sleep(min(3,max(0,until-time.monotonic())))
                record(stage,'PASS',rounds=rounds,slowest_ttft_s=longest_ttft)
            else:
                record(stage,'SKIPPED',reason='--smoke-soak-minutes 0')

            stage='final_health'
            monitor.phase=stage
            report['current_stage']=stage
            save()
            health=h.json('/v1/models')
            if not (health.get('data') or []):
                raise Error('Server no longer advertises its model after the stress test')
            record(stage,'PASS')

        stats=monitor.summary()
        report['gpu']=stats
        min_vram_mib=min((x['min_free_mib'] for x in stats['gpus'].values()),default=None)
        if monitor.errors:
            raise Error('GPU telemetry failed: '+str(monitor.errors[:2]))
        if stats.get('cgroup_oom_kill_delta') is not None and stats['cgroup_oom_kill_delta'] > 0:
            raise Error(f'Linux cgroup recorded {stats["cgroup_oom_kill_delta"]} OOM kills during the soak')
        if min_vram_mib is None:
            raise Error('No valid GPU telemetry samples obtained')
        if min_vram_mib < args.smoke_min_free_vram_mib:
            raise Error(f'Peak GPU headroom fell to {min_vram_mib:.0f} MiB, below '
                        f'the safety threshold of {args.smoke_min_free_vram_mib} MiB')
        report['status']='PASS'
        record('qualification','PASS',min_gpu_free_mib=round(min_vram_mib),
               actual_prompt_fraction=round(n_raw/n_ctx,4))
        say(f'PASS: {n_raw:,} real tokenizer tokens ({n_raw/n_ctx:.1%} of context), '
            f'full prefill + decode, long-context tool calls and continuation, '
            f'{args.smoke_soak_minutes:g}-minute soak, '
            f'lowest free GPU memory {min_vram_mib/1024:.2f} GiB. Report: {destination}')
        return 0
    except BaseException as exc:
        if monitor:
            report['gpu']=monitor.summary()
        report['status']='FAIL'
        record(stage,'FAIL',reason=str(exc)[:400])
        say(f'FAIL during {stage}: {exc}; report: {destination}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous_alarm)
        report['completed_unix']=time.time()
        report['duration_s']=round(time.time()-begun,2)
        save()


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog='start-remote-model.sh', description='Recursive Flint, Qwen, Ornith, DeepSeek and MiniMax rental-GPU launcher.',
        formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False,
        epilog='''DeepSeek profiles now use official Strata 0.2.7.1 / signed CUDA engine 0.1.52.
All existing DeepSeek hardware profiles and exact quant selections remain available.
Quad A40 defaults: pinned GSQ IQ2_XS, 1M context, DSpark on, 9,11,10,10 layers.
--reuse prefers complete existing IQ2_XXS files, with SHA checks at preparation/serving.
Explicit --quant wins. Other profiles retain their previous defaults.
Use --prepare-only to install the pinned runtime, verify/download GGUFs and create the
stock native pack. Normal serving requires prepared local files and never downloads.
No CUDA compiler is needed for Strata; the host driver must support its CUDA 13 engine.
--show-config/--dry-run is read-only. --prepare-only does not require a free API port.

Strata uses automatic GPU expert placement across all selected GPUs.
--expert-ram-gb is a soft RAM cache in GiB; --gpu-expert-fraction is a 0..1 fraction.
These do not create a hard process limit. Leave host RAM for the OS and other weights.
--cpu-moe-layers, --tensor-split, --gsq-engram, --engram-ram, llama.cpp KV/cache/batch
controls and build-archive flags are rejected for DeepSeek. Use --spec dspark/none.
--layer-split takes per-GPU layer COUNTS summing to 40, or auto. It sets DS4_SPLIT.
--dspark N enables native DSpark with a block limit; 0 disables it. --dspark-conf defaults to 0.65.
--prepare-assets-only --prepare-only verifies the prepared runtime and stages a separate
model/draft pack without changing active server configuration, runtime packages or guard.
Native chat templates and tool parsing remain unmodified. Outgoing network operations
are blocked for the stock server and its children. Missing libseccomp prevents startup.

Example for existing /opt/qwen/gsq-models files on four A40 GPUs:
  bash start-remote-model.sh --profile deepseek-gsq-a40-quad --prepare-only --dry-run
  bash start-remote-model.sh --profile deepseek-gsq-a40-quad --prepare-only --prepare-assets-only --no-ssh
  bash start-remote-model.sh --profile deepseek-gsq-a40-quad --no-ssh
  # Ordinary rollback: --spec none --layer-split auto --quant IQ2_XXS --ctxt 1M

Qwen/Flint/Ornith/MiniMax retain their prior llama.cpp runtime and controls.
Auto selection: one supported 48GB card or 5090 -> Flint; two A40/A6000 -> Ornith;
3..7 A40/A6000 -> one Flint; >=8 A40/A6000 -> DeepSeek; four PRO6000 -> DeepSeek.
One B300/B200 or one/two H200 -> their DeepSeek profile. One to three PRO6000
select the blocked FP4-Local path unless an explicit supported profile is supplied.
The Qwen profile is Qwen3.8-27B. All API endpoints bind to 127.0.0.1, default 8897.
''')
    p.add_argument('--version',action='version',version=VERSION)
    p.add_argument('--profile',choices=('auto','flint','qwen','ornith','deepseek',
                   'deepseek-h200-iq2','deepseek-h200-dual-iq3','deepseek-b300-iq3',
                   'deepseek-b200-iq2','deepseek-pro6000-quad-q8',
                   *MINIMAX_PROFILES,*GSQ_PROFILES,
                   'flash-next-fp4'),default=os.environ.get('QWEN_PROFILE','auto'))
    p.add_argument('--root',default=os.environ.get('QWEN_ROOT','/opt/qwen'))
    p.add_argument('--model-dir',default=os.environ.get('QWEN_MODEL_DIR'))
    p.add_argument('--ds-source',choices=('unsloth','gsq-rco'),default=os.environ.get('DS_SOURCE','unsloth'),
                   help='1017 source: existing Unsloth profiles, or pinned GSQ-RCO configs (launch blocked until external Engram support is verified)')
    p.add_argument('--gsq-engram',choices=('cpu','gpu'),
                   help='Dedicated GSQ profiles: SSD/host Engram or GPU-resident tables; full-RAM CPU mode uses --engram-ram')
    p.add_argument('--port',type=int,default=envint('QWEN_PORT',8897))
    p.add_argument('--ctx','--context','--ctx-size','--ctxt','-c',dest='context',type=parse_context,
                   help='Token count or binary shorthand: 256K, 512K, 768K, 1M')
    p.add_argument('--reuse',action='store_true',help='Native GSQ: prefer existing complete IQ2_XXS; otherwise IQ2_XS. Verify SHA at prepare/serve. Explicit --quant wins.')
    p.add_argument('--quant',help='Exact GGUF quant name (legacy; --variant iq3_s is easier for B300/dual H200)')
    p.add_argument('--variant',type=str.lower,choices=('iq3_xxs','iq3_s'),
                   help='B300 or dual H200: iq3_xxs (default) or iq3_s (higher fidelity)')
    p.add_argument('--reuse-engram',action='store_true',
                   help='B300 / dual-H200 IQ3: reuse verified Q6_K Engram from other variant '
                        'using CoW reflink and GGUF header restamp; requires CoW filesystem')
    p.add_argument('--kv',choices=('f16','q8_0','q4_0'))
    p.add_argument('--spec',choices=('none','mtp','dflash','dspark'))
    p.add_argument('--dflash',dest='spec',action='store_const',const='dflash',default=argparse.SUPPRESS,
                   help='Enable target-specific DFlash: existing Qwen path or pinned MiniMax DFlash2 conversion')
    p.add_argument('--dspark',type=int,metavar='N',
                   help='Strata DSpark block limit 1..64; 0 disables. Omit to use the native block limit.')
    p.add_argument('--dspark-conf',type=float,help='Strata DSpark confidence threshold, 0..1 (default 0.65 when enabled)')
    p.add_argument('--layer-split',help='Strata layer counts per GPU, summing to 40, or auto; quad A40 defaults to 9,11,10,10')
    p.add_argument('--no-mtp','--no-dflash',dest='no_spec',action='store_true')
    p.add_argument('--draft-depth','--mtp-depth',type=int)
    p.add_argument('--draft-sampling',choices=('greedy','probabilistic'),default='greedy')
    p.add_argument('--reasoning-effort')
    p.add_argument('--gpu-count',type=int,help='Local A40/A6000 count; 4 for RTX PRO 6000 Blackwell; 1/2 for H200; 1 for B200/B300')
    p.add_argument('--cpu-moe-layers',type=int,
                   help='H200 IQ2: existing early-layer setting (default 18); MiniMax: explicitly offload 0..57 routed-expert layers (default 0)')
    p.add_argument('--expert-ram-gb',type=float,default=36,help='Strata expert RAM cache in GiB; soft budget, not a process cap (default 36)')
    p.add_argument('--gpu-expert-fraction',type=float,default=1.0,help='Strata fraction of automatically available GPU expert capacity, 0..1')
    p.add_argument('--vram-reserve-mib',type=int,default=2048,help='Strata free VRAM reserve per GPU in MiB (default 2048)')
    p.add_argument('--prefill-chunk',type=int,default=4096,help='Strata prompt chunk size in tokens (default 4096)')
    p.add_argument('--engram-cache-mib',type=int,default=256,help='Strata Engram row cache in MiB (default 256)')
    p.add_argument('--engram-threads',type=int,default=16,help='Strata Engram SSD read threads (default 16)')
    p.add_argument('--thinking-budget',type=int,default=32768,help='Strata maximum reasoning tokens per request (default 32768)')
    p.add_argument('--cpu-threads',type=int,default=envint('QWEN_CPU_THREADS',min(os.cpu_count() or 8,32)),
                   help='H200 IQ2 / MiniMax CPU MoE threads (default min(CPUs,32))')
    p.add_argument('--engram-ram',action='store_true', help='llama.cpp DeepSeek only: load the Engram table in host RAM instead of lazy SSD mapping (RAM requirement depends on quantization)')
    p.add_argument('--tensor-split',help='Comma-separated relative layer split weights for selected GPUs; e.g. 0.8,1,1,1,1,1,1,1.2')
    p.add_argument('--cache-ram',type=int,default=envint('QWEN_CACHE_RAM',8192),metavar='MiB')
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--ubatch',type=int,default=128)
    p.add_argument('--show-config','--dry-run','--ornith-dry-run','--deepseek-dry-run',dest='show_config',action='store_true')
    p.add_argument('--prepare-only','--ornith-prepare-only','--deepseek-prepare-only',dest='prepare_only',action='store_true')
    p.add_argument('--prepare-assets-only',action='store_true',help='With --prepare-only, verify the prepared Strata runtime and stage model/draft packs without changing active server files')
    p.add_argument('--smoke-test','--ornith-smoke-test','--deepseek-smoke-test',dest='smoke_test',action='store_true')
    p.add_argument('--smoke-tokens',type=int,default=8192)
    p.add_argument('--smoke-full-context','--smoke-long',dest='smoke_full_context',action='store_true',
                   help='After the quick smoke test, fill ~95%% of the running server context with '
                        'synthetic tokens, test long prefill/decode/tool continuation and soak')
    p.add_argument('--smoke-fill',type=float,default=0.95,
                   help='Fraction of server context occupied by real synthetic input tokens (default .95)')
    p.add_argument('--smoke-decode-tokens',type=int,default=512,
                   help='Maximum decode tokens during the initial full-context request (default 512)')
    p.add_argument('--smoke-soak-minutes',type=float,default=10.0,
                   help='Additional near-full-context requests after prefill/tool tests (default 10)')
    p.add_argument('--smoke-timeout-seconds',type=int,default=7200,
                   help='HTTP socket timeout during the long-context test, in seconds (default 7200)')
    p.add_argument('--smoke-max-minutes',type=float,default=180.0,
                   help='Absolute wall-clock limit for the entire full-context test (default 180 minutes)')
    p.add_argument('--smoke-min-free-vram-mib',type=int,default=2048,
                   help='Fail if any model-resident GPU drops below this amount of free VRAM (default 2048)')
    p.add_argument('--smoke-gpu-sample-seconds',type=int,default=5,
                   help='GPU monitoring interval during long inference (default 5 seconds)')
    p.add_argument('--smoke-report',help='Report path, JSON metadata/timings only; never saves prompts or responses')
    p.add_argument('--accept-experimental',action='store_true',default=enabled('DS_ACCEPT_EXPERIMENTAL'), help='Explicitly accept the pinned experimental DeepSeek PR or MiniMax all-sparse patch')
    p.add_argument('--install-deps',action='store_true',default=enabled('QWEN_INSTALL_DEPS','ORNITH_INSTALL_DEPS','DS_INSTALL_DEPS'))
    p.add_argument('--build-jobs',type=int,default=envint('QWEN_BUILD_JOBS',min(os.cpu_count() or 4,12)))
    p.add_argument('--archive-dir',default=os.environ.get('QWEN_BUILD_ARCHIVE_DIR','/workspace/archive'),
                   help='Persistent executable-only .tar.gz cache directory (default /workspace/archive)')
    p.add_argument('--no-build-archive',action='store_true',
                   help='Do not read or write persistent compiled binary archives; compile from pinned source')
    p.add_argument('--archive-build-only',action='store_true',
                   help='Save or restore compiled llama-server without model downloads, VRAM checks or server start')
    p.add_argument('--archive-sha256',default=os.environ.get('QWEN_BUILD_ARCHIVE_SHA256'),
                   help='Expected SHA-256 of .tar.gz from a trusted source; refuses modified/untrusted archives')
    p.add_argument('--download-workers',type=int,default=envint('QWEN_DOWNLOAD_WORKERS',8),
                   help='Bounded concurrent Hugging Face shard downloads (1-32; default 8)')
    p.add_argument('--verify-workers',type=int,default=envint('QWEN_VERIFY_WORKERS',4),
                   help='Bounded parallel SHA checksum readers (1-16; default 4)')
    p.add_argument('--reverify',action='store_true',default=enabled('QWEN_REVERIFY','ORNITH_REVERIFY','DS_REVERIFY'))
    p.add_argument('--allow-network-storage',action='store_true',default=enabled('QWEN_ALLOW_NETWORK_STORAGE','ORNITH_ALLOW_NETWORK_STORAGE','DS_ALLOW_NETWORK_STORAGE'))
    ssh=p.add_mutually_exclusive_group()
    ssh.add_argument('--ssh-only',action='store_true')
    ssh.add_argument('--gpu-only','--no-ssh',dest='no_ssh',action='store_true')
    p.add_argument('--setup-ssh',action='store_true',help='Start sshd if absent, using existing or explicitly supplied public keys')
    p.add_argument('--ssh-port',type=int,default=envint('SSH_PORT',22))
    p.add_argument('--ornith-help','--deepseek-help',action='help',help=argparse.SUPPRESS)
    return p


def gsq_config(args, mode: str, chosen: list[GPU]) -> dict[str, Any]:
    if not mode.startswith('deepseek'):
        raise Error('--ds-source gsq-rco requires an existing DeepSeek 1017 hardware profile')
    if args.variant is not None:
        raise Error('GSQ-RCO selects IQ1_M/IQ2_XXS/IQ2_XS with --quant; Unsloth --variant does not apply')
    if args.spec not in (None, 'none') or args.dspark is not None:
        raise Error('GSQ-RCO speculation cannot be configured without a verified compatible runtime')
    if args.reuse_engram:
        raise Error('GSQ external IQ4_NL Engram cannot use Unsloth Q6 shard header restamping')
    quant = args.quant or 'IQ2_XXS'
    if quant not in ('IQ1_M', 'IQ2_XXS', 'IQ2_XS'):
        raise Error('GSQ-RCO --quant must be IQ1_M, IQ2_XXS or IQ2_XS; Unsloth UD-* names do not apply')
    files = {name: dict(meta, kind='engram' if name == GSQ_ENGRAM_NAME else 'model')
             for name, meta in GSQ_FILES.items()
             if name == GSQ_ENGRAM_NAME or f'-{quant}-' in name}
    shard_sequence([n for n in files if n != GSQ_ENGRAM_NAME])
    backbone = sum(f['size'] for f in files.values() if f['kind'] == 'model')
    engram = files[GSQ_ENGRAM_NAME]['size']
    return dict(profile=mode, source='gsq-rco', repo=GSQ_REPO, revision=GSQ_REV,
                quant=quant, files=files, backbone_bytes=backbone, engram_bytes=engram,
                total_download_bytes=backbone+engram, status='blocked',
                runtime=None, supported_publisher_engine='Strata', reason=GSQ_BLOCK,
                existing_unsloth_runtime=DEEPSEEK_RUNTIME, runtime_branch_audit=GSQ_RUNTIME_AUDIT,
                capacity_before_context_draft_and_runtime_bytes=dict(
                    free_vram=sum(g.free*MIB for g in chosen),
                    remaining_after_backbone=sum(g.free*MIB for g in chosen)-backbone,
                    remaining_after_backbone_and_engram=sum(g.free*MIB for g in chosen)-backbone-engram,
                    caveat='Raw byte subtraction only; no fit or GPU-Engram kernel compatibility guarantee'),
                requested_context=args.context, gpus=[dataclasses.asdict(g) for g in chosen],
                memory_estimates_not_gpu_validation=True,
                full_engram_residency_host_floor_bytes=engram+32*GIB,
                fidelity_note='Publisher results are reported at effort 50; no new local quality or GPU validation')


# Official Strata pins and bootstrap. Serving never updates these artifacts.
STRATA_PINS = {'source_revision': '46c152a1310f77cd2f0ce2c7cd213962df14fe83', 'source_archive_url': 'https://api.github.com/repos/Niko1221/Strata/tarball/46c152a1310f77cd2f0ce2c7cd213962df14fe83', 'source_archive_sha256': 'fae7217a8c4239c3dc7e1cae5a6e813b2b87d556840d55579e5663371d566d2f', 'source_receipt_sha256': '2b517964fba0c633ba61dcfde1d8f12730fd83b072bcb18408b5d2c9da9feb73', 'control_version': '0.1.53', 'runtime': {'release_tag': 'v0.2.7', 'version': '0.1.52', 'archive_url': 'https://github.com/Niko1221/Strata/releases/download/v0.2.7/strata-dsv41-0.1.52-linux-x86_64-cuda-cuda13.tar.gz', 'archive_sha256': 'c1b3d0ca2a575655a395ab4a2d0afc5cd731d0e5d26fbcdb955f2359dbbc1a57', 'archive_size': 444709965, 'feed_url': 'https://github.com/Niko1221/Strata/releases/download/v0.2.7/strata-feed.json', 'manifest_sha256': 'dda8e2470c8cb96bd93b2f19fd0d23d24f7e328fea820c692e5ab7829a55db1a', 'binary_sha256': '5792fd39d3ea5ad2fd82fe4e88b5a4afd2edb15c051e87d05e1a01f17679692c', 'python_packages': ['jinja2==3.1.6', 'markupsafe==3.0.3', 'regex==2026.9.10', 'psutil==7.2.2']}}
STRATA_INSTALL_CODE = r'''
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, cast
import urllib.request


def digest(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    with path.open("rb") as stream:
        value = hashlib.sha256()
        while block := stream.read(1024**2):
            value.update(block)
        return value.hexdigest()


def checked_download(url: str, path: Path, expected: str) -> None:
    """Download an HTTPS artifact and check its pinned digest before use."""
    if path.is_file():
        if digest(path) != expected:
            raise RuntimeError(f"The existing artifact differs from its pin: {path}.")
        return
    if not url.startswith("https://"):
        raise RuntimeError("The artifact URL must use HTTPS.")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, 1024**2)
    if digest(partial) != expected:
        partial.unlink()
        raise RuntimeError(f"The downloaded artifact differs from its pin: {path}.")
    partial.replace(path)


def install_source(source: Path, info: dict[str, Any], cache: Path) -> None:
    """Extract the pinned source archive without links or path traversal."""
    if source.exists():
        return
    archive = cache / "source.tar.gz"
    checked_download(str(info["source_archive_url"]), archive, str(info["source_archive_sha256"]))
    source.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="strata-source-", dir=source.parent) as directory:
        stage = Path(directory) / "source"
        stage.mkdir()
        files: dict[str, str] = {}
        unpacked = 0
        with tarfile.open(archive) as contents:
            for entry in contents:
                parts = Path(entry.name).parts
                if entry.name.startswith("/") or ".." in parts:
                    raise RuntimeError("The source archive contains an unsafe path.")
                if len(parts) < 2 or entry.isdir():
                    continue
                if not entry.isfile():
                    raise RuntimeError("The source archive contains a link or special file.")
                unpacked += entry.size
                if unpacked > 1024**3:
                    raise RuntimeError("The source archive exceeds the 1 GiB extraction limit.")
                name = str(Path(*parts[1:]))
                if name in files:
                    raise RuntimeError("The source archive contains a duplicate file.")
                entry.name = name
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with contents.extractfile(entry) as incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                target.chmod(entry.mode & 0o777)
                files[name] = digest(stage / name)
        receipt = stage / "source-files.json"
        receipt.write_text(json.dumps(files, indent=2, sort_keys=True) + "\n")
        if digest(receipt) != info["source_receipt_sha256"]:
            raise RuntimeError("The source receipt differs from its pin.")
        (stage / "SOURCE_COMMIT").write_text(str(info["source_revision"]) + "\n")
        stage.rename(source)


def verify_source(source: Path, info: dict[str, Any]) -> None:
    """Check every upstream source file against the pinned receipt."""
    receipt = source / "source-files.json"
    if not receipt.is_file() or digest(receipt) != info["source_receipt_sha256"]:
        raise RuntimeError("The pinned Strata source receipt is missing or changed.")
    files = cast(dict[str, str], json.loads(receipt.read_text()))
    for name, expected in files.items():
        path = source / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"The upstream Strata source file is missing or changed: {path}.")
    extras = [str(path.relative_to(source)) for path in source.rglob("*.py")
              if str(path.relative_to(source)) not in files]
    if extras:
        raise RuntimeError("The Strata source contains unpinned Python files: " + ", ".join(extras[:8]))


def verify_native(source: Path, runtime: Path, settings: dict[str, Any], install: bool) -> Path:
    """Use the stock verifier for the signed feed and every engine file."""
    feed_path = runtime / "strata-feed.json"
    signature_path = runtime / "strata-feed.json.sig"
    url = str(settings["feed_url"])
    cached = feed_path.is_file() and signature_path.is_file()
    if not cached and not install:
        raise RuntimeError("The prepared Strata feed is missing. Run --prepare-only first.")
    parts: list[bytes] = []
    for path, suffix in ((feed_path, ""), (signature_path, ".sig")):
        if cached:
            parts.append(path.read_bytes())
        else:
            if not url.startswith("https://"):
                raise RuntimeError("The feed URL must use HTTPS.")
            with urllib.request.urlopen(url + suffix, timeout=120) as response:
                if not response.geturl().startswith("https://"):
                    raise RuntimeError("The feed redirected outside HTTPS.")
                payload = response.read(4 * 1024**2 + 1)
            if len(payload) > 4 * 1024**2:
                raise RuntimeError("The Strata feed or signature exceeds 4 MiB.")
            parts.append(payload)
    sys.path.insert(0, str(source))
    signing = importlib.import_module("strata_control.signing")
    feed = importlib.import_module("strata_control.feed")
    component = importlib.import_module("strata_control.engine_component")
    # The release catalog can change. Authenticate it, then require the pinned engine.
    # Catalog expiry must not force a change to this fixed runtime.
    keys = signing.load_trusted_keys(environ={})
    key_id = signing.verify_bytes(keys, parts[0], json.loads(parts[1]))
    index = feed.parse(parts[0], source=url, key_id=key_id)
    package = next((entry for entry in index.dsv41
                    if entry.version == settings["version"] and entry.platform == "linux-x86_64"
                    and entry.backend == "cuda" and entry.variant == "cuda13"), None)
    if (package is None or package.sha256 != settings["archive_sha256"]
            or package.manifest_sha256 != settings["manifest_sha256"]
            or package.size != int(settings["archive_size"])
            or index.url(package.url) != settings["archive_url"]):
        raise RuntimeError("The signed feed does not contain the pinned CUDA engine.")
    if not cached:
        for path, payload in zip((feed_path, signature_path), parts):
            partial = path.with_name(path.name + ".part")
            partial.write_bytes(payload)
            partial.replace(path)
    binary = runtime / "engine-dsv41" / f"{package.version}-linux-x86_64-cuda-cuda13" / "strata-dsv41"
    if install and not binary.exists():
        archive = runtime / Path(str(settings["archive_url"])).name
        checked_download(str(settings["archive_url"]), archive, str(settings["archive_sha256"]))
        if archive.stat().st_size != int(settings["archive_size"]):
            raise RuntimeError("The engine archive has the wrong size.")
        installer = importlib.import_module("strata_control.dsv41_engine")
        if installer.install_package(runtime / "engine-dsv41", archive) != binary:
            raise RuntimeError("The official Strata package installer refused the archive.")
    manifest = binary.parent / component.MANIFEST
    if not manifest.is_file() or digest(manifest) != package.manifest_sha256:
        raise RuntimeError("The engine manifest differs from the signed feed.")
    item = component.read_manifest(binary.parent)
    signing.install_component_verifier(keys)
    component.verify_signature(item, {component.ENV_REQUIRE_SIGNED: "1"})
    component.verify(item, full=True)
    if digest(binary) != settings["binary_sha256"]:
        raise RuntimeError("The Strata engine binary differs from its pin.")
    return binary


def ensure_venv(venv: Path, packages: list[str], install: bool) -> None:
    """Install or check only the pinned serving and tokenizer dependencies."""
    python = venv / "bin/python"
    if install and not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    if install:
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", *packages], check=True)
    code = "import importlib.metadata as m,json;print(json.dumps({d.metadata['Name'].lower():d.version for d in m.distributions()}))"
    actual = json.loads(subprocess.check_output([str(python), "-c", code], text=True))
    expected = dict(package.lower().split("==", 1) for package in packages)
    actual = {key: value for key, value in actual.items() if key not in ("pip", "setuptools")}
    if actual != expected:
        raise RuntimeError(f"The isolated Strata Python packages differ from the pins: {actual}.")


info = json.loads(sys.argv[1])
base = Path(sys.argv[2])
install = sys.argv[3] == "prepare"
source = base / "source"
settings = info["runtime"]
if install:
    base.mkdir(parents=True, exist_ok=True)
    install_source(source, info, base / "downloads")
verify_source(source, info)
verify_native(source, base, settings, install)
ensure_venv(base / "venv", settings["python_packages"], install)

'''
STRATA_NETWORK_GUARD = r'''
"""Block outbound sockets before the Strata server or its children start."""

from __future__ import annotations

import ctypes
import errno
import os
import socket
import sys


class Comparison(ctypes.Structure):
    """Use the argument comparison layout from libseccomp."""

    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


def restrict_network() -> None:
    """Retain inbound API replies. Deny new connections and addressed sends."""
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                          ctypes.c_uint, ctypes.POINTER(Comparison)]
    lib.seccomp_rule_add_array.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("libseccomp could not create the network policy.")

    def deny(name: str, *comparisons: Comparison) -> None:
        number = lib.seccomp_syscall_resolve_name(name.encode())
        if number < 0:
            raise RuntimeError(f"libseccomp does not know the required syscall: {name}.")
        args = (Comparison * len(comparisons))(*comparisons)
        result = lib.seccomp_rule_add_array(context, 0x00050000 | errno.EPERM,
                                            number, len(comparisons), args)
        if result != 0:
            raise RuntimeError(f"Cannot restrict {name}: {os.strerror(-result)}.")

    try:
        # Children retain the filter across fork and exec. Block alternate I/O paths.
        for name in ("connect", "sendmsg", "sendmmsg", "recvmsg", "recvmmsg", "io_uring_setup", "io_uring_enter",
                     "io_uring_register", "ptrace", "process_vm_writev", "pidfd_getfd"):
            deny(name)
        # send() uses sendto() with a null destination on accepted API sockets.
        deny("sendto", Comparison(4, 1, 0, 0))  # SCMP_CMP_NE
        # Exclude vsock, packet sockets, netlink and raw sockets.
        for family in range(socket.AF_INET6):
            if family not in (socket.AF_UNIX, socket.AF_INET):
                deny("socket", Comparison(0, 4, family, 0))  # SCMP_CMP_EQ
        deny("socket", Comparison(0, 6, socket.AF_INET6, 0))  # SCMP_CMP_GT
        for kind in range(16):
            if kind == socket.SOCK_SEQPACKET:
                # Linux CUDA needs a local sequenced-packet listener for UVM.
                deny("socket", Comparison(1, 7, 0xF, kind), Comparison(0, 1, socket.AF_UNIX, 0))
            elif kind != socket.SOCK_STREAM:
                deny("socket", Comparison(1, 7, 0xF, kind))  # SCMP_CMP_MASKED_EQ
        deny("socketpair", Comparison(0, 1, socket.AF_UNIX, 0))
        result = lib.seccomp_load(context)
        if result != 0:
            raise RuntimeError(f"Cannot load the network policy: {os.strerror(-result)}.")
    finally:
        lib.seccomp_release(context)


def main() -> None:
    """Close inherited descriptors, install the policy, and execute the command."""
    if len(sys.argv) < 2:
        raise RuntimeError("A command is required.")
    descriptors = [int(name) for name in os.listdir("/proc/self/fd") if name.isdecimal() and int(name) > 2]
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass
    restrict_network()
    print("[start-remote-model] Strata: outgoing connections blocked for the server and its children.",
          file=sys.stderr, flush=True)
    os.execv(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"Strata isolation failed. The server was not started: {error}")

'''


STRATA_DSPARK = {
    'repo': 'unsloth/DeepSeek-V4.1-Flash-1017-GGUF',
    'revision': 'f924b94f19c1fea02c06a65c8475b0a45e2b88b1',
    'name': 'dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf',
    'size': 7976914240,
    'sha256': '6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a',
}


def strata_reusable_gsq(args, quant: str) -> bool:
    """Find a complete candidate by size. Preparation verifies its SHA-256."""
    folder = Path(args.model_dir).expanduser().absolute() if args.model_dir else Path(args.root).expanduser().absolute()/'gsq-models'/GSQ_REV
    complete = all((folder/name).is_file() and not (folder/name).is_symlink()
                   and (folder/name).stat().st_size == item['size'] for name,item in gsq_files(quant).items())
    if complete:
        say('--reuse selected existing '+quant+' by file sizes; SHA-256 verification is required at preparation/serving.')
    return complete


def strata_pack_path(plan: Plan) -> Path:
    """Keep the DSpark pack separate from the ordinary decoding pack."""
    return model_folder(plan)/('strata-pack-'+plan.quant.lower()+('-dspark' if plan.spec == 'dspark' else ''))


def strata_draft_path(args) -> Path:
    """Use a separate directory so ordinary packers cannot discover the draft."""
    return Path(args.root).expanduser().absolute()/'strata-drafts'/STRATA_DSPARK['revision']/STRATA_DSPARK['name']


def strata_draft(args) -> None:
    """Download the exact draft only during preparation, then verify its bytes."""
    path = strata_draft_path(args)
    if path.is_symlink():
        raise Error('The Strata draft must be a regular file, not a link')
    if not path.is_file() and not args.prepare_only:
        raise Error('The pinned DSpark draft is missing. Run --prepare-only first.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent/'.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            if shutil.disk_usage(path.parent).free < STRATA_DSPARK['size'] + GIB:
                raise Error('DSpark preparation requires 8 GB for the draft plus 1 GiB free disk reserve')
            partial = path.with_name(path.name+'.part')
            if partial.is_symlink():
                raise Error('The draft partial file must not be a link')
            url = 'https://huggingface.co/'+STRATA_DSPARK['repo']+'/resolve/'+STRATA_DSPARK['revision']+'/'+STRATA_DSPARK['name']
            say('Downloading the pinned DSpark draft: '+STRATA_DSPARK['name'])
            with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as output:
                if not response.geturl().startswith('https://'):
                    raise Error('The draft download redirected outside HTTPS')
                received = 0
                while chunk := response.read(8*MIB):
                    received += len(chunk)
                    if received > STRATA_DSPARK['size']:
                        raise Error('The draft download exceeds its pinned size; partial file retained')
                    output.write(chunk)
            verify_file(partial, STRATA_DSPARK)
            partial.replace(path)
        metadata = path.stat()
        identity = [STRATA_DSPARK['sha256'], metadata.st_size, metadata.st_mtime_ns,
                    metadata.st_ctime_ns, metadata.st_ino, metadata.st_dev]
        receipt = path.parent/'verified-draft.json'
        cached = json.loads(receipt.read_text()) if receipt.is_file() else None
        if args.reverify or cached != identity:
            verify_file(path, STRATA_DSPARK)
            atomic_json(receipt, identity)
            say('Verified pinned DSpark draft SHA-256')


def strata_paths(args) -> tuple[Path, Path, Path]:
    """Keep the official runtime separate from all llama.cpp builds."""
    base = Path(args.root).expanduser().absolute() / 'runtimes' / ('strata-' + STRATA_PINS['source_revision'])
    return base, base / 'venv/bin/python', base / 'source'


def strata_options(args) -> None:
    """Reject controls whose llama.cpp meaning does not apply to Strata."""
    old = {'--cpu-moe-layers', '--tensor-split', '--gsq-engram', '--engram-ram', '--kv',
           '--cache-ram', '--batch', '--ubatch', '--draft-depth', '--mtp-depth', '--draft-sampling',
           '--archive-build-only', '--no-build-archive', '--archive-dir', '--archive-sha256', '--build-jobs'}
    used = sorted(old.intersection(args.explicit_flags))
    if used:
        raise Error('Strata does not implement these llama.cpp controls: ' + ', '.join(used)
                    + '. Use --expert-ram-gb, --gpu-expert-fraction, --prefill-chunk and --vram-reserve-mib.')
    if args.spec not in (None, 'none', 'dspark'):
        raise Error('Native Strata supports --spec none/dspark; MTP and DFlash are not configured')
    if args.dspark is not None and not 0 <= args.dspark <= 64:
        raise Error('--dspark must be 0..64; a positive value limits native drafts per round')
    if args.dspark_conf is not None and (not math.isfinite(args.dspark_conf) or not 0 <= args.dspark_conf <= 1):
        raise Error('--dspark-conf must be 0..1')
    if args.prepare_assets_only and (not args.prepare_only or args.install_deps or args.setup_ssh or args.ssh_only):
        raise Error('--prepare-assets-only requires --prepare-only and cannot install dependencies or set up SSH')
    if args.ds_source == 'gsq-rco' and args.profile not in GSQ_PROFILES:
        raise Error('Select an explicit deepseek-gsq-* profile for GSQ-RCO. The selected Unsloth quant is not substituted.')
    if not math.isfinite(args.expert_ram_gb) or not 0 <= args.expert_ram_gb <= 1024:
        raise Error('--expert-ram-gb must be 0..1024 GiB; it is a soft cache budget, not a process RAM cap')
    if not math.isfinite(args.gpu_expert_fraction) or not 0 <= args.gpu_expert_fraction <= 1:
        raise Error('--gpu-expert-fraction must be 0..1')
    if not 256 <= args.vram_reserve_mib <= 16384 or not 1 <= args.prefill_chunk <= 32768:
        raise Error('Use --vram-reserve-mib 256..16384 and --prefill-chunk 1..32768')
    if not 0 <= args.engram_cache_mib <= 4096 or not 1 <= args.engram_threads <= 64:
        raise Error('Use --engram-cache-mib 0..4096 and --engram-threads 1..64')
    if not 1 <= args.cpu_threads <= 64 or not 0 <= args.thinking_budget <= 1048576:
        raise Error('Strata uses --cpu-threads 1..64 and --thinking-budget 0..1048576')


def strata_plan(args, mode: str, gpus: list[GPU]) -> Plan:
    """Preserve checkpoint selection without llama.cpp placement estimates."""
    strata_options(args)
    if args.reuse and mode not in GSQ_PROFILES:
        raise Error('--reuse applies only to native GSQ IQ2_XXS/IQ2_XS selection')
    spec = args.spec or ('dspark' if mode == 'deepseek-gsq-a40-quad' else 'none')
    if args.dspark is not None:
        requested = 'dspark' if args.dspark else 'none'
        if args.spec is not None and args.spec != requested:
            raise Error('--spec and --dspark disagree')
        spec = requested
    if args.no_spec:
        if args.spec == 'dspark' or (args.dspark is not None and args.dspark > 0):
            raise Error('--no-mtp/--no-dflash conflicts with explicit DSpark')
        spec = 'none'
    if spec == 'none' and args.dspark_conf is not None:
        raise Error('--dspark-conf requires DSpark')
    split = args.layer_split if args.layer_split is not None else ('9,11,10,10' if mode == 'deepseek-gsq-a40-quad' else 'auto')
    if split != 'auto':
        try:
            counts = [int(value.strip()) for value in split.split(',')]
        except ValueError as exc:
            raise Error('--layer-split must be auto or positive integer layer counts') from exc
        if len(counts) != len(gpus) or any(value <= 0 for value in counts) or sum(counts) != 40:
            raise Error('--layer-split needs one positive count per selected GPU, summing to 40')
        split = ','.join(map(str, counts))
    args.layer_split = split
    if mode in GSQ_PROFILES:
        defaults = GSQ_PROFILES[mode]
        quant = args.quant or (('IQ2_XXS' if strata_reusable_gsq(args, 'IQ2_XXS') else 'IQ2_XS') if args.reuse else defaults['quant'])
        if quant not in GSQ_LAYOUT or args.variant is not None or args.reuse_engram:
            raise Error('GSQ uses IQ1_M/IQ2_XXS/IQ2_XS and the existing shared Engram; Unsloth variant/reuse flags do not apply')
        context = args.context if args.context is not None else defaults['context']
        if not 4096 <= context <= 1048576:
            raise Error('DeepSeek context must be 4096..1048576')
        ident = mode + '-' + quant.lower()
        root = Path(os.environ.get('GSQ_ROOT', str(Path(args.root).expanduser().absolute() / ident))).expanduser().absolute()
        plan = Plan(mode, ident, GSQ_REPO, GSQ_REV, quant, context, 'native', 'none', 0,
                    STRATA_PINS['source_revision'], 'official-Strata-v0.2.7.1',
                    sum(f['size'] for f in gsq_files(quant).values()), GSQ_FILES[GSQ_ENGRAM_NAME]['size'],
                    0, 0, root, gpus, args.reasoning_effort or 'high', 0, args.port, 0, 0,
                    Path(args.model_dir).expanduser().absolute() if args.model_dir else
                    Path(args.root).expanduser().absolute() / 'gsq-models' / GSQ_REV)
    else:
        # This function retains the original profile's exact quant, revision and directory.
        legacy = argparse.Namespace(**vars(args))
        legacy.reasoning_effort = 'high'  # Validate the native setting below.
        legacy.spec, legacy.dspark = 'none', None
        plan = make_plan(legacy, mode, gpus)
        if plan.spec != 'none':
            raise Error('Strata speculation must be disabled; clear the profile SPEC/DRAFT environment overrides')
        plan.cpu_moe_layers = 0
        plan.kv = 'native'
        plan.cache_ram = plan.batch = plan.ubatch = 0
        plan.runtime, plan.runtime_ref = STRATA_PINS['source_revision'], 'official-Strata-v0.2.7.1'
    effort = args.reasoning_effort or 'high'
    if effort not in ('none', 'low', 'medium', 'high'):
        raise Error('Strata configuration effort must be none/low/medium/high. API requests may send xhigh/max; no numeric llama.cpp default is retained.')
    plan.effort, plan.engine = effort, 'strata'
    plan.spec, plan.depth = spec, (args.dspark or 0) if spec == 'dspark' else 0
    plan.draft_bytes = STRATA_DSPARK['size'] if spec == 'dspark' else 0
    return plan


def strata_preflight(plan: Plan, args) -> None:
    """Check headroom without claiming a hard cap or precomputed GPU fit."""
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise Error('The pinned Strata engine requires Linux x86_64')
    minimum = (4 * GIB if args.prepare_assets_only else
               int(args.expert_ram_gb * GIB) + args.engram_cache_mib * MIB + 24 * GIB)
    available = available_ram()
    if available < minimum:
        mode = 'asset preparation' if args.prepare_assets_only else 'the cache plus 24 GiB headroom'
        raise Error(f'Strata needs {minimum/GIB:.1f} GiB available for {mode}; '
                    f'only {available/GIB:.1f} GiB is available.')
    if any(g.free < args.vram_reserve_mib + 4096 for g in plan.gpus) and not args.prepare_only:
        raise Error('A selected GPU lacks 4 GiB plus the requested reserve. Stop the old server before serving.')
    check_storage(model_folder(plan), args.allow_network_storage)
    if not args.prepare_only:
        check_port(plan.port)
    say(f'Strata expert RAM cache: {args.expert_ram_gb:g} GiB; available host RAM: {available/GIB:.1f} GiB. '
        'This is not a hard process limit. Native GPU expert placement is automatic; GPU fit remains unverified.')


def strata_settings(plan: Plan, args) -> dict[str, Any]:
    """Use the stock DeepSeek template, parser and native device ordinals."""
    base, python, source = strata_paths(args)
    binary = base / 'engine-dsv41' / '0.1.52-linux-x86_64-cuda-cuda13/strata-dsv41'
    pack = strata_pack_path(plan)
    ordinals = list(range(len(plan.gpus)))
    selected = ','.join('GPU-' + g.uuid[4:] for g in plan.gpus)
    native = ['--pack', str(pack), '--gpus', ','.join(map(str, ordinals)), '--max-context', str(plan.context),
              '--ssd-cache', str(args.expert_ram_gb * GIB / 10**9), '--cpu-threads', str(args.cpu_threads),
              '--vram-experts', str(args.gpu_expert_fraction), '--reserve-mib', str(args.vram_reserve_mib),
              '--prefill-chunk', str(args.prefill_chunk), '--dspark', 'on' if plan.spec == 'dspark' else 'none',
              '--experts', '6', '--vision', 'off']
    if plan.spec == 'dspark':
        native += ['--dspark-conf', str(args.dspark_conf if args.dspark_conf is not None else 0.65)]
        if plan.depth:
            native += ['--dspark-block', str(plan.depth)]
    return {'exe': str(binary), 'arch': 'deepseek41', 'args': native, 'cwd': str(source),
            'tokenizer': str(pack / 'tokenizer'), 'model_name': plan.model_id,
            'gpu': ordinals, 'gpu_order': 'as_given', 'log': str(plan.root / 'strata-engine.log'),
            'env': {**({'DS4_SPLIT': args.layer_split} if args.layer_split != 'auto' else {}),
                    'CUDA_VISIBLE_DEVICES': selected, 'DS4_ENGRAM_CACHE_MB': str(args.engram_cache_mib),
                    'DS4_ENGRAM_THREADS': str(args.engram_threads), 'STRATA_DS4_DISK_SKIP_W': '0',
                    'STRATA_DS4_SLOT_KEEP': '0'}, 'lib_dirs': [str(binary.parent)],
            'sampling': {'temperature': 1.0, 'top_p': 1.0}, 'reasoning_effort': plan.effort,
            'reasoning_budget_tokens': args.thinking_budget, 'fit_max_tokens': True,
            'api_monitor': False, 'allowed_hosts': [], 'trusted_origins': [], 'cors_origins': [],
            'engine_ready_s': 3600}


def strata_pack(plan: Plan, args, data: dict, python: Path, source: Path, env: dict) -> None:
    """Build a small index over verified GGUF files. Do not requantize weights."""
    folder = model_folder(plan)
    pack = strata_pack_path(plan)
    if pack.is_symlink() or (pack.exists() and not pack.is_dir()):
        raise Error('The Strata pack must be a regular directory')
    assets = {'ds4.txt', 'tensors.txt', 'engram.bin', 'tokenizer/chat_template.jinja',
              'tokenizer/token_type.json', 'tokenizer/merges.txt', 'tokenizer/vocab.json', 'tokenizer/tokenizer.json'}
    receipt = pack / 'remote-pack.json'
    identity = {'source_revision': STRATA_PINS['source_revision'], 'model_revision': plan.revision,
                'quant': plan.quant, 'model_directory': str(folder)}
    if plan.spec == 'dspark':
        identity['draft'] = STRATA_DSPARK
        identity['draft_path'] = str(strata_draft_path(args))
    if not receipt.is_file():
        if not args.prepare_only:
            raise Error('Strata pack is not prepared. Run the same profile with --prepare-only first.')
        shards = sorted(f['path'] for f in data['files'] if f['kind'] == 'model' and f['path'] != GSQ_ENGRAM_NAME)
        inputs = [folder/name for name in shards]
        model_base = inputs[0].parent
        if any(path.parent != model_base for path in inputs):
            raise Error('The selected GGUF shards must share one directory for the stock packer')
        parents = {model_base.parent}
        if any(list(parent.glob('dspark-*.gguf')) or list(parent.glob('mmproj*.gguf'))
               or (parent/'dspark/dspark.json').exists() or (parent/'vision').exists() for parent in parents):
            raise Error('Optional drafter/vision files beside this model would be auto-discovered by stock pack_gguf. '
                        'Use a model directory without those neighboring artifacts.')
        command = [str(python), '-I', '-B', str(source/'tools/dsv41/pack_gguf.py'),
                   '--model', str(model_base), '--out', str(pack), '--shards', *[path.name for path in inputs]]
        if plan.profile in GSQ_PROFILES:
            command += ['--engram', str(folder/GSQ_ENGRAM_NAME), '--engram-meta', str(folder/shards[0])]
        if plan.spec == 'dspark':
            command += ['--dspark-gguf', str(strata_draft_path(args)), '--dspark-noise-token', '128799',
                        '--dspark-markov-rank', '256']
        run(command, env=env)
        if not all((pack/name).is_file() and not (pack/name).is_symlink() for name in assets):
            raise Error('Stock Strata pack is incomplete; no server was started')
        atomic_json(receipt, {**identity, 'files': {name: file_sha256(pack/name) for name in sorted(assets)}})
    profile_sizes = {40 * 384 * 4, (43 if plan.spec == 'dspark' else 40) * 384 * 4}
    expected_entries = assets | {'remote-pack.json', 'expert-profile.bin', 'expert-profile.bin.tmp'}
    for entry in pack.rglob('*'):
        name = str(entry.relative_to(pack))
        if name == 'expert-profile.bin.tmp':
            try:
                pending = entry.lstat()
            except FileNotFoundError:
                continue  # The native writer completed its atomic rename.
            if not stat.S_ISREG(pending.st_mode) or pending.st_size > max(profile_sizes):
                raise Error('Unexpected Strata temporary expert profile')
            continue
        if entry.is_symlink() or (entry.is_dir() and name != 'tokenizer') or (not entry.is_dir() and (not entry.is_file() or name not in expected_entries)):
            raise Error('Unexpected Strata pack entry: ' + name)
        if name == 'expert-profile.bin' and entry.stat().st_size not in profile_sizes:
            raise Error('Unexpected Strata expert profile size')
    record = json.loads(receipt.read_text())
    if any(record.get(key) != value for key, value in identity.items()) or set(record.get('files', {})) != assets:
        raise Error('Strata pack receipt does not match this source, quant or model directory')
    for name, expected in record['files'].items():
        if (pack/name).is_symlink() or file_sha256(pack/name) != expected:
            raise Error('Strata pack asset changed: ' + name)


def strata_main(plan: Plan, args) -> int:
    """Prepare explicitly, or serve verified local files with outgoing sockets blocked."""
    global LOG
    strata_preflight(plan, args)
    settings = strata_settings(plan, args)
    base, python, source = strata_paths(args)
    command = [str(python), '-I', '-B', str(base/'network_guard.py'), str(python), '-I', '-B',
               str(source/'serve/server.py'), '--engine', 'strata', '--config', str(plan.root/'strata-server.json'),
               '--host', '127.0.0.1', '--port', str(plan.port), '--offline']
    print(json.dumps({'plan': plan.public(), 'server': settings, 'command': command,
                      'runtime_pins': STRATA_PINS, 'outgoing_network': 'blocked by inherited seccomp policy',
                      'hard_ram_limit': None, 'expert_ram_gib': args.expert_ram_gb}, indent=2), flush=True)
    if args.show_config:
        say('Read-only Strata preview complete. No downloads, builds, writes or service changes.')
        return 0
    env = {k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_ARG_', 'GGML_', 'LLAMA_DSV', 'STRATA_', 'MAYA_', 'DS4_'))
           and k not in ('PYTHONPATH', 'PYTHONHOME')}
    env['CUDA_VISIBLE_DEVICES'] = settings['env']['CUDA_VISIBLE_DEVICES']
    if args.install_deps:
        if not args.prepare_only:
            raise Error('Strata --install-deps requires --prepare-only; serving never installs dependencies')
        if os.geteuid() != 0 or not shutil.which('apt-get'):
            raise Error('--install-deps requires root on Debian/Ubuntu')
        run(['apt-get', 'update'], env=dict(env, DEBIAN_FRONTEND='noninteractive'))
        run(['apt-get', 'install', '-y', '--no-install-recommends', 'python3-venv', 'ca-certificates',
             'libseccomp2', 'util-linux'], env=dict(env, DEBIAN_FRONTEND='noninteractive'))
    ctypes.CDLL('libseccomp.so.2')  # Missing enforcement prevents preparation/serving.
    plan.root.mkdir(parents=True, exist_ok=True)
    base.mkdir(parents=True, exist_ok=True)
    with (base/('.assets-check.lock' if args.prepare_assets_only else '.runtime.lock')).open('a') as runtime_lock, (plan.root/('.strata-assets.lock' if args.prepare_assets_only else '.strata-launcher.lock')).open('a') as lock:
        try:
            fcntl.flock(runtime_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Error('Another Strata preparation or server owns this runtime/profile') from exc
        run([sys.executable, '-I', '-B', '-c', STRATA_INSTALL_CODE, json.dumps(STRATA_PINS), str(base),
             'prepare' if args.prepare_only and not args.prepare_assets_only else 'check'], env=env, quiet=True)
        guard = base/'network_guard.py'
        if args.prepare_only and not args.prepare_assets_only:
            guard.write_text(STRATA_NETWORK_GUARD)
        elif not guard.is_file() or guard.read_text() != STRATA_NETWORK_GUARD:
            raise Error('Strata network guard is missing or changed; run --prepare-only')
        weights_plan = dataclasses.replace(plan, spec='none', depth=0, draft_bytes=0)
        path = plan.root / f'manifest-{plan.quant}-{weights_plan.spec}-{plan.revision[:12]}.json'
        if plan.profile in GSQ_PROFILES:
            data = dict(repo=plan.repo, revision=plan.revision, quant=plan.quant, spec='none', official_files=[],
                        files=[dict(path=n, size=f['size'], sha256=f['sha256'], kind='model')
                               for n,f in sorted(gsq_files(plan.quant).items())])
            validate_manifest(weights_plan, data)
            if args.prepare_only and not args.prepare_assets_only:
                atomic_json(path, data)
        elif path.is_file():
            data = json.loads(path.read_text())
            validate_manifest(weights_plan, data)
        elif args.prepare_only:
            hub = hub_python(args, env)
            path, data = manifest(hub, weights_plan, env)
        else:
            raise Error('Pinned GGUF manifest is missing. Run --prepare-only first.')
        missing = any(not (model_folder(plan)/f['path']).is_file()
                      or (model_folder(plan)/f['path']).stat().st_size != f['size'] for f in data['files'])
        if missing and not args.prepare_only:
            raise Error('Pinned GGUF files are missing. Run --prepare-only; serving does not download.')
        if missing and not path.is_file():
            atomic_json(path, data)
        hub = hub_python(args, env) if missing else python
        download(hub, weights_plan, path, data, args, env)
        if plan.spec == 'dspark':
            strata_draft(args)
        strata_pack(plan, args, data, python, source, env)
        if args.prepare_assets_only:
            say('Strata assets prepared. Runtime, guard, active server configuration and launch record were not changed.')
            return 0
        atomic_json(plan.root/'strata-server.json', settings)
        atomic_json(plan.root/'last-launch.json', {'launcher':VERSION, **plan.public(), 'command':command,
                    'server':settings, 'hard_ram_limit':None, 'qualification':'Remote GPU inference not validated'})
        if args.prepare_only:
            say('Strata runtime, GGUF checks and stock pack complete. No model loaded or server started.\n' + shlex.join(command))
            return 0
        LOG = (Path(args.root).expanduser().absolute()/'start.log').open('a', buffering=1)
        ssh_setup(args)
        say(f'Strata API: http://127.0.0.1:{plan.port}/v1; server log: {plan.root/"server.log"}')
        return serve(command, plan, env)


def main(argv=None) -> int:
    global LOG
    if sys.version_info < (3,10):
        raise Error('Python 3.10+ is required')
    args=parser().parse_args(argv)
    args.explicit_flags = {value.split('=', 1)[0] for value in (sys.argv[1:] if argv is None else argv) if value.startswith('--')}
    if args.prepare_assets_only and not args.prepare_only:
        raise Error('--prepare-assets-only requires --prepare-only')
    if args.prepare_assets_only and (args.smoke_test or args.smoke_full_context or args.ssh_only or args.setup_ssh):
        raise Error('--prepare-assets-only cannot be combined with smoke tests or SSH setup')
    minimax_accepted = ('--accept-experimental' in (sys.argv[1:] if argv is None else argv)
                        or enabled('MINIMAX_ACCEPT_EXPERIMENTAL'))
    if not 1024<=args.port<=65535 or not 1<=args.ssh_port<=65535:
        raise Error('Invalid API/SSH port')
    if not 1<=args.build_jobs<=128 or not 1<=args.ubatch<=args.batch<=32768:
        raise Error('Invalid build-jobs or batch/ubatch setting')
    if not 1<=args.download_workers<=32 or not 1<=args.verify_workers<=16:
        raise Error('download-workers must be 1..32; verify-workers 1..16')
    if not 1<=args.cpu_threads<=256:
        raise Error('cpu-threads must be 1..256')
    if not 0<=args.cache_ram<=262144 or not 128<=args.smoke_tokens<=32768:
        raise Error('Invalid host-cache/smoke token budget')
    if args.archive_sha256 and not re.fullmatch(r'[a-fA-F0-9]{64}', args.archive_sha256):
        raise Error('--archive-sha256 must contain 64 hexadecimal characters')
    if args.archive_sha256:
        args.archive_sha256 = args.archive_sha256.lower()
    if args.archive_build_only and args.no_build_archive:
        raise Error('--archive-build-only conflicts with --no-build-archive')
    if args.gpu_count is not None and not 1<=args.gpu_count<=64:
        raise Error('Invalid --gpu-count')
    if args.ds_source == 'gsq-rco' and (args.ssh_only or args.setup_ssh):
        raise Error('GSQ-RCO config inspection never starts SSH services')
    if args.show_config and (args.ssh_only or args.setup_ssh):
        raise Error('--show-config never sets up SSH; run SSH setup separately')
    if args.smoke_full_context and not (0.50 <= args.smoke_fill <= 0.97):
        raise Error('--smoke-fill must be 0.50..0.97; leave room for generated tokens and chat history')
    if not 128 <= args.smoke_decode_tokens <= 8192:
        raise Error('--smoke-decode-tokens must be 128..8192')
    if not 0 <= args.smoke_soak_minutes <= 240:
        raise Error('--smoke-soak-minutes must be 0..240')
    if not 30 <= args.smoke_timeout_seconds <= 21600 or not 1 <= args.smoke_gpu_sample_seconds <= 60:
        raise Error('Invalid stress timeout or GPU sample interval')
    if not 1 <= args.smoke_max_minutes <= 1440:
        raise Error('--smoke-max-minutes must be 1..1440')
    if not 0 <= args.smoke_min_free_vram_mib <= 262144:
        raise Error('Invalid --smoke-min-free-vram-mib')
    if args.smoke_test or args.smoke_full_context:
        smoke(args.port,args.smoke_tokens)
        if args.smoke_full_context:
            return smoke_full_context(args)
        return 0
    if args.ssh_only:
        ssh_setup(args)
        return 0
    gpus=visible_gpus(inventory(),cuda_inventory())
    mode,chosen=select(gpus,args.profile,args.gpu_count)
    if mode=='flash-next-fp4':
        if args.show_config:
            print(json.dumps({'profile':mode,'status':'blocked','reason':FP4_BLOCK},indent=2))
            return 0
        raise Error(FP4_BLOCK)
    if mode.startswith('deepseek'):
        return strata_main(strata_plan(args, mode, chosen), args)
    native_flags = {'--reuse', '--layer-split', '--dspark-conf', '--prepare-assets-only', '--expert-ram-gb', '--gpu-expert-fraction', '--vram-reserve-mib', '--prefill-chunk',
                    '--engram-cache-mib', '--engram-threads', '--thinking-budget'}
    if native_flags.intersection(args.explicit_flags):
        raise Error('Strata cache, prefill and thinking-budget controls require a DeepSeek profile')
    if args.ds_source == 'gsq-rco' and mode not in GSQ_PROFILES:
        config = gsq_config(args,mode,chosen)
        print(json.dumps(config,indent=2),flush=True)
        if args.show_config:
            say('Read-only GSQ-RCO config inspection; launch remains blocked pending a verified compatible runtime.')
            return 0
        raise Error(GSQ_BLOCK)
    plan=make_plan(args,mode,chosen)
    if args.reuse_engram and (mode not in ('deepseek-b300-iq3', 'deepseek-h200-dual-iq3')
                              or plan.quant not in Q6_ENGRAM_SHARDS):
        raise Error('--reuse-engram applies only to the IQ3_S/IQ3_XXS B300 and dual-H200 profiles')
    if args.engram_ram and mode not in GSQ_PROFILES and mode not in ('deepseek','deepseek-h200-iq2','deepseek-h200-dual-iq3','deepseek-b300-iq3', 'deepseek-b200-iq2', 'deepseek-pro6000-quad-q8'):
        raise Error('--engram-ram is only supported by the llama.cpp DeepSeek profiles')
    if args.gsq_engram is not None and mode not in GSQ_PROFILES:
        raise Error('--gsq-engram only applies to dedicated deepseek-gsq-* profiles')
    if args.tensor_split is not None:
        try:
            weights = [float(x.strip()) for x in args.tensor_split.split(',')]
        except ValueError as exc:
            raise Error('--tensor-split must contain comma-separated positive numbers') from exc
        if (len(chosen) <= 1 or len(weights) != len(chosen)
                or not all(math.isfinite(x) and x > 0 for x in weights)):
            raise Error(f'--tensor-split needs {len(chosen)} positive finite weights on a multi-GPU profile')
        say('Layer split weights (relative, not exact allocations): ' + args.tensor_split)
    for g in chosen:
        say(f'GPU {g.index}: {g.name}, {g.free/1024:.2f}/{g.total/1024:.2f} GiB free, SM {g.cc}, {g.uuid}')
    print(json.dumps(plan.public(),indent=2),flush=True)
    if args.archive_build_only:
        if mode in GSQ_PROFILES and not ('--accept-experimental' in (sys.argv[1:] if argv is None else argv) or enabled('GSQ_ACCEPT_EXPERIMENTAL')):
            raise Error('GSQ archive build needs explicit --accept-experimental for its pinned loader adapter')
        if mode.startswith('minimax-') and not minimax_accepted:
            raise Error('MiniMax archive build requires --accept-experimental (pinned all-60-layer patch)')
        if mode.startswith('deepseek') and mode not in GSQ_PROFILES and not args.accept_experimental:
            raise Error('DeepSeek archive build requires --accept-experimental (pinned draft PR)')
        install_dependencies(args, mode)
        plan.root.mkdir(parents=True, exist_ok=True)
        env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_ARG_', 'GGML_', 'LLAMA_DSV'))}
        env.update(CUDA_VISIBLE_DEVICES=','.join('GPU-'+g.uuid[4:] for g in chosen))
        binary,_=build(plan,args,env)
        say(f'Build archive preparation complete: {binary}. No model files touched or server started.')
        return 0
    preflight(plan,args)
    if len(chosen)<len(gpus):
        say(f'Using {len(chosen)} of {len(gpus)} CUDA-visible GPUs; no allocation is combined across hosts.')
    if args.show_config:
        if mode in GSQ_PROFILES:
            target=Path(sorted(n for n in gsq_files(plan.quant) if n!=GSQ_ENGRAM_NAME)[0])
            say('Planned command (not executed): '+shlex.join(command_for(plan,Path('llama-server'),target,None,None,args)))
        if mode.startswith('minimax-'):
            preview = command_for(plan, Path('llama-server'), Path(sorted(MINIMAX_FILES)[0]), Path('MiniMax-M3.1-Flash-DFlash2-F16.gguf') if plan.spec == 'dflash' else None, None, args)
            say('Planned command (not executed): ' + shlex.join(preview))
        say('Read-only check complete. No network access, downloads, writes, builds or service changes.')
        return 0
    if mode in GSQ_PROFILES and not ('--accept-experimental' in (sys.argv[1:] if argv is None else argv) or enabled('GSQ_ACCEPT_EXPERIMENTAL')):
        raise Error('GSQ requires explicit --accept-experimental for its pinned external-Engram loader adapter; full GPU inference is untested')
    if mode.startswith('minimax-') and not minimax_accepted:
        raise Error('MiniMax requires --accept-experimental for the exact all-60-layer sparse-attention patch; GPU inference is unvalidated')
    if mode.startswith('deepseek') and mode not in GSQ_PROFILES and not args.accept_experimental:
        raise Error('DeepSeek needs experimental pinned llama.cpp PR #30817. '
                    'Re-run with --accept-experimental; B300 and H200 paths are unverified.')
    install_dependencies(args, mode)
    plan.root.mkdir(parents=True,exist_ok=True)
    common_root=Path(args.root).expanduser().absolute()
    common_root.mkdir(parents=True,exist_ok=True)
    LOG=(common_root/'start.log').open('a',buffering=1)
    say(f'Launcher {VERSION}; selected {plan.model_id}; other deployments/builds are not modified.')
    with (common_root/f'.port-{plan.port}.lock').open('a') as port_lock, (plan.root/'.launcher.lock').open('a') as lock:
        try:
            fcntl.flock(port_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise Error('Another launcher/server owns this profile or API port') from e
        env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_ARG_', 'GGML_', 'LLAMA_DSV'))}
        env.update(CUDA_VISIBLE_DEVICES=','.join('GPU-'+g.uuid[4:] for g in chosen),
                   HF_HOME=str(common_root/'hf-home'),HF_HUB_DISABLE_TELEMETRY='1',
                   HF_XET_CHUNK_CACHE_SIZE_BYTES='0')
        # Check the toolchain before paying to download hundreds of GB.
        binary,_ = build(plan,args,env)
        python=hub_python(args,env)
        manifest_path,data=manifest(python,plan,env)
        actual_weights=sum(f['size'] for f in data['files'] if f['kind']=='model')
        actual_draft=sum(f['size'] for f in data['files'] if f['kind']=='draft')
        preflight(plan,args,actual_weights,actual_draft)
        target,draft,template=download(python,plan,manifest_path,data,args,env)
        if mode.startswith('minimax-') and plan.spec == 'dflash':
            draft = prepare_minimax_dflash(python,plan,binary,args,env)
            actual_draft = draft.stat().st_size
        command=command_for(plan,binary,target,draft,template,args)
        atomic_json(plan.root/'last-launch.json',{'launcher':VERSION,**plan.public(),'command':command,
                    'actual_model_file_bytes':actual_weights,'actual_draft_file_bytes':actual_draft,
                    'qualification':'Static/offline tests only; not GPU-inference-tested by the author'})
        if args.prepare_only:
            say('Build/download/checks complete; server not started.\n'+shlex.join(command))
            return 0
        # Other workloads may have claimed memory during a long model download.
        fresh={g.uuid:g for g in inventory()}
        if any(g.uuid not in fresh for g in chosen):
            raise Error('A selected GPU disappeared during preparation')
        plan.gpus=[dataclasses.replace(fresh[g.uuid],cc=g.cc) for g in chosen]
        preflight(plan,args,actual_weights,actual_draft)
        ssh_setup(args)
        say(f'When ready: http://127.0.0.1:{plan.port}/v1 ; model {plan.model_id}')
        say(f'Server log: {plan.root / "server.log"}. Use tmux/screen to survive SSH disconnects.')
        return serve(command,plan,env)


def interrupt(_sig,_frame):
    raise KeyboardInterrupt

if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        say("Interrupted; this launcher's child is stopped, downloads retained for resume.")
        raise SystemExit(130)
    except (Error,OSError,ValueError,KeyError,TypeError,subprocess.TimeoutExpired) as exc:
        print('[start-remote-model] ERROR: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
PYTHON_START_REMOTE_MODEL

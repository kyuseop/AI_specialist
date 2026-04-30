# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import argparse
import numpy as np
import matplotlib.pyplot as plt


#####################################
# Gated DeltaNet 설명
#####################################
# Gated DeltaNet은 Qwen3-Next와 Kimi Linear에서 사용하는 선형 어텐션(Linear Attention) 기법입니다.
# 기존 Transformer의 O(n²) 복잡도를 O(n)으로 줄입니다.
#
# ============================================
# 일반 Attention vs Gated DeltaNet 비교
# ============================================
#
# 일반 Attention (Quadratic):
#   모든 토큰이 모든 토큰을 참조 → O(n²)
#   softmax(Q @ K.T) @ V → n×n 어텐션 행렬 생성
#   KV 캐시: batch × n_tokens × n_heads × d_head × 2
#
# Gated DeltaNet (Linear):
#   토큰을 하나씩 순차 처리 → O(n)
#   고정 크기 메모리 상태(S)를 업데이트하는 RNN 방식
#   상태: batch × n_heads × d_head × d_head (컨텍스트 길이와 무관!)
#
# ============================================
# 핵심 아이디어: Delta Rule + Gating
# ============================================
#
#         입력 x
#            │
#  ┌─────────┼─────────┬─────────┐
#  ▼         ▼         ▼         ▼
# W_Q       W_K       W_V      W_gate
#  │         │         │         │
#  ▼         ▼         ▼         ▼
#  Q         K         V       gate
#  │         │         │
#  └────┬────┴────┬────┘
#       │         │
#       ▼         ▼
#  ┌────────────────────┐
#  │  Delta Rule 상태   │  ← 고정 크기 메모리 S (d_head × d_head)
#  │  업데이트 (RNN)    │
#  │                    │
#  │  S = S × α (decay) │  ← α: 이전 메모리 얼마나 유지?
#  │  delta = (v - kv_mem) × β │  ← β: 새 정보 얼마나 반영?
#  │  S = S + k × delta │
#  └────────────────────┘
#            │
#            ▼
#       RMSNorm + SiLU(gate)
#            │
#            ▼
#          출력
#
# ============================================
# 핵심 게이트
# ============================================
# | 게이트 | 역할 |
# |--------|------|
# | α (decay gate) | 이전 메모리를 얼마나 유지할지 (망각률) |
# | β (update gate) | 새 입력이 상태를 얼마나 수정할지 |
# | output gate | 최종 출력을 얼마나 통과시킬지 (SiLU) |
#
# ============================================
# 핵심 코드 (Delta Rule)
# ============================================
# S = zeros(batch, num_heads, head_dim, head_dim)  # 고정 크기 상태
#
# for t in range(num_tokens):  # 선형 복잡도!
#     S = S * alpha_t                    # 1. 메모리 감쇠
#     kv_mem = (S * k_t).sum()           # 2. 현재 키로 메모리 조회
#     delta = (v_t - kv_mem) * beta_t    # 3. Delta 계산: 새 값 - 예측 값
#     S = S + k_t * delta                # 4. 메모리 업데이트
#     y_t = (S * q_t).sum()              # 5. 쿼리로 출력 생성
#
# ============================================
# KV 캐시 메모리 비교 (이 스크립트의 핵심)
# ============================================
#
# MHA KV 캐시:
#   batch × n_tokens × n_heads × d_head × 2
#   → 컨텍스트 길이에 선형 증가 📈
#
# DeltaNet 상태:
#   batch × n_heads × d_head × d_head
#   → 컨텍스트 길이와 무관! (고정) 📊
#
# | 컨텍스트 길이 | MHA KV 캐시 | DeltaNet 상태 |
# |--------------|-------------|---------------|
# | 1K tokens    | 증가        | 고정          |
# | 100K tokens  | 100× 증가   | 동일          |
# | 1M tokens    | 1000× 증가  | 동일          |
#
# ============================================
# 3:1 하이브리드 구조 (Qwen3-Next, Kimi Linear)
# ============================================
#
# Layer 0: DeltaNet (선형)
# Layer 1: DeltaNet (선형)
# Layer 2: DeltaNet (선형)
# Layer 3: Full Attention (전체 컨텍스트 참조)
# Layer 4: DeltaNet
# Layer 5: DeltaNet
# Layer 6: DeltaNet
# Layer 7: Full Attention
# ...
#
# 이유: DeltaNet은 고정 크기 메모리(S)로 컨텍스트를 압축하므로 정보 손실 가능.
#       일부 레이어에서 Full Attention을 사용해 전체 컨텍스트 모델링 능력 보완.
#
# ============================================
# Trade-off
# ============================================
# | 장점                        | 단점                           |
# |-----------------------------|--------------------------------|
# | O(n) 선형 복잡도            | 전역 컨텍스트 모델링 제한      |
# | KV 캐시 불필요 (고정 상태)  | RNN처럼 병렬화 어려움 (학습 시)|
# | 매우 긴 컨텍스트 효율적     | 메모리 병목 (정보 압축)        |
#
# ============================================
# 사용처
# ============================================
# - Qwen3-Next: Gated DeltaNet + Gated Attention (3:1)
# - Kimi Linear: KDA (채널별 게이팅) + MLA (3:1)

# 데이터 타입별 바이트 수
DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def kv_bytes_total_mha(batch, context_length, emb_dim, n_layers, bytes_per_elem, n_heads):
    """
    MHA (Multi-Head Attention) KV 캐시 메모리 계산

    KV 캐시 공식:
      batch × context_length × n_heads × d_head × 2 × bytes_per_elem
                                                   ↑
                                              K와 V 두 개

    예: batch=1, context=1024, emb=2048, heads=16, bf16
      d_head = 2048 / 16 = 128
      per_layer = 1 × 1024 × 16 × 128 × 2 × 2 = 8,388,608 bytes (8MB)

    → 컨텍스트 길이에 비례하여 선형 증가!
    """
    d_head = emb_dim // n_heads
    per_layer = batch * context_length * n_heads * d_head * 2 * bytes_per_elem
    return per_layer * n_layers


def kv_bytes_total_deltanet_no_conv(batch, emb_dim, n_layers, bytes_per_elem, n_heads):
    """
    Gated DeltaNet 상태 메모리 계산 (Convolutional mixing 제외 단순 버전)

    상태 S 공식:
      batch × n_heads × d_head × d_head × bytes_per_elem
                        ↑        ↑
                     d_head × d_head 고정 크기 상태 행렬

    예: batch=1, emb=2048, heads=16, bf16
      d_head = 2048 / 16 = 128
      per_layer = 1 × 16 × 128 × 128 × 2 = 524,288 bytes (0.5MB)

    → 컨텍스트 길이와 무관! 항상 고정 크기!
    → context_length 파라미터가 없음에 주목
    """
    d_head = emb_dim // n_heads
    per_layer = batch * n_heads * d_head * d_head * bytes_per_elem
    return per_layer * n_layers


def gb(x):
    return x / 1e9


def main():
    p = argparse.ArgumentParser(description="Memory vs. Context Length: MHA vs. DeltaNet (3:1 mix)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--emb_dim", type=int, default=2048)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--n_layers", type=int, default=48)
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="bf16")
    p.add_argument("--min_ctx", type=int, default=128)
    p.add_argument("--max_ctx", type=int, default=131_072)
    args = p.parse_args()

    step = 100
    ctx = np.arange(args.min_ctx, args.max_ctx + 1, step, dtype=int)
    bytes_per_elem = DTYPE_BYTES[args.dtype]

    ####################################################
    # 1) Full Attention (MHA) - 모든 레이어가 MHA
    ####################################################
    # KV 캐시가 컨텍스트 길이에 따라 선형 증가
    # 그래프에서 가파른 직선으로 표시됨
    mha_bytes = np.array([
        kv_bytes_total_mha(args.batch, int(t), args.emb_dim, args.n_layers,
                           bytes_per_elem, args.n_heads)
        for t in ctx
    ], dtype=float)

    ####################################################
    # 2) DeltaNet Only - 모든 레이어가 DeltaNet
    ####################################################
    # 상태 S는 컨텍스트 길이와 무관하게 고정!
    # 그래프에서 수평선으로 표시됨
    dnet_bytes_const = kv_bytes_total_deltanet_no_conv(
        args.batch, args.emb_dim, args.n_layers,
        bytes_per_elem, args.n_heads
    )
    dnet_bytes = np.full_like(mha_bytes, fill_value=dnet_bytes_const, dtype=float)

    ####################################################
    # 3) 3:1 Hybrid - Qwen3-Next, Kimi Linear 방식
    ####################################################
    # 48 레이어 기준:
    #   - MHA 레이어: 48 / 4 = 12개 (매 4번째 레이어)
    #   - DeltaNet 레이어: 48 - 12 = 36개 (나머지)
    #
    # 메모리 = MHA KV 캐시 (12 레이어) + DeltaNet 상태 (36 레이어)
    #        = 컨텍스트 비례 부분 + 고정 부분
    #
    # 그래프에서 완만한 직선으로 표시됨 (기울기가 MHA의 1/4)
    n_mha_layers = args.n_layers / 4      # 1/4은 Full Attention
    n_dnet_layers = args.n_layers - n_mha_layers  # 3/4은 DeltaNet
    mix_bytes = np.array([
        kv_bytes_total_mha(args.batch, int(t), args.emb_dim, n_mha_layers,
                           bytes_per_elem, args.n_heads)
        + kv_bytes_total_deltanet_no_conv(args.batch, args.emb_dim, n_dnet_layers,
                                          bytes_per_elem, args.n_heads)
        for t in ctx
    ], dtype=float)

    # Convert to GB
    mha_gb = gb(mha_bytes)
    dnet_gb = gb(dnet_bytes)
    mix_gb = gb(mix_bytes)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ctx, mha_gb, label="Full Attention (MHA) KV cache")
    ax.plot(ctx, dnet_gb, label="All Gated DeltaNet (no conv)")
    ax.plot(ctx, mix_gb, label="3:1 layer ratio (3 DeltaNet : 1 Full Attention)")

    ax.set_xlabel("Context length (number of tokens)")
    ax.set_ylabel("KV cache size (GB)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    fig.tight_layout()
    plt.savefig("deltanet_memory_plot.pdf", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

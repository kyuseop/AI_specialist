# 이 파일은 3~4장에서 다룬 모든 관련 코드를 모아둔 것입니다.
# 독립적인 스크립트로 바로 실행할 수 있습니다.

import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# 3장: 멀티 헤드 어텐션 (Multi-Head Attention)
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 출력 차원은 헤드 수로 정확히 나누어 떨어져야 합니다.
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # 각 어텐션 헤드가 담당할 임베딩 차원의 크기입니다.
        self.head_dim = d_out // num_heads  

        # 쿼리(Query), 키(Key), 밸류(Value)를 만들기 위한 선형 변환 레이어
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # 여러 헤드의 결과를 합친 후 마지막으로 통과시키는 선형 레이어
        self.out_proj = nn.Linear(d_out, d_out)  
        self.dropout = nn.Dropout(dropout)
        
        # 미래의 토큰을 보지 못하게 가리는 인과적 마스크(Causal Mask) 생성 (상삼각행렬)
        # persistent=False는 모델 가중치 저장 시 이 버퍼를 무시하라는 의미입니다.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False
        )

        ####################################################
        # 신규 추가: KV 캐시 버퍼 및 위치 포인터 (항상 사용)
        self.register_buffer("cache_k", None, persistent=False) # Key 캐시
        self.register_buffer("cache_v", None, persistent=False) # Value 캐시
        self.ptr_current_pos = 0 # 현재 생성 중인 시퀀스의 위치(인덱스)를 추적
        ####################################################

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 입력 x로부터 새로운 Key, Value, Query를 계산합니다.
        keys_new = self.W_key(x)  
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # 여러 헤드로 연산을 나누기 위해 텐서의 형태를 변경합니다.
        # (배치 크기, 토큰 수, 전체 차원) -> (배치 크기, 토큰 수, 헤드 수, 헤드당 차원)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # 항상 캐시 적용 (ALWAYS CACHE)
        if self.cache_k is None:
            # 첫 번째 스텝(프롬프트 처리)에서는 캐시를 새로 계산한 값으로 초기화합니다.
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            # 이어지는 스텝에서는 기존 캐시에 새로 계산된 Key, Value를 이어붙입니다(Concatenate).
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        
        # 이번 연산에 사용할 최종 Key와 Value는 누적된 캐시 전체입니다.
        keys, values = self.cache_k, self.cache_v
        ####################################################

        # 내적 연산을 위해 차원 위치를 바꿉니다 (Transpose).
        # -> (배치 크기, 헤드 수, 토큰 수, 헤드당 차원)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 스케일 점곱 어텐션 연산 (Query 행렬과 Key 행렬의 내적)
        attn_scores = queries @ keys.transpose(2, 3) 

        ####################################################
        # 항상 캐시된 마스크 로직 사용
        num_tokens_Q = queries.shape[-2] # 이번 스텝에서 처리할 쿼리(새 토큰)의 개수
        num_tokens_K = keys.shape[-2]    # 지금까지 누적된 모든 키(캐시 포함)의 개수
        
        # 현재 위치에 맞게 마스크를 잘라옵니다.
        mask_bool = self.mask.bool()[
            self.ptr_current_pos : self.ptr_current_pos + num_tokens_Q, 
            :num_tokens_K
        ]
        # 다음 스텝을 위해 위치 포인터를 업데이트합니다.
        self.ptr_current_pos += num_tokens_Q
        ####################################################

        # 마스크에서 True인 위치(미래의 토큰)를 -무한대(-torch.inf)로 채워 
        # 소프트맥스 후 0이 되게 만듭니다.
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 어텐션 스코어를 스케일링(헤드 차원의 제곱근으로 나눔) 후 소프트맥스 적용
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 어텐션 가중치와 Value를 곱하여 문맥 벡터(Context Vector) 생성
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 분리되었던 여러 헤드의 결과를 하나로 다시 합칩니다.
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 최종 선형 투영

        return context_vec

    # 텍스트 생성을 새로 시작할 때 캐시를 비워주는 함수
    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


#####################################
# 4장: 트랜스포머 아키텍처 구성 요소
#####################################

# 레이어 정규화 (Layer Normalization)
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) # 학습 가능한 스케일 파라미터 (Gamma)
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # 학습 가능한 이동 파라미터 (Beta)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# GELU 활성화 함수
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# 피드포워드 신경망 (FeedForward Network)
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # 차원을 4배로 확장
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), # 원래 차원으로 축소
        )

    def forward(self, x):
        return self.layers(x)


# 트랜스포머 블록 (Attention + FeedForward 결합)
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 어텐션 블록의 잔차 연결(Shortcut connection / Residual connection)
        shortcut = x
        x = self.norm1(x)
        
        # 어텐션 통과 (이제 항상 내부에 저장된 캐시를 활용합니다)
        x = self.att(x)

        x = self.drop_shortcut(x)
        x = x + shortcut  # 원본 입력을 다시 더해줌 (잔차 연결)

        # 피드포워드 블록의 잔차 연결
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 원본 입력을 다시 더해줌

        return x


# 최종 GPT 모델 구조
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 토큰 임베딩 (단어 -> 벡터)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 위치 임베딩 (위치 정보 -> 벡터)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 여러 개의 트랜스포머 블록을 쌓음
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 전체 모델 레벨에서 현재 위치를 추적 (위치 임베딩용)
        self.current_pos = 0 

        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 최종 출력 헤드 (임베딩 벡터 -> 다음 단어의 확률값(Logits) 변환)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        ####################################################
        # 항상 위치 정보를 캐시 상황에 맞게 계산
        # 캐시가 쌓인 상황에서는 0부터가 아니라 current_pos부터 위치를 매겨야 함
        pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
        self.current_pos += seq_len # 위치 업데이트
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        ####################################################

        x = tok_embeds + pos_embeds  # 토큰 임베딩과 위치 임베딩을 더함
        x = self.drop_emb(x)

        # 모든 트랜스포머 블록 순차적 통과
        for blk in self.trf_blocks:
            x = blk(x)

        x = self.final_norm(x)
        logits = self.out_head(x) # 각 단어가 다음에 올 확률의 로짓값 예측
        return logits

    # 새 문장을 생성하기 전에 모든 어텐션 층의 캐시를 초기화
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0


####################################################
# 텍스트 생성 함수 (항상 캐시 사용)
def generate_text_simple_cached(model, idx, max_new_tokens, context_size=None):
    model.eval() # 평가(추론) 모드 설정 (드롭아웃 비활성화)
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad(): # 그래디언트 계산 비활성화 (메모리 및 속도 최적화)
        # 생성 시작 전, 이전 데이터가 남아있지 않도록 캐시 완벽 초기화
        model.reset_kv_cache()
        
        # 1. 프롬프트 전체(최초 입력 문장)를 모델에 넣어 캐시를 Pre-fill(사전 채우기) 합니다.
        logits = model(idx[:, -ctx_len:])

        for _ in range(max_new_tokens):
            # 2. 방금 나온 출력에서 마지막 위치(-1)의 단어 중 가장 확률이 높은 것(argmax)을 고릅니다.
            next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
            
            # 3. 고른 새 단어를 기존 문장 끝에 이어붙여 최종 출력물을 만듭니다.
            idx = torch.cat([idx, next_idx], dim=1)
            
            # 4. 다음 스텝에서는 전체 문장이 아니라 '방금 뽑은 새 단어(next_idx)' 하나만 모델에 넣습니다.
            # (나머지 정보는 이미 model 안의 캐시에 다 저장되어 있으므로 연산이 획기적으로 줄어듦)
            logits = model(next_idx)

    return idx
####################################################


def main():
    # 124M 파라미터를 가진 소형 GPT 모델 설정
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 어휘 사전 크기 (GPT-2 기준)
        "context_length": 1024,  # 최대 문맥 길이
        "emb_dim": 768,          # 임베딩 벡터의 차원 수
        "n_heads": 12,           # 어텐션 헤드의 개수
        "n_layers": 12,          # 트랜스포머 블록(레이어)의 개수
        "drop_rate": 0.1,        # 드롭아웃 비율
        "qkv_bias": False        # Q, K, V 레이어에 편향(Bias) 사용 여부
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 추론 모드 (드롭아웃 비활성화)

    start_context = "Hello, I am"

    # OpenAI의 tiktoken 라이브러리를 사용하여 텍스트를 토큰 ID로 인코딩
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0) # 배치 차원 추가

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize() # 정확한 시간 측정을 위한 동기화
    start = time.time()

    ####################################################
    # 수정된 캐시 기반 텍스트 생성 함수 호출
    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200, # 새로 생성할 토큰의 최대 개수
    )
    ####################################################

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    # 토큰 ID들을 다시 사람이 읽을 수 있는 텍스트로 디코딩
    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    
    # GPU 메모리 사용량 확인 (학습과 달리 추론 시 캐시 메모리가 얼마나 차지하는지 확인 가능)
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
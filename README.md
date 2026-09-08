# Yaoyao — 桃注意力 CPU 解码 / 原生 CUDA 训练

当前D256、1024词表、320维SwiGLU预测头的小型研究模型。可逆链保留16个token ID，读取器混合17个链状态；服务使用64-token窗口。无QKV/KV cache，不宣称无限上下文。

## 源码
- tao_d256_api.cpp：CPU前向、采样、持久进程协议。
- yaoyao_api.py/openai_client.py：OpenAI兼容服务与计时，Python不参与训练。
- train_tao_fixed_boundaries.cu：固定边界预测头训练。
- train_tao_context_supported.cu：Wbi门训练。
- train_tao_state_gates.cu/tao_state_gates.hpp：状态门、SHA/CRC、联合最多两个历史距离。
- eval_tao_state_gates.cpp、tao_ordered_diagnostics.cpp：评估与有序诊断。
- self_decoding_node.hpp、inverse_reader.hpp、inverse_reader_cuda.cuh：链和读取器。
- yaoyao_v21_full.cpp、yaoyao_v21_cuda_train_stable_ce.cu、test_integrated_inverse_reader.cpp 是当前代码实际依赖，不是废弃副本。

## 构建
Windows/MSVC x64/AVX2。运行build_project.bat，将CPU程序输出到build目录。GPU构建用build_tao_state_gates.bat，需要CUDA12.6/cuBLAS，sm75目标。

从项目根目录运行build/test_tao_server_checkpoint.exe tao_fixed_step50002.bin及build/test_tao_server_forward.exe。Python服务依赖在requirements.txt。在线API仍使用已验证的根目录tao_d256_api_state.exe；构建不会自动替换运行中二进制。

## 本地资产（不上传Git）
- tinystories_train.txt：原文和词表来源。
- experiments/d256_nibble64_baseline/train_tokens.bin、eval_tokens.bin和multislice/slice_0..7.bin：原训练/评估数据，字节和划分保持不变。
- tao_fixed_step50002.bin及tao_alternating_step47002.bin：正式模型和验证父模型。
- tao_coef_rts1_128.reader、tao_context_supported.tcg、tao_state_supported_v2_full.tds：正式读取器/门。

缺少资产时需自行提供具有匹配SHA256的文件，不允许随机替代。tokens格式uint32计数+int32 IDs。模型v4由源码加载器定义。

## 结果边界
见TAO_STATE_V2_RESULT.md、TAO_ORDERED_DIAGNOSTICS.md。NLL小幅改善不保证top1或故事质量，滑动窗口第三切片略退化；合成检索接近随机，文本连贯性仍弱。

## 清理
旧版本、无效实验、日志、重复构建产物已移到项目外可恢复归档。保留完整传递include依赖，不重写Git历史。清理回执CLEANUP_RECEIPT.json仅本地保存。

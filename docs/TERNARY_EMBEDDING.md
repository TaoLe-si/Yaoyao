# Independent scalable ternary token embedding — TTE1

Implemented tao_ternary_embedding.hpp on shared tao_ternary.hpp. Every token has its own packed ternary row and finite positive FP32 scale. Token IDs uint32, not quantized values. Zero-initialized code rows,scale1 until caller explicitly sets them; no learned embedding claim. Tested V8192 is a fixture,NOT final vocabulary decision.

## API and format
set_row validates ID,dimension,trits,scale before mutation. lookup returns scaled float vector;codes returns unscaled trits. These are distinct interfaces:scaled embedding must NOT be silently inserted into mod3 state.

bytes()/load() serialize/deserialize an in-memory versioned byte buffer (not filesystem convenience APIs). Little-endian:magicTTE1 u32,version1 u32,V u32,D u32,rowStride u32,tokenizerSHA25632bytes, V IEEE754 FP32 scales, V row-aligned packed ternary rows,CRC32 u32. Total56+4V+V*ceil(D/4). 00=0,01=+1,10=-1,11 invalid;zero padding bits.

Loader checks version,dimensions,exact byte length,row stride,expected tokenizer digest,CRC,all scales,codes and padding. Dimensions limited V<=1048576,D<=65536,packed storage<=1GiB. CRC is corruption detection,not authentication. Caller supplies expected tokenizer SHA; no tokenizer hashing/config pipeline implemented in this module.

## Actual verification
CPU test_tao_ternary_embedding:V8192,D257; IDs0,1,1023,1024,8191;lookup/scales/code roundtrip,serialized canonical roundtrip565304bytes;invalid IDs/rows/trits/scales,wrong tokenizer,version,length,CRC;recomputed-CRC malformed Inf scale,reserved code,padding,oversized dimensions all rejected. PASS.
CUDA test_tao_embedding_cuda:validated table uploaded;7 IDs including1024/4096/8191 and repeat,1799elements exactly equal to CPU,finite check PASS. Device kernel assumes IDs/buffers prevalidated (test inputs satisfy this); not public hardened runtime dispatch. CPU decoding remains intended production path;CUDA here verifies reusable embedding lookup for future training.

Storage example only:V8192,D256 codes524288bytes+scales32768bytes=557056bytes (~544KiB), excluding header/object overhead. Equivalent FP32 table8388608bytes. This is representation size,NOT measured speed gain or quality evidence. No new denseVxV Wbi allocated.

## Compatibility and scope
Old Q1,10bit token tape,reader/state gate formats,model weights and live API unchanged. No tokenizer chosen/trained, no dataset retokenized,no QAT/backprop/master-weight pipeline implemented. This does NOT make the current server support8192tokens;it establishes independent next-generation token embedding storage/lookup. Persistent checkpoint migration and binding this interface to a larger-ID reversible state are later architecture work.

CPU build:build_project.bat;GPU build/tests:build_ternary_cuda.bat. Existing uncommitted ternary-foundation work retained;no push this turn.

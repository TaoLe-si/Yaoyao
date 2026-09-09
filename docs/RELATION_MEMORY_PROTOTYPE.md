# Bounded ternary relational memory — explicit semantic prototype

## What was implemented
Header tao_relation_memory.hpp, native CPU tests test_tao_relation_memory.cpp. Bounded slot store keyed by(domain,entity,relation),one latest value per key. Distinct task-constraint domain protected from ordinary observation. Updating a key replaces its prior value and refreshes write age; unrelated keys unchanged. Irrelevant event (caller flag false) does not write or affect eviction. Missing/retracted/evicted query returns std::nullopt,not a fabricated value. All-constraint full memory rejects new entries without mutation;authorized task API may update existing constraints.

## Precision and representation
Semantic uint32 IDs encoded injectively as21 ternary digits,packed into6bytes via shared ternary encoding. Entity,relation,value each use this representation. This is exact symbolic ID encoding,NOT a learned embedding or semantic similarity code. Domain/control metadata and CPU loop indices remain ordinary types. The new token embedding is not connected to this store. No quantized neural weights or float gradients here.

## Retention contract
Finite slot capacity. Evict oldest-written unprotected FACT on insertion when full, never task constraints. Reads do not refresh age. Latest-value semantics intentionally discard overwritten fact; asking previous location is unsupported unless separately modeled as an event. Age is maintained by vector order, no overflowing timestamp in implementation. Bound<=1048576slots;actual memory includes std::vector/object alignment overhead beyond18byte ternary fields. Linear scans/erase,not optimized attention,GEMM,or large-scale associative lookup.

## Task checks actually run
- Same entity+relation overwrites in place logically (physical slot moves to newest);other entity preserved.
-10000 events explicitly tagged irrelevant leave facts unchanged. This tests write gating contract,NOT learned relevance detection.
-Ordinary events cannot overwrite task constraints;explicit set_constraint updates permitted. This is caller API authority separation,NOT security authentication.
-Capacity eviction,full protected memory,retraction,unknown and reset.
-50000 random semantic events compared against independent std::map+age oracle,3200000 fact queries and protected-constraint checks.
-uint32 ID edge/random roundtrip;invalid trit code,padding,overflow and zero capacity rejection.

Demonstration IDs: red-ball location transitions Xiaoming->Xiaohong->cabinet;blue-ball location staysXiaoming. English output labels are supplied by test,not generated language.

## Meaning and limitations
This proves basic memory bookkeeping on explicit semantic events. It does NOT prove the model can parse Chinese,resolve pronouns,identify entities,select relevant facts,obey instructions in generation,or generalize. No natural-language benchmark accuracy claimed. Keeping a constraint in memory does not guarantee decoder compliance. No raw history stored, no recovery of forgotten facts promised. The whole selective store is lossy despite reversible primitive arithmetic elsewhere.

## Next architecture decision
Need learned/event interface that maps contextual input intoentity,relation,value,write/no-write and query;needs uncertainty/conflict handling and distinction between quoted text and task-controller instructions. Current store is a reference oracle/environment for that learning task,not automatic replacement for Tao attention. Decide canonical relational state(e.g.location vs owner) before multi-field atomic updates:one event here changes one relation only. New parser/controller training would need heldout templates/entities and separate end-to-end metrics.

## Isolation
No current model,tokenizer,data,live API modified,no training run. build_project.bat compiles tests. No commit/push performed. Previous ternary/embedding uncommitted work preserved.

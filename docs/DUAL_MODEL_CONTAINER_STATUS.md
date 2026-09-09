# DSM1 container validation status

Round4: test_dual_model_invalid.cpp checks12malformedfiles:5truncations,magic,trailingbyte,layersbounds,zero/infinite scale,11code,nonzero rowpadding. All rejected;validzero matrix roundtrip exact. No GPU/retraining repetition. Generatedfixturefilesnotdeploymentartifacts.

DSM1 still prototype. Header identifies configuration only; no tensor directory, tokenizer fingerprint, integrity checksum or transactional writer. Allocationcap100M floats permits~400MB before bodytruncationdetected,needexpectedlengthcheckbeforeallocation. Finitebitflipvalidfloat isNOTdetected withoutchecksum. Loaderreturnsnewobject notmutatingliveone;save mayleavepartialfileonfailure. Checkpoint DCP0 native-layout notportable/atomic;missingdataRNGmetadata.

Next integrate versioned manifest/checksum/tokenizer binding inoneformat revision and verify complete matrix once;not re-runold training. CPU/CUDA graphandoptimizerstillprototype,fullshape performance/microbatch/data/tokenizer unresolved. Goalactive.

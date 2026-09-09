# Actual training corpus is a pipeline pilot

Confirmed retokenize_pilot.cpp line7 reads build/pilot_train.bin etc and converts same documents into TLP2 with frozen tokenizer; NO larger corpus ingestion. Tokenizer fitting corpus described BPE_CORPUS_MANIFEST.md uses29498conversations/41MB across3shards, but that does NOT train model on those conversations.

Readonly exporter audit: export_language_pilot.cpp selects2percent of one100krow shard, documents report intentionallybiasedpipelinepilot; actual2015train documents657053BPEtokens. Not fullInfinity7M training.

Repair implication: even ifshuffle helps, it cannot replace needed language/knowledge coverage. Next datarepair should build separate larger supervised corpus from local allowed trainpartition with frozen tokenizer and unchangedofficialvalidation exclusions, preserving source row/prompt hashes. No data reexport executed; retokenize_pilot overwrites paths so DO NOT run it onexistingofficialfiles. Newnames required.

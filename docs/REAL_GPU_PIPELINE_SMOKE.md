# Round6 real-data GPU pipeline smoke

Native integrate_gpu_pilot.cu uses existing pilot train doc0, removes known record-level trailing EOS. Processes context from BOS across16position TBPTT chunks until2supervised chunks;464positions18supervised,1AdamWupdate. No CPU training. CUDA forward/backward/update then DCP0checkpoint and DSB2effectivebundle,CPU loadexactweightparity. CPU 8greedyIDs all106 fromBOSonly,not valid assistant-prompt generation or language ability. Loss5.6244603beforeupdate nottrainingtrend.

Deliberatelysmall plumbingfixture L1D4S2M3E8V261,random .12 initialization not approved formal initialization;not formal30Mtraining. Fingerprint64zerosplaceholder,not real tokenizer hash. DCP0 no datasetcursor/RNG yet;saveonlythisrun, resume alreadytestedseparatelysynthetic. Filesbuild/pilot_gpu_step.dcp/.dsb preserved. No performance claim; slowreferenceprojection/transfers. nvcc INFINITYmacro overflowwarningonlydecode mask;shouldreplace numeric_limits<float>::infinity() nextcodechangewithoutretraining.

Remaining actual tokenizer/data sampling, formal init/shape GPUbudget, microbatch scheduling, checkpoint metadata/durability. Mainobjectiveactive.

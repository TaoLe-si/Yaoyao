# Round3 five seeds and hard4token implementation

Originalonline3000stepbudget:addedseed7fails(train diagnostic95.7%,mixednoise32hard68.75%,reverse91.8%,separator75.39);seed999hard100all. Do not claim5seedrobustness at3000. Uniform6000stepbudget fromscratchall5 using optimize_stream_write4_long.cpp;weightsdifferentfiles. Four seeds100percenthardtestallformatsnoise2/32;seed7mixednoise2 99.2188%,noise32 96.875%,others100. Diagnosticpool512 accuracy98.8281%seed7,others100. Budget expansionhelpsbutnotcomplete.

Newtoken_write_hard4.hpp 1065FP64weights,3previousIDs,44->57input,gate comparezero/contentargmax,noexp/allocation. eval_stream_write4 exactfilesizecheck;freshseed913831+syntax,noise128eachside,128seq/syntax/seed. All1920hardreferencecomparisonsagree;four seeds100answersall,seed7mixed94.53125%(7errors),reverse/separator100. Datasetlengthoutside training;no withinbatch exactdedupcheck here.

CPU7roundsx400hotstreams MSVC14.44 /O2 /archAVX2 Ryzen7940H. Medians.115636/.115919/.124959/.135018/.126350us/token. Not directpaired3windowtiming,do not assert speedgain. Weightmemory8520bytes vs3window6856(+1664bytes),oneextra lookup per16hidden plusoneID. Denseanalogue5Dinputvs4D raises inputprojection25percent atsamehiddenwidth;previousLLMcostestimates not automaticallyunchanged.

Next:seed7gate/contentcounterexample mining,heldout devvsfinal data partition before further tuning. Could be sparsegate dead region,accidental crossboundarywrites orfinitecoverage. Do not simplykeepdoublingbudget. Goal remainsactive;model generalization/ternaryQAT and rawlanguage unresolved.

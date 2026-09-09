# Token-window gate and content learning diagnostic

Native diagnose_token_write.cpp. Removes per-event object/owner fields from MODEL input:generator emits[objecttoken,ownertoken,delimiter] per event. Model eachtoken sees queryonehot4 plus threeorderedtokenonehots13each,bias;44input->16tanh->gate sigmoid+8content softmax,857FP64parameters. Window usesdelimiter12 as initialpadding. Query supplied beforestream,ownermemory explicitly8class probabilities;grammar fixedtriplets. No handcoded equality,eventboundary trigger,owner-copy branch or write labels. Generator uses semantic truth ONLY to supply final target.

Memory soft update eachtoken;(1-p)m+p*learnedcontent. TerminalCE only,exactBPTT. Hard inference threshold.5 replaces memory with predictedcontent distribution;not onehot content,not no-softmax inference. Gradient finite differences every17thcoordinate(one sequence/seed),max2.77096920062e-10;not exhaustive.

Train256random sequences/seed noise0..2 eachside,1200SGD lr.15 L2.0001. Fixed protocol acrossseeds,no earlystop. Test256draws/noise0,2,8,32 seed18181+noise. No duplicate audit;short sequences may overlap,not clean split generalization. IDs/types allknown,unseenlength only.

seed42 trainCE.200292462 acc100percent;noise0 soft/hard100/100;2 99.6094/100;8 88.2813/100;32 19.9219/100.
seed123 trainCE.808846173 acc73.4375;noise0 100/100;2 36.3281/35.9375;8 36.3281/36.3281;32 25/31.6406.
seed2026 trainCE.845500248 acc72.6563;noise0 100/100;2 43.3594/43.75;8 35.5469/35.5469;32 34.375/35.9375.

Conclusion:one run learns effective hard gate/content across tested lengths,but2/3underfit;not robust proof learned selective memory. Token-level localization+content learning increases optimization burden;precise cause(gate saturation/content/credit assignment/data budget)not isolated. Soft leakage worsens length even successfulseed. No claim natural-language parsing,pronouns,ternaryQAT,originalchain or reversiblememory. No production/checkpoint modifications. Next diagnose gate/content separately with same tokeninput and controlled budget,do not cherry-pick seed42.

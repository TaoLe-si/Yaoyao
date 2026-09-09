# Grouped projection decoder accepted as faster independent candidate

Full32step logits/states, advance states and generated128trace exact versus two-thread AVX2 reference. AB/BA baseline374.809/367.785 vs grouped479.469/523.670 tps; aggregate371.264->500.596 (~34.8percent gain). Same frozen step360 model, exact row arithmetic, two sleeping-thread participants. Group independent s,m,read projections per layer; state dependencies preserved.

Resident multi-turn/reset comparison: textual output and all token/end/truncation fields identical. Baseline decode379.68/421.75/444.048 vs grouped507.405/504.286/457.536tps. Still degenerate step360 outputs; not quality claim or4000tps attainment.

Standalone executable build/cpu_resident_stage_grouped.exe available; original executables preserved. Captured evidence build/stage_grouped_result.log and build/stage_grouped_resident_comparison.json. CPU decode development independent of GPUtraining. No default training evaluation substitutions.

# Prebound scratch result

32greedy +8forced steps afterreset bitwise logits/states/argmax passed. Sharedweights30,566,400bytes,scratch87,552bytes. Alternating fullgreedy byte388.770/389.533/368.564/375.550 versus prebound384.025/388.017/367.977/341.425positions/s. No demonstrated gain; production unchanged. Removing allocation/map costs alone is not supported as current primary bottleneck. Background load means exact attribution remains limited. Next independent candidate reconstructs +/-scale via integer masks rather than int8-to-float conversion.

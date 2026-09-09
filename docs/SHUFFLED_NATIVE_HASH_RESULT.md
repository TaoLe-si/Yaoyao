# Native checkpoint digest independent test

Parent compiled/ran test_shuffled_native_hash.cu.11binaryinput lengths0,1,55,56,63,64,65,127,128,4097,1000000 matchedexistingWindows cryptographicprovider SHA256 exactly. Tests padding boundaries+embeddedzero/nonASCIIbytes. NoCUDAcalls/noCPUtraining. This validates digestfunction, NOT nativecheckpoint/GPUresume. Realresume test remains separate pending.

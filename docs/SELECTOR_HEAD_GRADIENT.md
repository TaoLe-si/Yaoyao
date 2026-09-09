# Frozen real prediction head / selector CE gradient bridge

Executable test_tao_selector_head.cpp includes actual ServerCore and loads current50002checkpoint,TCG,reader,vocabulary source and token stream. Four overlapping64token windows from beginning of existing train token stream;next shifted token target. Identity state gates,existing Wbi gates. No changes to checkpoint or deployed executable.

Rebuild actual reader256 and hash64. Double precision replica of actual frozen320->320SwiGLU->1024 output and fixed Wbi bias;stable CE. Serving float logit comparison max_abs2.72761346096e-05.

Selector modifies only reader256;hash64 and Wbi bias frozen. Two candidate contents are previous recovered trit states,NOT final aligned learned content. They are a diagnostic injection fixture:dimension matches but semantics/scale not established. alpha=beta=.05 explicit diagnostic setting. This does not settle original token+context candidate design or demonstrate useful retrieval.

Complete numerical path:effective ternary score weights(q*.001) -> joint logits ->soft selected reader ->real frozen SwiGLU+output ->1024CE. Analytic backprop through head,selector,linear action score effective weights. Floating perturbations check derivative with respect to effective weight,NOT derivative of discrete quantization. Scoring uses equivalent expanded doubles,not packed-kernel backward. No optimizer/master weight QAT loop.

Finite differences epsilon1e-5,4examples,input coordinates every17 (19 each),all5selector logits,6effective weight coordinates/example. Maximum errors:head input3.01365377098e-10;selector3.45335274266e-10;effective score weights3.45335274266e-10. Partial coordinate checks,not exhaustive Jacobian.

Results baselineCE / softCE / after one diagnostic logit step / hardCE:
0 target3:2.317117289 /2.393031222 /2.393017718 /2.517856362
1 target77:6.264590186 /6.139253102 /6.139233630 /6.264590186
2 target35:4.336725887 /4.360700471 /4.360696605 /4.314353449
3 target307:6.261195546 /6.110812432 /6.110783111 /6.261195546

Diagnostic line search may fall back to unchanged logits;reported four cases actually decreased softCE. This does not establish generalization or baseline improvement:soft CE worsened2/4,hard worse1/better1/same2. No language quality/coreference claim.

NativeCPU verifier only;no CUDA head backward verification this turn. Build_project compiles executable. No files written by runtime,no production model updates,no dataset retokenization,training run or commit/push. Next: reusable gradient API/QAT optimizer,content alignment,matching projection derivative,heldout evaluation before adoption.

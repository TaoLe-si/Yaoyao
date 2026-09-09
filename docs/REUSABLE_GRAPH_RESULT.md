# Reusable CUDA Graph complete-update result

Implemented fixed 4-slot 256-step forward/loss/backward graph, padded inactive positions, GPU state import/export, shared effective parameter/gradient nodes with existing AdamW. Graph allocations replay with AutoFreeOnLaunch; intermediate gradients reset by captured node construction, initial-state gradients explicitly reset, parameter gradients preserved across eight accumulation rounds. Compile per-thread default stream with asynchronous allocation and D2D.

Restored formal step60: graph build4.783102s, complete update12.911100s excluding checkpoint I/O. 4408positions2344targets, NLL8.2983973 norm.905890. Step61 SCP/DSB byte-identical to existing device batch reference. Continuous run build4.687928s, updates12.738206s and12.281533s. Step62 SCP/DSB byte-identical to existing continuous batch reference, which also matched resumed reference.

Historical same-checkpoint batch update25.543180s, optimized serial38.542271s, original64.810441s; observed graph speed approximately1.98x/2.99x/5.02x respectively, not simultaneous randomized comparison. First build plus first update17.694202s still below25.543180s. No full-device utilization claim.

Controlled graph entry compiled; original STOP causes no-update exit. Step62 native preflight restore passed. Original checkpoint60 and STOP retained. New diagnostic outputs separate. Still requires graph memory/timeline sampling, controlled controller integration and preservation of old output collisions before formal resume. Graph owns allocation pointers through host tape; do not clear tape or export graph pointers as durable session states. Destructor synchronization and graph allocator lifetime merit longer stress validation.

@echo off
setlocal
cd /d "%~dp0.."
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
if errorlevel 1 exit /b 1
if not exist build mkdir build
for %%S in (test_tao_selector_head test_tao_hard_selector test_tao_joint_selector test_tao_local_association test_tao_relation_memory test_tao_ternary_embedding test_tao_ternary tao_d256_api test_tao_server_checkpoint test_tao_server_forward test_tao_state_header test_self_decoding_node eval_tao_state_gates tao_ordered_diagnostics) do (
    cl /nologo /EHsc /O2 /std:c++17 /arch:AVX2 /I src src\%%S.cpp /Fo:build\%%S.obj /Fe:build\%%S.exe
    if errorlevel 1 exit /b 1
)

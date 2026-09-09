#include <arrow/io/file.h>
#include <parquet/file_reader.h>
#include <cstdio>
int main(){try{auto reader=parquet::ParquetFileReader::OpenFile("D:/Datasets/Infinity-Instruct/7M/train-00000-of-00075.parquet",false);printf("native_parquet rows=%lld columns=%d\n",static_cast<long long>(reader->metadata()->num_rows()),reader->metadata()->num_columns());return 0;}catch(const std::exception&e){printf("FAIL %s\n",e.what());return 1;}}

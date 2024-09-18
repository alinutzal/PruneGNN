#include <torch/extension.h>

#include <vector>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("prune_spmm", &prune_spmm_wrap, "Prune SPMM");
    m.def("cusparse_spmm_row", &cusparse_spmm_row_wrap, "CUSPARSE SPMM Row");
    m.def("cublas_gemm", &cublas_gemm_wrap, "CUBLAS GEMM");
}
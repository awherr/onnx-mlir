# SPDX-License-Inxtifier: Apache-2.0

add_onnx_mlir_dialect(Nx nx)
add_onnx_mlir_dialect_doc(nx Nx.td)

add_onnx_mlir_library(NxOps
  NxOps.cpp
  NxTypes.cpp

  DEPENDS
  OMKrnlIncGen
  OMNxIncGen
  OMSpecializedKernelOpInterface

  LINK_LIBS PUBLIC
  OMCompilerOptions
  OMONNXOps
  MLIRLLVMCommonConversion
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRShapeDialect
  )

# SPDX-License-Inxtifier: Apache-2.0

add_onnx_mlir_dialect(NxKernel nxk)
add_onnx_mlir_dialect_doc(nxk NxKernel.td)

add_mlir_dialect_library(NxKernelOps
  NxKernel.cpp
  NxKernelTypes.cpp
  NxKernelOps.cpp
  
  DEPENDS
  OMNxKernelIncGen
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

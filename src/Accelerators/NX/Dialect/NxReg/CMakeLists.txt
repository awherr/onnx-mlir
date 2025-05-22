# SPDX-License-Inxtifier: Apache-2.0

add_onnx_mlir_dialect(NxReg nxk)
add_onnx_mlir_dialect_doc(nxk NxReg.td)

add_onnx_mlir_library(NxRegOps
NxRegOps.cpp
NxRegTypes.cpp

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

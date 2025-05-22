// SPDX-License-Identifier: Apache-2.0

//===--------------------- NxKernel.td - MLIR Operations ---------*- tablegen -===//
//
// Copyright 2025 Intel Labs.
//
// =============================================================================
//
// This file contains TableGen definition of nx operations.
//
// REQUEST:
// When you perform a change to the DEN dialect, please follow the steps below
// to update the NxKernel dialect documentation.
//
// 1) cd onnx-mlir/build
// 2) make onnx-mlir-docs
// 3) the newly build nx.md file resides in the docs/Dialects directory.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "src/Interface/SpecializedKernelOpInterface.td"

def NxKernel_Dialect : Dialect {
  let name = "nx";
  let cppNamespace = "::mlir";
  let usePropertiesForAttributes = 0;
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = [
    "affine::AffineDialect",
    "arith::ArithDialect",
    "func::FuncDialect",
    "linalg::LinalgDialect",
    "math::MathDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "shape::ShapeDialect",
  ];
}

def StringType : Type<CPred<"mlir::isa<nx::StringType>($_self)">, "string type">;

// Require regions to have nx.terminate terminator operation.
def ImplicitNxKernelTerminator : SingleBlockImplicitTerminator<"NxKernelTerminatorOp">;

def NxKernelEndOp : Op<NxKernel_Dialect, "end", [Pure, Terminator, ReturnLike,
    MemRefsNormalizable]> {
  let summary = "";
  let description = [{}];
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

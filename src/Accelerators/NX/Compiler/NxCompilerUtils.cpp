/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NXCompilerUtils.cpp ---------------------===//
//
// Copyright 2025 Intel Labs.
//
// =============================================================================
//
// Compiler Utilities for  NX
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "src/Accelerators/NX/Compiler/NXCompilerOptions.hpp"
#include "src/Accelerators/NX/Compiler/NXCompilerUtils.hpp"
#include "src/Accelerators/NX/Compiler/ZHighDisposableGarbageCollector.hpp"
#include "src/Accelerators/NX/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NX/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NX/Pass/NXPasses.hpp"
#include "src/Accelerators/NX/Support/NXLimit.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "NXCompilerUtils"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

void configurePassesNX() {
  
}

void addPassesNX(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, EmissionTargetType &emissionTarget,
    std::string outputNameNoExt) {
}

} // namespace onnx_mlir

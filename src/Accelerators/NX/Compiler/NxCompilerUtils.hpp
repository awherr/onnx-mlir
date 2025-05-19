/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- NXCompilerUtils.hpp ----------------------===//
//
// Copyright 2025 Intel Labs.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_NX_COMPILER_UTILS_H
#define ONNX_MLIR_NX_COMPILER_UTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"

namespace onnx_mlir {

void addPassesNX(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    std::string outputNameNoExt);

void configurePassesNX();

} // namespace onnx_mlir
#endif

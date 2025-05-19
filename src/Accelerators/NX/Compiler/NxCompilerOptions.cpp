/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- NXCompilerOptions.cpp --------------------===//
//
// Copyright 2025 Intel Labs.
//
// =============================================================================
//
// Compiler Options for NX
//
//===----------------------------------------------------------------------===//
#include "src/Accelerators/NX/Compiler/NXCompilerOptions.hpp"

#define DEBUG_TYPE "NXCompilerOptions"

namespace onnx_mlir {

llvm::cl::opt<NXEmissionTargetType> nxEmissionTarget(
    llvm::cl::desc("[Optional] NX Example Enum Attrib):"),
    llvm::cl::values(
        clEnumVal(ExampleA, "Example A"),
        clEnumVal(ExampleB, "Example B"),
        clEnumVal(ExampleC, "Example C")),
    llvm::cl::init(ExampleA), llvm::cl::cat(OnnxMlirOptions));
} // namespace onnx_mlir

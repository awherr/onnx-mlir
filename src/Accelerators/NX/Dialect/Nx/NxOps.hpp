/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- NxOps.hpp - Nx Operations ------------------===//
//
// Copyright 2025 Intel Labs.
//
// =============================================================================
//
// This file contains declarations of nx operations.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_NX_H
#define ONNX_MLIR_NX_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "src/Accelerators/NX/Dialect/Nx/NxTypes.hpp"

#include "src/Interface/SpecializedKernelOpInterface.hpp"
#include "src/Accelerators/NX/Dialect/Nx/NxDialect.hpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/NX/Dialect/Nx/NxAttributes.hpp.inc"

#define GET_OP_CLASSES
#include "src/Accelerators/NX/Dialect/Nx/NxOps.hpp.inc"
#endif

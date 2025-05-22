/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- NxKernelOps.hpp - NxKernel Operations ------------------===//
//
// Copyright 2025 Intel Labs.
//
// =============================================================================
//
// This file contains declarations of nx operations.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_NX_KERNEL_H
#define ONNX_MLIR_NX_KERNEL_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "src/Accelerators/NX/Dialect/NxKernel/NxKernelTypes.hpp"

#include "src/Interface/SpecializedKernelOpInterface.hpp"

#include "src/Accelerators/NX/Dialect/NxKernel/NxKernelTypes.hpp.inc"

#include "src/Accelerators/NX/Dialect/NxKernel/NxKernelEnums.hpp.inc"
#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/NX/Dialect/NxKernel/NxKernelAttributes.hpp.inc"

#include "src/Accelerators/NX/Dialect/NxKernel/NxKernelDialect.hpp.inc"

#define GET_OP_CLASSES
#include "src/Accelerators/NX/Dialect/NxKernel/NxKernelOps.hpp.inc"

#endif

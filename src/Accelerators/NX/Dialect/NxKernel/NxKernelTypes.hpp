/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- NxKernelTypes.hpp - NxKernel Operations ------------------===//
//
// Copyright 2025 Intel Labs.
//
// =============================================================================
//
// This file contains declarations of nx types.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_NX_KERNEL_TYPES_H
#define ONNX_MLIR_NX_KERNEL_TYPES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace onnx_mlir {
namespace nx {

class KernelType : public mlir::Type::TypeBase<KernelType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr const char *name = "kernel.type";
};

class ModuleType : public mlir::Type::TypeBase<ModuleType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr const char *name = "module.type";
};

class SynapseType : public mlir::Type::TypeBase<SynapseType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr const char *name = "synapse.type";
};

/// Add custom type conversions to convert nx types to the given \p
/// typeConverter.
void customizeTypeConverter(mlir::LLVMTypeConverter &typeConverter);

} // namespace nx
} // namespace onnx_mlir
#endif

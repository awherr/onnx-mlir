/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- NxTypes.hpp - Nx Operations ------------------===//
//
// Copyright 2025 Intel Labs.
//
// =============================================================================
//
// This file contains definition of nx types.
//
//===----------------------------------------------------------------------===//

#include "NxTypes.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace nx {

void customizeTypeConverter(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([&](MemRefType type) -> std::optional<Type> {
    Type elementType = type.getElementType();
    if (!mlir::isa<nx::StringType>(elementType))
      return std::nullopt;

    elementType = mlir::cast<nx::StringType>(elementType)
                      .getLLVMType(type.getContext());
    return typeConverter.convertType(
        MemRefType::get(type.getShape(), elementType));
  });

  typeConverter.addConversion([&](nx::StringType type) -> Type {
    return typeConverter.convertType(type.getLLVMType(type.getContext()));
  });

  typeConverter.addConversion([&](NoneType type) -> Type { return type; });
}

} // namespace nx
} // namespace onnx_mlir

/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------------- NxKernelHelper.hpp - NxKernel Dialect Helper----------------===//
//
// // Copyright 2025 Intel Labs.
//
// =============================================================================
//
// This file implements helper methods to build NxKernel Dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_NXKERNAL_HELPER_H
#define ONNX_MLIR_NXKERNAL_HELPER_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "src/Dialect/Mlir/IndexExpr.hpp"

namespace onnx_mlir {
namespace nx {

struct NxKernelIterateOperandPack {
  NxKernelIterateOperandPack(mlir::Builder &builder,
      llvm::ArrayRef<mlir::Value> inputLoops,
      llvm::ArrayRef<mlir::Value> optimizedLoops)
      : inputLoops(inputLoops), optimizedLoops(optimizedLoops),
        builder(builder) {
    operands.insert(
        operands.end(), optimizedLoops.begin(), optimizedLoops.end());
  }

  void pushConstantBound(int64_t bound);

  void pushOperandBound(mlir::Value operand);

  void pushAffineMapBound(
      mlir::AffineMap map, mlir::ArrayRef<mlir::Value> operands);

  // When used in a lower bound, set isLb to true, when used in an upper bound,
  // set isLb to false.
  void pushIndexExprBound(IndexExpr expr, bool isLb);

  void pushIndexExprsBound(llvm::SmallVectorImpl<IndexExpr> &exprVector);

  llvm::SmallVector<mlir::Value, 8> getOperands() const { return operands; }

  mlir::ArrayAttr getAttributes() const {
    return builder.getArrayAttr(boundMaps);
  }

  size_t getNumOptimizedLoops() const { return optimizedLoops.size(); }

  size_t getNumInputLoops() const { return inputLoops.size(); }

private:
  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::SmallVector<mlir::Attribute, 8> boundMaps;
  llvm::ArrayRef<mlir::Value> inputLoops, optimizedLoops;
  mlir::Builder &builder;
};

} // namespace nx
} // namespace onnx_mlir
#endif

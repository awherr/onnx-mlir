/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- NxAccelerator.hpp ----------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// ===========================================================================
//
// Accelerator support for the IBM Telum coprocessor.
//
//===---------------------------------------------------------------------===//

#ifndef ONNX_MLIR_NX_ACCELERATOR_H
#define ONNX_MLIR_NX_ACCELERATOR_H

#include "mlir/IR/BuiltinTypes.h"
#include "src/Accelerators/Accelerator.hpp"

namespace onnx_mlir {
namespace accel {

/// Singleton class to construct an Nx accelerator.
class NxAccelerator final : public Accelerator {
private:
  static NxAccelerator *instance;
  NxAccelerator();

public:
  NxAccelerator(NxAccelerator &) = delete;
  void operator=(const NxAccelerator &) = delete;

  ~NxAccelerator();

  static NxAccelerator *getInstance();

  uint64_t getVersionNumber() const final;

  //===--------------------------------------------------------------------===//
  // Hooks for onnx-mlir driver
  //===--------------------------------------------------------------------===//
  virtual void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
      std::string outputNameNoExt) const final;
  //===--------------------------------------------------------------------===//
  // Hooks for onnx-mlir-opt driver
  //===--------------------------------------------------------------------===//
  virtual void registerDialects(mlir::DialectRegistry &registry) const final;
  virtual void registerPasses(int optLevel) const final;
  virtual void configurePasses() const final;
 
  virtual mlir::MemRefType convertTensorTypeToMemRefType(
    const mlir::TensorType tensorType) const final{
      return nullptr;
    }

  virtual int64_t getDefaultAllocAlignment(const mlir::TensorType type) const final{
    return -1;
  }

  virtual void conversionTargetONNXToKrnl(
    mlir::ConversionTarget &target) const final{}

  virtual void rewritePatternONNXToKrnl(mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) const final{}

  virtual void conversionTargetKrnlToLLVM(
    mlir::ConversionTarget &target) const final{}

  virtual void rewritePatternKrnlToLLVM(mlir::RewritePatternSet &patterns,
    mlir::LLVMTypeConverter &typeConverter, mlir::MLIRContext *ctx) const final {}
};
} // namespace accel
} // namespace onnx_mlir
#endif
